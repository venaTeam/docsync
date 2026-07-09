"""Pluggable LLM backends.

docsync's LLM stages (impact judge, edit generation) only depend on a tiny slice
of the Anthropic client surface:

    resp = client.messages.parse(model=..., system=[...], messages=[...],
                                 output_format=SomePydanticModel, **ignored)
    resp.parsed_output          # a validated SomePydanticModel instance

The production backend is `anthropic.Anthropic()`. This module adds three more that
satisfy the same interface without native structured outputs — they *prompt* for the
schema and validate locally: `GatewayClient` over the Anthropic SDK transport (for
Anthropic-compatible gateways serving non-Anthropic models, which don't enforce
`output_format`), and two **dev-only** CLI backends — `ClaudeCodeClient` over the
`claude` CLI (`claude -p --output-format json`, reusing your Claude Code
authentication) and `CursorClient` over the Cursor CLI (`cursor-agent -p
--output-format json`, reusing your Cursor subscription / CURSOR_API_KEY). The CLI
pair runs the full pipeline WITHOUT an ANTHROPIC_API_KEY.

Caveats (why these are dev-only, not the product path):
  * No schema-enforced structured outputs — we prompt for JSON and validate with
    pydantic ourselves, retrying once on a parse/validation miss.
  * Per-call overhead: the CLIs inject their own system prompt/tooling, so each
    call costs more tokens/latency than a raw API call.
  * They bill your subscription/credits; automated/batch use of subscription auth
    has Terms-of-Service limits. Use for local dogfooding only.
  * `cursor-agent` reports no token usage in its JSON envelope, so cursor runs
    show no cost estimate.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Interleaved reasoning some gateway-served models (MiniMax, DeepSeek-R1 style) emit
# in the *text* reply. Anthropic models return thinking out-of-band, so these tags
# never appear there — but through an Anthropic-compatible gateway they land inline
# and would corrupt JSON extraction / the whole-document reply.
_THINK_BLOCK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"\A\s*<think(?:ing)?>", re.IGNORECASE)
# An entire reply wrapped in one ``` fence (any language tag), e.g. the model
# returning ```mdx\n<page>\n```. Anchored start+end so an *internal* fence in the
# document body (a code sample) is never mistaken for the wrapper.
_OUTER_FENCE_RE = re.compile(r"\A\s*```[A-Za-z0-9]*\n(.*?)\n?```\s*\Z", re.DOTALL)
# A fence wrapping ONLY the frontmatter block (observed weak-model tic:
# ```yaml\n---\ntitle: …\n---\n``` followed by the unfenced body). The closer must
# sit immediately after the frontmatter's closing ---, so the non-greedy body can
# never run past the frontmatter into the page's own code fences. Must be stripped
# BEFORE _OUTER_FENCE_RE: on a body that happens to end with a code block, the
# outer-fence regex would otherwise pair this wrapper's opener with that block's
# closer and corrupt the document.
_FM_FENCE_WRAPPER_RE = re.compile(
    r"\A\s*```[A-Za-z0-9]*[ \t]*\n(---\n.*?\n---)[ \t]*\n```[ \t]*\n?", re.DOTALL
)

# Ceiling on a single CLI call. Generous — Opus authoring calls run for minutes —
# but bounded, so a hung CLI (e.g. blocking on an interactive login prompt) fails
# the call instead of stalling a pipeline worker forever.
_CLI_TIMEOUT_S = 600

# Anthropic model ids -> Cursor-native names (longest-prefix match). Unknown ids
# pass through verbatim so ModelConfig can hold Cursor-native names ("gpt-5",
# "auto") directly. Cheap-tier name unverified against `cursor-agent models`; an
# unrecognized name fails the run with the CLI's own error rather than silently
# rerouting.
_CURSOR_MODEL_MAP = {
    "claude-opus-4": "opus",
    "claude-sonnet-4": "sonnet-4.5",
    "claude-haiku-4": "haiku-4.5",
}


def _cursor_model(model: str) -> str:
    """Translate an Anthropic model id to its Cursor-native name."""
    best = ""
    for prefix in _CURSOR_MODEL_MAP:
        if model.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return _CURSOR_MODEL_MAP[best] if best else model


def _system_text(system: Any) -> str:
    """Flatten the SDK `system` arg (str or list of text blocks) to plain text."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "\n\n".join(p for p in parts if p)


def _user_text(messages: Any) -> str:
    """Flatten the SDK `messages` arg to the concatenated user text."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n\n".join(parts)


def _strip_reasoning(text: str) -> str:
    """Drop inline <think>/<thinking> reasoning blocks from a reply.

    A reply that is *only* an unterminated reasoning block (the model hit its token
    cap mid-think) strips to "" — the callers' empty/parse checks then trigger their
    retry rather than validating reasoning prose as the answer.
    """
    stripped = _THINK_BLOCK_RE.sub("", text)
    if _THINK_OPEN_RE.match(stripped):
        return ""
    return stripped.strip()


def _extract_json(text: str) -> str:
    """Pull the first complete JSON object out of the model's reply.

    Handles ```json fences, then scans from the first ``{`` with brace/string
    awareness, so trailing prose — which may itself contain braces — can't be
    swept into the candidate the way a last-``}`` slice would.
    """
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in CLI reply: {text[:200]!r}")
    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"unterminated JSON object in CLI reply: {text[:200]!r}")


def _single_text_field(model: type[BaseModel]) -> str | None:
    """Name of the sole required `str` field, or None.

    Identifies "whole-document" output models like `AuthoredPage(content: str)`,
    where coaxing the model to emit the document *inside JSON* is fragile (the
    document itself contains braces, quotes, and code fences). For those we take the
    raw reply as the field value instead of parsing JSON.
    """
    required = [n for n, f in model.model_fields.items() if f.is_required()]
    if len(required) != 1:
        return None
    name = required[0]
    return name if model.model_fields[name].annotation is str else None


def _unwrap_outer_fence(text: str) -> str:
    """Strip a single ``` fence wrapping the *entire* reply, then trim surrounding space."""
    m = _OUTER_FENCE_RE.match(text)
    return (m.group(1) if m else text).strip()


def _unwrap_frontmatter_fence(text: str) -> str:
    """Strip a ``` fence wrapping only the leading frontmatter block, if present."""
    return _FM_FENCE_WRAPPER_RE.sub(lambda m: m.group(1) + "\n", text, count=1)


def _iter_dicts(node: Any):
    """Yield every dict in a parsed-JSON tree, depth-first in document order."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dicts(value)


def _harvest_target(model: type[BaseModel]) -> tuple[str, type[BaseModel]] | None:
    """(field, item model) when `model` has exactly one list-of-BaseModel field.

    Identifies "collection" output models like `DocPlan(pages: list[PlannedPage])`,
    whose payload can be harvested item-by-item out of a wrong-shaped reply.
    """
    hits: list[tuple[str, type[BaseModel]]] = []
    for name, field in model.model_fields.items():
        if get_origin(field.annotation) is list:
            (item,) = get_args(field.annotation) or (None,)
            if isinstance(item, type) and issubclass(item, BaseModel):
                hits.append((name, item))
    return hits[0] if len(hits) == 1 else None


def _salvage(data: Any, output_format: type[BaseModel]) -> BaseModel | None:
    """Rescue a near-miss reply: valid JSON whose *shape* wandered off the schema.

    Weak-schema models behind a gateway return the right payload under invented
    wrappers/nesting (observed: ``{"platform": …, "docPlan": {"getting-started":
    {"pages": […]}}, "total_pages": …}`` for a ``DocPlan``). For collection models
    (one list-of-BaseModel field) harvest every dict anywhere in the tree that
    validates as the item model — this collects across ALL invented sections, where
    a first-nested-match would keep only one. Other models fall back to the first
    nested dict that validates whole. Returns None when nothing salvages.
    """
    target = _harvest_target(output_format)
    if target is not None:
        field, item_model = target
        items: list[BaseModel] = []
        seen: set[str] = set()
        for node in _iter_dicts(data):
            try:
                item = item_model.model_validate(node)
            except ValidationError:
                continue
            key = item.model_dump_json()
            if key not in seen:
                seen.add(key)
                items.append(item)
        if items:
            try:
                return output_format.model_validate({field: [i.model_dump() for i in items]})
            except ValidationError:
                return None
        return None
    for node in _iter_dicts(data):
        if node is data:
            continue
        try:
            return output_format.model_validate(node)
        except ValidationError:
            continue
    return None


def _empty_collection(obj: BaseModel, output_format: type[BaseModel]) -> bool:
    """True when a collection model validated with its list field empty."""
    target = _harvest_target(output_format)
    return target is not None and not getattr(obj, target[0])


def _salvage_text(candidate: str, output_format: type[BaseModel]):
    """`_salvage` over a raw JSON candidate string; None when it isn't parseable JSON."""
    try:
        data = json.loads(candidate)
    except ValueError:
        return None
    return _salvage(data, output_format)


# Raw-reply observability for opaque gateways: when DOCSYNC_LLM_DEBUG names a
# directory, every CLI call's command, prompt, and raw stdout/stderr is written
# there — the only way to see what a gateway-served model actually returned when
# a stage silently drops everything. Best-effort: a dump failure never fails a call.
_DEBUG_DIR_ENV = "DOCSYNC_LLM_DEBUG"
_dump_lock = threading.Lock()
_dump_seq = 0


def _dump_call(label: str, request: str, prompt: str, output: str, errors: str = "") -> None:
    dump_dir = os.environ.get(_DEBUG_DIR_ENV)
    if not dump_dir:
        return
    global _dump_seq
    with _dump_lock:
        _dump_seq += 1
        seq = _dump_seq
    try:
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        body = (
            f"# {label} call {seq}\n\n{request}\n\n"
            f"## prompt\n{prompt}\n\n"
            f"## output\n{output}\n"
        ) + (f"\n## stderr\n{errors}\n" if errors else "")
        (out / f"call-{seq:03d}.txt").write_text(body, encoding="utf-8")
    except OSError:
        pass


class _Resp:
    def __init__(self, parsed_output: BaseModel, usage: Any = None):
        self.parsed_output = parsed_output
        # The CLI envelope's token usage (dict), so MeteredClient can account for it.
        self.usage = usage


class _Messages:
    _label = "claude CLI"  # backend name for error messages

    def __init__(self, default_model: str | None, cli: str, extra_args: list[str]):
        self._default_model = default_model
        self._cli = cli
        self._extra_args = extra_args

    def _run(self, model: str, system: str, prompt: str,
             max_tokens: int | None = None) -> tuple[str, Any]:
        # Replace Claude Code's default prompt; pass the user prompt via STDIN so a
        # long/multiline prompt is never mis-parsed as CLI args. The system prompt
        # already forbids tools, so a pure JSON completion needs none. `max_tokens`
        # is unused here — the CLI has no such flag; the SDK transport needs it.
        cmd = [
            self._cli, "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt", system,
            *self._extra_args,
        ]
        return self._invoke(cmd, prompt)

    def _invoke(self, cmd: list[str], prompt: str, cwd: str | None = None) -> tuple[str, Any]:
        """Run the CLI and unpack its JSON envelope -> (result text, usage or None)."""
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                cwd=cwd, timeout=_CLI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{self._label} timed out after {_CLI_TIMEOUT_S}s")
        _dump_call(
            self._label,
            f"command: {json.dumps(cmd)}\nreturncode: {proc.returncode}",
            prompt, proc.stdout, errors=proc.stderr,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self._label} failed ({proc.returncode}): {proc.stderr.strip()}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(f"{self._label} returned non-JSON output: {proc.stdout[:200]!r}")
        if envelope.get("is_error"):
            raise RuntimeError(f"{self._label} error: {envelope.get('result')}")
        return envelope.get("result", ""), envelope.get("usage")

    def parse(self, *, output_format: type[BaseModel], model: str | None = None,
              system: Any = None, messages: Any = None, max_tokens: int | None = None,
              **_ignored: Any) -> _Resp:
        """Mimic anthropic .messages.parse(..., output_format=Model), prompted."""
        mdl = model or self._default_model or "claude-opus-4-8"
        text_field = _single_text_field(output_format)
        if text_field is not None:
            return self._parse_text(output_format, text_field, mdl, system, messages, max_tokens)
        return self._parse_json(output_format, mdl, system, messages, max_tokens)

    def _parse_json(self, output_format, mdl, system, messages, max_tokens=None) -> _Resp:
        """Structured (multi-field) outputs: prompt for one JSON object, validate it.

        Weak-schema models need more than the schema: the exact top-level keys are
        spelled out (invented wrappers are the dominant failure), a validation miss
        feeds the actual error back into the retry, and a reply that is valid JSON
        in the wrong shape goes through `_salvage` before burning a retry.
        """
        schema = json.dumps(output_format.model_json_schema(), separators=(",", ":"))
        keys = ", ".join(f'"{k}"' for k in output_format.model_json_schema().get("properties", {}))
        sys_text = (
            _system_text(system)
            + "\n\nYou are a JSON API. Respond with ONLY a single JSON object that "
            "validates against the schema below — no prose, no code fences, no tools.\n"
            f"The top-level object must have exactly the key(s): {keys}. Do NOT invent "
            "other keys, wrapper objects, or extra nesting.\n"
            f"JSON schema: {schema}"
        )
        user_text = _user_text(messages)

        last_err: Exception | None = None
        for _ in range(3):  # two retries on a parse/validation miss
            raw, usage = self._run(mdl, sys_text, user_text, max_tokens)
            cleaned = _strip_reasoning(raw)
            obj = None
            try:
                candidate = _extract_json(cleaned)
            except ValueError as exc:
                candidate, last_err = None, exc
            if candidate is not None:
                try:
                    obj = output_format.model_validate_json(candidate)
                except ValidationError as exc:
                    last_err = exc
                # Salvage when validation missed — and ALSO when it "succeeded" with an
                # empty collection: extra keys are ignored and list fields default, so a
                # hallucinated wrapper shape validates as an empty DocPlan while the real
                # items sit nested inside the reply.
                if obj is None or _empty_collection(obj, output_format):
                    rescued = _salvage_text(candidate, output_format)
                    if rescued is not None:
                        return _Resp(rescued, usage=usage)
                if obj is not None:
                    return _Resp(obj, usage=usage)
            user_text += (
                f"\n\nYour previous reply did not validate: {str(last_err)[:300]}\n"
                f"Return ONLY one JSON object whose top-level key(s) are exactly: {keys}."
            )
        raise RuntimeError(f"{self._label} did not return schema-valid JSON: {last_err}")

    def _parse_text(self, output_format, field, mdl, system, messages, max_tokens=None) -> _Resp:
        """Whole-document output: take the raw reply as the single string field.

        The model returns the document directly (not JSON), so a full MDX page with
        its own braces and code fences can't corrupt parsing. We strip only wrapper
        fences — around the leading frontmatter block or around the entire reply,
        in that order — and reject an empty body.
        """
        sys_text = (
            _system_text(system)
            + "\n\nRespond with ONLY the complete document content — the exact text of "
            "the file, starting at its first character. No commentary, no JSON, and do "
            "NOT wrap the whole document in a code fence."
        )
        user_text = _user_text(messages)

        last_err: Exception | None = None
        for _ in range(2):
            raw, usage = self._run(mdl, sys_text, user_text, max_tokens)
            content = _unwrap_outer_fence(_unwrap_frontmatter_fence(_strip_reasoning(raw)))
            try:
                if not content.strip():
                    raise ValueError("empty document reply")
                return _Resp(output_format.model_validate({field: content}), usage=usage)
            except (ValueError, ValidationError) as exc:
                last_err = exc
                user_text += "\n\nYour previous reply was empty or malformed. Return the " \
                             "full document text now."
        raise RuntimeError(f"{self._label} did not return a usable document: {last_err}")


class _CursorMessages(_Messages):
    """`cursor-agent` variant: no --system-prompt flag (the system text is prepended
    to the stdin prompt in a delimited block) and no usage in the JSON envelope."""

    _label = "cursor-agent"

    def __init__(self, default_model: str | None, cli: str, extra_args: list[str],
                 workdir: str):
        super().__init__(default_model, cli, extra_args)
        self._workdir = workdir

    def _run(self, model: str, system: str, prompt: str,
             max_tokens: int | None = None) -> tuple[str, Any]:
        # Never pass --force (it auto-approves the agent's commands); run in the
        # client's empty temp cwd so there is no workspace to read or edit. Prompt
        # via STDIN, as with the claude backend (print mode is inferred from the
        # pipe; fall back to a positional prompt arg if a CLI version rejects it).
        cmd = [
            self._cli, "-p",
            "--output-format", "json",
            "--model", _cursor_model(model),
            *self._extra_args,
        ]
        full_prompt = f"<system-instructions>\n{system}\n</system-instructions>\n\n{prompt}"
        text, _ = self._invoke(cmd, full_prompt, cwd=self._workdir)
        return text, None  # Cursor's envelope carries no usage


class _SdkMessages(_Messages):
    """Anthropic-SDK transport for the prompted-JSON engine (the `gateway` backend).

    Plain `messages.create` completions — no native structured outputs, no CLI
    process. Inherits the whole prompted parse machinery (schema + top-level-keys
    prompt, reasoning stripping, salvage, error-informed retries) from `_Messages`.
    """

    _label = "gateway"
    _DEFAULT_MAX_TOKENS = 8192  # `messages.create` requires max_tokens; parse may omit it

    def __init__(self, default_model: str | None, client: Any):
        self._default_model = default_model
        self._client = client

    def _run(self, model: str, system: str, prompt: str,
             max_tokens: int | None = None) -> tuple[str, Any]:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens or self._DEFAULT_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(block, "text", "")
            for block in resp.content
            if getattr(block, "type", "text") == "text"
        )
        _dump_call(self._label, f"model: {model}", prompt, text)
        return text, getattr(resp, "usage", None)


class GatewayClient:
    """Backend for Anthropic-compatible gateways serving NON-Anthropic models.

    Same transport and env wiring as the `api` backend (`anthropic.Anthropic()`,
    honoring ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY) but structured output is
    *prompted* and validated locally instead of relying on the API's native
    `output_format` enforcement — which such gateways typically don't implement
    (the served model returns prose or near-miss JSON and the SDK-side validation
    fails). Use `api` for real Anthropic endpoints; `gateway` when the endpoint
    serves e.g. MiniMax/DeepSeek behind the Anthropic wire format.
    """

    def __init__(self, default_model: str | None = None, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.messages = _SdkMessages(default_model, client)


class ClaudeCodeClient:
    """Dev backend: drop-in for `anthropic.Anthropic()` over the `claude` CLI.

    Usage:
        from docsync.llm_backends import ClaudeCodeClient
        result = pipeline.run(diff, docs_repo, config, manifest, client=ClaudeCodeClient())
    """

    def __init__(self, default_model: str | None = None, cli: str | None = None,
                 extra_args: list[str] | None = None):
        resolved = cli or shutil.which("claude")
        if not resolved:
            raise RuntimeError("`claude` CLI not found on PATH — install Claude Code.")
        self.messages = _Messages(default_model, resolved, extra_args or [])


class CursorClient:
    """Dev backend: drop-in for `anthropic.Anthropic()` over the Cursor CLI.

    Auth comes from `cursor-agent login` or CURSOR_API_KEY. cursor-agent is a coding
    *agent* with tools; this client keeps it a pure text generator: the parse system
    suffixes forbid tools, every call runs in a fresh empty temp cwd (no workspace to
    read or edit), and --force is never passed. The CLI reports no token usage, so
    metered runs show no cost estimate.
    """

    def __init__(self, default_model: str | None = None, cli: str | None = None,
                 extra_args: list[str] | None = None):
        resolved = cli or shutil.which("cursor-agent")
        if not resolved:
            raise RuntimeError(
                "`cursor-agent` CLI not found on PATH — install the Cursor CLI and "
                "authenticate (`cursor-agent login` or CURSOR_API_KEY)."
            )
        # Kept for the client's lifetime; cleaned up when the client is collected.
        self._workdir = tempfile.TemporaryDirectory(prefix="docsync-cursor-")
        self.messages = _CursorMessages(
            default_model, resolved, extra_args or [], workdir=self._workdir.name
        )


def get_client(backend: str):
    """Factory: 'api' -> anthropic.Anthropic(); 'gateway' -> GatewayClient();
    'claude-code' -> ClaudeCodeClient(); 'cursor' -> CursorClient()."""
    if backend == "claude-code":
        return ClaudeCodeClient()
    if backend == "cursor":
        return CursorClient()
    if backend == "gateway":
        return GatewayClient()
    if backend == "api":
        import anthropic

        return anthropic.Anthropic()
    raise ValueError(
        f"unknown backend: {backend!r} (use 'api', 'gateway', 'claude-code', or 'cursor')"
    )
