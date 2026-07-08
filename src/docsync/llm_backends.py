"""Pluggable LLM backends.

docsync's LLM stages (impact judge, edit generation) only depend on a tiny slice
of the Anthropic client surface:

    resp = client.messages.parse(model=..., system=[...], messages=[...],
                                 output_format=SomePydanticModel, **ignored)
    resp.parsed_output          # a validated SomePydanticModel instance

The production backend is `anthropic.Anthropic()`. This module adds two **dev-only**
backends that satisfy the same interface by shelling out to a local CLI in headless
mode: `ClaudeCodeClient` over the `claude` CLI (`claude -p --output-format json`,
reusing your Claude Code authentication) and `CursorClient` over the Cursor CLI
(`cursor-agent -p --output-format json`, reusing your Cursor subscription /
CURSOR_API_KEY). Either lets you run the full pipeline WITHOUT an ANTHROPIC_API_KEY.

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
import re
import shutil
import subprocess
import tempfile
from typing import Any

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# An entire reply wrapped in one ``` fence (any language tag), e.g. the model
# returning ```mdx\n<page>\n```. Anchored start+end so an *internal* fence in the
# document body (a code sample) is never mistaken for the wrapper.
_OUTER_FENCE_RE = re.compile(r"\A\s*```[A-Za-z0-9]*\n(.*?)\n?```\s*\Z", re.DOTALL)

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


def _extract_json(text: str) -> str:
    """Pull the JSON object out of the model's reply (handles ```json fences)."""
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in CLI reply: {text[:200]!r}")
    return text[start : end + 1]


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

    def _run(self, model: str, system: str, prompt: str) -> tuple[str, Any]:
        # Replace Claude Code's default prompt; pass the user prompt via STDIN so a
        # long/multiline prompt is never mis-parsed as CLI args. The system prompt
        # already forbids tools, so a pure JSON completion needs none.
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
              system: Any = None, messages: Any = None, **_ignored: Any) -> _Resp:
        """Mimic anthropic .messages.parse(..., output_format=Model) via the CLI."""
        mdl = model or self._default_model or "claude-opus-4-8"
        text_field = _single_text_field(output_format)
        if text_field is not None:
            return self._parse_text(output_format, text_field, mdl, system, messages)
        return self._parse_json(output_format, mdl, system, messages)

    def _parse_json(self, output_format, mdl, system, messages) -> _Resp:
        """Structured (multi-field) outputs: prompt for one JSON object, validate it."""
        schema = json.dumps(output_format.model_json_schema(), separators=(",", ":"))
        sys_text = (
            _system_text(system)
            + "\n\nYou are a JSON API. Respond with ONLY a single JSON object that "
            "validates against the schema below — no prose, no code fences, no tools.\n"
            f"JSON schema: {schema}"
        )
        user_text = _user_text(messages)

        last_err: Exception | None = None
        for _ in range(2):  # one retry on a parse/validation miss
            raw, usage = self._run(mdl, sys_text, user_text)
            try:
                obj = output_format.model_validate_json(_extract_json(raw))
                return _Resp(obj, usage=usage)
            except (ValueError, ValidationError) as exc:
                last_err = exc
                user_text += "\n\nYour previous reply was not valid JSON for the schema. " \
                             "Return ONLY the JSON object."
        raise RuntimeError(f"{self._label} did not return schema-valid JSON: {last_err}")

    def _parse_text(self, output_format, field, mdl, system, messages) -> _Resp:
        """Whole-document output: take the raw reply as the single string field.

        The model returns the document directly (not JSON), so a full MDX page with
        its own braces and code fences can't corrupt parsing. We strip only an outer
        fence wrapping the entire reply and reject an empty body.
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
            raw, usage = self._run(mdl, sys_text, user_text)
            content = _unwrap_outer_fence(raw)
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

    def _run(self, model: str, system: str, prompt: str) -> tuple[str, Any]:
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
    """Factory: 'api' -> anthropic.Anthropic(); 'claude-code' -> ClaudeCodeClient();
    'cursor' -> CursorClient()."""
    if backend == "claude-code":
        return ClaudeCodeClient()
    if backend == "cursor":
        return CursorClient()
    if backend == "api":
        import anthropic

        return anthropic.Anthropic()
    raise ValueError(f"unknown backend: {backend!r} (use 'api', 'claude-code', or 'cursor')")
