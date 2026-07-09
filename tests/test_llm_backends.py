"""Unit tests for the CLI dev backends (claude-code, cursor) — no real CLI calls."""

from __future__ import annotations

import json

import pytest

from docsync import llm_backends as be
from docsync.models import AuthoredPage, DocPlan, JudgeVerdict


def test_extract_json_handles_fences_and_prose():
    assert json.loads(be._extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(be._extract_json('Here you go: {"a": 2} done')) == {"a": 2}
    with pytest.raises(ValueError):
        be._extract_json("no json here")


def test_extract_json_stops_at_the_balanced_close():
    # Trailing prose containing a stray `}` must not be swept into the candidate.
    assert json.loads(be._extract_json('{"a": {"b": 1}} and a stray } here')) == {"a": {"b": 1}}
    # Braces inside JSON strings don't affect the balance.
    assert json.loads(be._extract_json('{"a": "curly } brace"} trailing')) == {"a": "curly } brace"}
    with pytest.raises(ValueError):
        be._extract_json('{"a": 1')  # unterminated


def test_strip_reasoning_removes_think_blocks():
    # Closed blocks (any case, <think> or <thinking>) are removed wherever they sit.
    assert be._strip_reasoning('<think>plan {with} braces</think>\n{"a": 1}') == '{"a": 1}'
    assert be._strip_reasoning('<THINKING>hm</THINKING>doc') == "doc"
    # A reply that is only an unterminated reasoning block strips to "" (-> retry).
    assert be._strip_reasoning("<think>never closed {") == ""
    # Replies without reasoning tags pass through (trimmed).
    assert be._strip_reasoning('  {"a": 1} ') == '{"a": 1}'


def test_system_and_user_flattening():
    sys = [{"type": "text", "text": "be terse"}, {"type": "text", "text": "rule two"}]
    assert be._system_text(sys) == "be terse\n\nrule two"
    assert be._system_text("plain") == "plain"
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert be._user_text(msgs) == "hello"
    assert be._user_text([{"role": "user", "content": "str form"}]) == "str form"


def _fake_envelope(result_text: str) -> str:
    return json.dumps({"type": "result", "is_error": False, "result": result_text})


def test_parse_returns_validated_output(monkeypatch):
    calls = {}

    def fake_run(cmd, input, **kwargs):
        calls["cmd"] = cmd
        calls["input"] = input
        out = _fake_envelope('```json\n{"page_path":"p.mdx","affected":true,'
                             '"confidence":0.8,"reason":"r"}\n```')
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")

    client = be.ClaudeCodeClient(default_model="claude-haiku-4-5")
    resp = client.messages.parse(
        output_format=JudgeVerdict,
        model="claude-haiku-4-5",
        system=[{"type": "text", "text": "judge"}],
        messages=[{"role": "user", "content": "is it affected?"}],
        thinking={"type": "adaptive"},  # extra kwargs must be ignored
    )
    assert isinstance(resp.parsed_output, JudgeVerdict)
    assert resp.parsed_output.affected is True
    # model + headless flags were passed through; the prompt goes via stdin
    assert "--model" in calls["cmd"] and "claude-haiku-4-5" in calls["cmd"]
    assert "-p" in calls["cmd"] and "--output-format" in calls["cmd"]
    assert "is it affected?" in calls["input"]


def test_parse_retries_then_raises_on_bad_json(monkeypatch):
    attempts = {"n": 0}

    def fake_run(cmd, input, **kwargs):
        attempts["n"] += 1
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope("not json"),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    with pytest.raises(RuntimeError):
        client.messages.parse(
            output_format=JudgeVerdict, system="s",
            messages=[{"role": "user", "content": "q"}],
        )
    assert attempts["n"] == 3  # two retries


def test_single_text_field_detection():
    # AuthoredPage has one required str field -> whole-document path.
    assert be._single_text_field(AuthoredPage) == "content"
    # JudgeVerdict is multi-field -> JSON path (None).
    assert be._single_text_field(JudgeVerdict) is None


def test_unwrap_outer_fence_only_strips_the_wrapper():
    # A reply wrapped entirely in one fence is unwrapped.
    assert be._unwrap_outer_fence("```mdx\n# Title\nbody\n```") == "# Title\nbody"
    # A document with an *internal* code fence is left intact (not starting with ```).
    page = "---\ntitle: X\n---\n\nText\n\n```py\ncode\n```\n"
    assert be._unwrap_outer_fence(page) == page.strip()


def test_unwrap_frontmatter_fence_strips_the_yaml_wrapper():
    # A ```yaml fence around only the frontmatter is removed; the body's own
    # fences are untouched.
    wrapped = (
        "```yaml\n---\ntitle: X\ndescription: d\n---\n```\n\n"
        "# Body\n\n```bash\necho hi\n```\n\nmore text\n"
    )
    out = be._unwrap_frontmatter_fence(wrapped)
    assert out.startswith("---\ntitle: X")
    assert "```yaml" not in out
    assert "```bash\necho hi\n```" in out
    assert out.count("```") == 2  # only the body's real fence pair remains


def test_unwrap_frontmatter_fence_noop_on_clean_pages():
    # A proper page whose body contains a ```yaml code SAMPLE is byte-identical
    # (the wrapper regex is anchored to the start of the reply).
    page = "---\ntitle: X\ndescription: d\n---\n\nExample:\n\n```yaml\nkey: value\n```\n"
    assert be._unwrap_frontmatter_fence(page) == page
    # No frontmatter at all — nothing to unwrap.
    assert be._unwrap_frontmatter_fence("# Just a body\n") == "# Just a body\n"
    # A whole-reply wrapper (closer at end-of-text, not after ---) is not ours;
    # it stays intact for _unwrap_outer_fence to handle.
    whole = "```mdx\n---\ntitle: X\n---\n\nbody\n```"
    assert be._unwrap_frontmatter_fence(whole) == whole


def test_parse_text_field_takes_raw_document(monkeypatch):
    # The model returns a full MDX page (NOT JSON); it must pass through verbatim
    # even though it contains braces (cols={2}) and an internal code fence.
    page = (
        "---\ntitle: Alerts\ndescription: ref\n---\n\n"
        "<CardGroup cols={2}>\n  <Card>x</Card>\n</CardGroup>\n\n"
        "```python\nget_alerts()\n```\n"
    )

    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope(page),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient(default_model="claude-opus-4-8")
    resp = client.messages.parse(
        output_format=AuthoredPage, system="author", messages=[{"role": "user", "content": "write it"}]
    )
    assert isinstance(resp.parsed_output, AuthoredPage)
    assert resp.parsed_output.content == page.strip()
    assert "cols={2}" in resp.parsed_output.content  # braces didn't corrupt parsing
    assert "```python\nget_alerts()\n```" in resp.parsed_output.content  # internal fence intact


def test_parse_text_field_unwraps_fenced_reply(monkeypatch):
    page = "---\ntitle: X\ndescription: y\n---\n\nBody text here.\n"

    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope(f"```mdx\n{page}\n```"),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    resp = client.messages.parse(
        output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.parsed_output.content == page.strip()


def test_parse_text_field_unwraps_frontmatter_fence_before_outer(monkeypatch):
    # Regression: a ```yaml-fenced frontmatter on a document whose body ENDS with a
    # real code block. _OUTER_FENCE_RE alone would pair the yaml opener with that
    # block's closer (its \Z anchor sees a fence at end-of-text) and corrupt the
    # page; the frontmatter unwrap must run first and defuse it.
    reply = (
        "```yaml\n---\ntitle: X\ndescription: y\n---\n```\n\n"
        "# Body\n\n```bash\necho hi\n```\n"
    )

    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope(reply),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    resp = client.messages.parse(
        output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
    )
    content = resp.parsed_output.content
    assert content.startswith("---\ntitle: X")
    assert "```bash\necho hi\n```" in content  # last block kept its closer
    assert content.count("```") == 2  # wrapper markers gone, real pair intact


def test_parse_text_field_retries_then_raises_on_empty(monkeypatch):
    attempts = {"n": 0}

    def fake_run(cmd, input, **kwargs):
        attempts["n"] += 1
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope("   "), "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    with pytest.raises(RuntimeError):
        client.messages.parse(
            output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
        )
    assert attempts["n"] == 2


def test_parse_json_survives_inline_reasoning(monkeypatch):
    # Gateway-served models (MiniMax style) emit <think> blocks inline in the text
    # reply; the braces inside must not corrupt JSON extraction.
    reply = (
        '<think>The page {p.mdx} looks affected because {reasons}...</think>\n'
        '{"page_path":"p.mdx","affected":true,"confidence":0.9,"reason":"r"}'
    )

    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope(reply), "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    resp = client.messages.parse(
        output_format=JudgeVerdict, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.parsed_output.affected is True


def test_parse_text_field_strips_reasoning_prefix(monkeypatch):
    # A <think> preamble before the document must not land in the page (it would
    # fail the frontmatter gate downstream); the fenced document behind it survives.
    page = "---\ntitle: X\ndescription: y\n---\n\nBody text here.\n"

    def fake_run(cmd, input, **kwargs):
        out = _fake_envelope(f"<think>outline the page...</think>\n```mdx\n{page}\n```")
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    resp = client.messages.parse(
        output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.parsed_output.content == page.strip()


def test_debug_dump_writes_call_transcripts(monkeypatch, tmp_path):
    dump_dir = tmp_path / "llm-debug"
    monkeypatch.setenv("DOCSYNC_LLM_DEBUG", str(dump_dir))

    def fake_run(cmd, input, **kwargs):
        out = _fake_envelope('{"page_path":"p.mdx","affected":true,"confidence":0.8,"reason":"r"}')
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": "warn: x"})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    client.messages.parse(
        output_format=JudgeVerdict, system="s", messages=[{"role": "user", "content": "the ask"}]
    )
    dumps = sorted(dump_dir.glob("call-*.txt"))
    assert len(dumps) == 1
    body = dumps[0].read_text()
    # The transcript carries everything needed to diagnose an opaque gateway reply:
    # the stdin prompt, the raw stdout envelope, and stderr.
    assert "the ask" in body and "p.mdx" in body and "warn: x" in body


def test_get_client_unknown_backend():
    with pytest.raises(ValueError, match="'cursor'"):
        be.get_client("nope")


# --- near-miss salvage (weak-schema models) --------------------------------------

# The exact wrapper shape observed from MiniMax behind an Anthropic-compatible
# gateway: the real pages exist, nested under invented keys and split by section.
_WRONG_SCHEMA_PLAN = {
    "platform": "keep",
    "docPlan": {
        "getting-started": {"pages": [
            {"page_path": "getting-started/introduction.mdx", "title": "Introduction",
             "kind": "guide", "section": "Getting Started"},
        ]},
        "reference": {"pages": [
            {"page_path": "reference/alerts.mdx", "title": "Alerts API",
             "kind": "reference", "section": "Reference"},
        ]},
    },
    "total_pages": 36,
}


def test_salvage_harvests_pages_across_invented_sections():
    plan = be._salvage(_WRONG_SCHEMA_PLAN, DocPlan)
    assert plan is not None
    # BOTH sections' pages are collected — a first-nested-match would keep only one.
    assert [p.page_path for p in plan.pages] == [
        "getting-started/introduction.mdx", "reference/alerts.mdx",
    ]
    assert plan.pages[0].kind == "guide" and plan.pages[1].kind == "reference"


def test_salvage_finds_nested_dict_for_non_collection_models():
    wrapped = {"result": {"page_path": "p.mdx", "affected": True,
                          "confidence": 0.7, "reason": "r"}}
    verdict = be._salvage(wrapped, JudgeVerdict)
    assert verdict is not None and verdict.affected is True


def test_salvage_returns_none_on_unrescuable_json():
    assert be._salvage({"totally": "unrelated"}, DocPlan) is None
    assert be._salvage({"nested": {"also": "unrelated"}}, JudgeVerdict) is None


def test_parse_json_salvages_wrong_schema_reply(monkeypatch):
    # Attempt 1: the JSON instruction is ignored outright (markdown). Attempt 2:
    # JSON in a hallucinated wrapper shape — salvage must rescue it, no third call.
    replies = iter([
        "# Documentation Plan\n\n## Getting Started\n- introduction\n",
        json.dumps(_WRONG_SCHEMA_PLAN),
    ])
    attempts = {"n": 0}

    def fake_run(cmd, input, **kwargs):
        attempts["n"] += 1
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope(next(replies)),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    resp = client.messages.parse(
        output_format=DocPlan, system="plan", messages=[{"role": "user", "content": "q"}]
    )
    assert attempts["n"] == 2
    assert {p.page_path for p in resp.parsed_output.pages} == {
        "getting-started/introduction.mdx", "reference/alerts.mdx",
    }


def test_json_prompt_names_top_level_keys_and_feeds_back_errors(monkeypatch):
    calls = []

    def fake_run(cmd, input, **kwargs):
        calls.append((cmd, input))
        return type("P", (), {"returncode": 0, "stdout": _fake_envelope("not json"),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/claude")
    client = be.ClaudeCodeClient()
    with pytest.raises(RuntimeError):
        client.messages.parse(
            output_format=DocPlan, system="s", messages=[{"role": "user", "content": "q"}]
        )
    # The system prompt spells out the exact top-level keys, not just the schema.
    first_cmd = calls[0][0]
    system_arg = first_cmd[first_cmd.index("--system-prompt") + 1]
    assert 'exactly the key(s): "pages"' in system_arg
    # The retry user prompt carries the actual failure, not a bare "not valid JSON".
    assert "did not validate" in calls[1][1]


# --- gateway backend (Anthropic SDK transport, prompted JSON) ---------------------


class _FakeSdkClient:
    """Stub for `anthropic.Anthropic()`: `.messages.create` pops scripted replies."""

    def __init__(self, replies: list[str], usage=None):
        self._replies = list(replies)
        self._usage = usage
        self.calls: list[dict] = []
        self.messages = self  # expose .messages.create on the same object

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("B", (), {"type": "text", "text": self._replies.pop(0)})()
        return type("R", (), {"content": [block], "usage": self._usage})()


def test_gateway_parse_returns_validated_output():
    sdk = _FakeSdkClient(
        ['{"page_path":"p.mdx","affected":true,"confidence":0.8,"reason":"r"}'],
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    client = be.GatewayClient(client=sdk)
    resp = client.messages.parse(
        output_format=JudgeVerdict,
        model="minimax-2.7",
        system=[{"type": "text", "text": "judge"}],
        messages=[{"role": "user", "content": "is it affected?"}],
        max_tokens=1234,
        thinking={"type": "adaptive"},  # extra kwargs must be ignored
    )
    assert isinstance(resp.parsed_output, JudgeVerdict)
    assert resp.usage == {"input_tokens": 10, "output_tokens": 5}
    call = sdk.calls[0]
    # The model id passes through verbatim; max_tokens is threaded to create().
    assert call["model"] == "minimax-2.7" and call["max_tokens"] == 1234
    assert "judge" in call["system"] and "is it affected?" in call["messages"][0]["content"]


def test_gateway_defaults_max_tokens_when_parse_omits_it():
    sdk = _FakeSdkClient(['{"page_path":"p.mdx","affected":false,"confidence":0.1,"reason":"r"}'])
    be.GatewayClient(client=sdk).messages.parse(
        output_format=JudgeVerdict, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert sdk.calls[0]["max_tokens"] == be._SdkMessages._DEFAULT_MAX_TOKENS


def test_gateway_whole_document_path():
    page = "---\ntitle: X\ndescription: y\n---\n\nBody text here.\n"
    sdk = _FakeSdkClient([f"<think>outline</think>\n```mdx\n{page}\n```"])
    resp = be.GatewayClient(client=sdk).messages.parse(
        output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.parsed_output.content == page.strip()


def test_gateway_retries_then_raises_on_prose():
    sdk = _FakeSdkClient(["# markdown, not JSON"] * 3)
    with pytest.raises(RuntimeError, match="gateway"):
        be.GatewayClient(client=sdk).messages.parse(
            output_format=JudgeVerdict, system="s", messages=[{"role": "user", "content": "q"}]
        )
    assert len(sdk.calls) == 3


def test_get_client_gateway_returns_gateway_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert isinstance(be.get_client("gateway"), be.GatewayClient)


# --- cursor backend -------------------------------------------------------------


def _fake_cursor_envelope(result_text: str) -> str:
    # cursor-agent's envelope: no `usage` key (the CLI reports no token counts).
    return json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": result_text}
    )


def test_cursor_model_translation():
    assert be._cursor_model("claude-opus-4-8") == "opus"
    assert be._cursor_model("claude-haiku-4-5") == "haiku-4.5"
    assert be._cursor_model("claude-sonnet-4-5") == "sonnet-4.5"
    # Unknown ids pass through so ModelConfig can hold Cursor-native names.
    assert be._cursor_model("gpt-5") == "gpt-5"
    assert be._cursor_model("auto") == "auto"


def test_cursor_parse_returns_validated_output(monkeypatch):
    calls = {}

    def fake_run(cmd, input, **kwargs):
        calls["cmd"] = cmd
        calls["input"] = input
        calls["kwargs"] = kwargs
        out = _fake_cursor_envelope('{"page_path":"p.mdx","affected":true,'
                                    '"confidence":0.8,"reason":"r"}')
        return type("P", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")

    client = be.CursorClient()
    resp = client.messages.parse(
        output_format=JudgeVerdict,
        model="claude-haiku-4-5",
        system=[{"type": "text", "text": "judge"}],
        messages=[{"role": "user", "content": "is it affected?"}],
        thinking={"type": "adaptive"},  # extra kwargs must be ignored
    )
    assert isinstance(resp.parsed_output, JudgeVerdict)
    assert resp.usage is None  # cursor reports no token usage
    # headless flags + the *translated* model name; no system-prompt flag, no --force
    assert "-p" in calls["cmd"] and "--output-format" in calls["cmd"]
    assert "haiku-4.5" in calls["cmd"] and "claude-haiku-4-5" not in calls["cmd"]
    assert "--system-prompt" not in calls["cmd"]
    assert "--force" not in calls["cmd"] and "-f" not in calls["cmd"]
    # system text is prepended into the stdin prompt alongside the user text
    assert "judge" in calls["input"] and "is it affected?" in calls["input"]
    # every call runs in the client's empty temp cwd
    assert calls["kwargs"].get("cwd") == client._workdir.name


def test_cursor_whole_document_path(monkeypatch):
    page = "---\ntitle: X\ndescription: y\n---\n\nBody text here.\n"

    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0,
                              "stdout": _fake_cursor_envelope(f"```mdx\n{page}\n```"),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")
    client = be.CursorClient()
    resp = client.messages.parse(
        output_format=AuthoredPage, system="s", messages=[{"role": "user", "content": "q"}]
    )
    assert resp.parsed_output.content == page.strip()


def test_cursor_retries_then_raises_on_bad_json(monkeypatch):
    attempts = {"n": 0}

    def fake_run(cmd, input, **kwargs):
        attempts["n"] += 1
        return type("P", (), {"returncode": 0, "stdout": _fake_cursor_envelope("not json"),
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")
    client = be.CursorClient()
    with pytest.raises(RuntimeError, match="cursor-agent"):
        client.messages.parse(
            output_format=JudgeVerdict, system="s",
            messages=[{"role": "user", "content": "q"}],
        )
    assert attempts["n"] == 3  # two retries


def test_cursor_nonzero_exit_raises(monkeypatch):
    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 1, "stdout": "", "stderr": "not logged in"})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")
    client = be.CursorClient()
    with pytest.raises(RuntimeError, match="not logged in"):
        client.messages.parse(
            output_format=JudgeVerdict, system="s",
            messages=[{"role": "user", "content": "q"}],
        )


def test_cursor_non_json_stdout_raises(monkeypatch):
    # cursor-agent's failure mode can be non-JSON stdout (unlike claude's envelope).
    def fake_run(cmd, input, **kwargs):
        return type("P", (), {"returncode": 0, "stdout": "plain text, no envelope",
                              "stderr": ""})()

    monkeypatch.setattr(be.subprocess, "run", fake_run)
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")
    client = be.CursorClient()
    with pytest.raises(RuntimeError, match="non-JSON"):
        client.messages.parse(
            output_format=JudgeVerdict, system="s",
            messages=[{"role": "user", "content": "q"}],
        )


def test_cursor_missing_cli_raises(monkeypatch):
    monkeypatch.setattr(be.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="cursor-agent"):
        be.CursorClient()


def test_get_client_cursor_returns_cursor_client(monkeypatch):
    monkeypatch.setattr(be.shutil, "which", lambda _: "/usr/bin/cursor-agent")
    assert isinstance(be.get_client("cursor"), be.CursorClient)
