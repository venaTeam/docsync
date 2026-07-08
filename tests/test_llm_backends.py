"""Unit tests for the CLI dev backends (claude-code, cursor) — no real CLI calls."""

from __future__ import annotations

import json

import pytest

from docsync import llm_backends as be
from docsync.models import AuthoredPage, JudgeVerdict


def test_extract_json_handles_fences_and_prose():
    assert json.loads(be._extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    assert json.loads(be._extract_json('Here you go: {"a": 2} done')) == {"a": 2}
    with pytest.raises(ValueError):
        be._extract_json("no json here")


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
    assert attempts["n"] == 2  # one retry


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


def test_get_client_unknown_backend():
    with pytest.raises(ValueError, match="'cursor'"):
        be.get_client("nope")


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
    assert attempts["n"] == 2  # one retry


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
