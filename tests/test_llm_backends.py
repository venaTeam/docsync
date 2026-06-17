"""Unit tests for the Claude Code dev backend — no real CLI calls (subprocess mocked)."""

from __future__ import annotations

import json

import pytest

from docsync import llm_backends as be
from docsync.models import JudgeVerdict


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

    def fake_run(cmd, input, capture_output, text):
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

    def fake_run(cmd, input, capture_output, text):
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


def test_get_client_unknown_backend():
    with pytest.raises(ValueError):
        be.get_client("nope")
