"""Tests for Stage 1.5 — event-native ingestion (docsync.events).

No network / git: event payloads are inlined as dicts and the diffing call is
intercepted by monkeypatching ``docsync.events.diff_github`` with a recorder.
"""

from __future__ import annotations

import json

import pytest

from docsync import events
from docsync.events import (
    NULL_SHA,
    EventInfo,
    diff_from_event,
    is_null_sha,
    load_event,
    parse_event,
)
from docsync.models import CodeDiff

# ---------------------------------------------------------------------------
# Fixtures (inline event payloads; mirror real GitHub event JSON shapes)
# ---------------------------------------------------------------------------

PUSH_EVENT = {
    "repository": {"full_name": "keephq/keep-api-gateway"},
    "before": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "after": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "head_commit": {"message": "fix: handle null base sha"},
}

DISPATCH_EVENT = {
    "action": "docsync",
    "client_payload": {
        "repo": "keephq/keep-workflows",
        "base_sha": "1111111111111111111111111111111111111111",
        "head_sha": "2222222222222222222222222222222222222222",
        "pr_number": 42,
        "pr_title": "Add foreach to actions",
    },
}


def _recorder():
    """A stub diff_github that records its call args and returns a CodeDiff."""
    calls: list[dict] = []

    def fake_diff_github(repo, base, head, *, pr_number=None, pr_title=None):
        calls.append(
            {
                "repo": repo,
                "base": base,
                "head": head,
                "pr_number": pr_number,
                "pr_title": pr_title,
            }
        )
        return CodeDiff(
            repo=repo,
            base_sha=base,
            head_sha=head,
            pr_number=pr_number,
            pr_title=pr_title,
        )

    return fake_diff_github, calls


# ---------------------------------------------------------------------------
# is_null_sha
# ---------------------------------------------------------------------------


def test_is_null_sha():
    assert is_null_sha(NULL_SHA)
    assert is_null_sha("0" * 40)
    assert not is_null_sha("abc123")
    assert not is_null_sha("")
    assert not is_null_sha(None)


# ---------------------------------------------------------------------------
# parse_event
# ---------------------------------------------------------------------------


def test_parse_push_event():
    info = parse_event(PUSH_EVENT)
    assert isinstance(info, EventInfo)
    assert info.repo == "keephq/keep-api-gateway"
    assert info.base_sha == PUSH_EVENT["before"]
    assert info.head_sha == PUSH_EVENT["after"]
    assert info.pr_title == "fix: handle null base sha"
    assert info.pr_number is None


def test_parse_dispatch_event_with_pr_number():
    info = parse_event(DISPATCH_EVENT)
    assert info.repo == "keephq/keep-workflows"
    assert info.base_sha == DISPATCH_EVENT["client_payload"]["base_sha"]
    assert info.head_sha == DISPATCH_EVENT["client_payload"]["head_sha"]
    assert info.pr_number == 42
    assert isinstance(info.pr_number, int)
    assert info.pr_title == "Add foreach to actions"


def test_parse_dispatch_event_pr_number_as_string():
    """JSON may carry pr_number as a string; it must coerce to int."""
    event = {"client_payload": {**DISPATCH_EVENT["client_payload"], "pr_number": "7"}}
    info = parse_event(event)
    assert info.pr_number == 7
    assert isinstance(info.pr_number, int)


def test_parse_dispatch_event_without_pr_number():
    payload = {
        "repo": "keephq/keep-ui",
        "base_sha": "3333333333333333333333333333333333333333",
        "head_sha": "4444444444444444444444444444444444444444",
    }
    info = parse_event({"client_payload": payload})
    assert info.pr_number is None
    assert info.pr_title is None


def test_parse_push_event_without_head_commit():
    event = {
        "repository": {"full_name": "keephq/keep-ui"},
        "before": "5555555555555555555555555555555555555555",
        "after": "6666666666666666666666666666666666666666",
    }
    info = parse_event(event)
    assert info.pr_title is None
    assert info.pr_number is None


def test_parse_unrecognized_event_raises():
    with pytest.raises(ValueError):
        parse_event({"zen": "Keep it logically awesome."})


# ---------------------------------------------------------------------------
# load_event
# ---------------------------------------------------------------------------


def test_load_event_reads_and_parses(tmp_path):
    path = tmp_path / "event.json"
    path.write_text(json.dumps(PUSH_EVENT))
    assert load_event(path) == PUSH_EVENT


# ---------------------------------------------------------------------------
# diff_from_event
# ---------------------------------------------------------------------------


def test_diff_from_event_threads_args(tmp_path, monkeypatch):
    fake, calls = _recorder()
    monkeypatch.setattr(events, "diff_github", fake)

    path = tmp_path / "event.json"
    path.write_text(json.dumps(DISPATCH_EVENT))

    result = diff_from_event(path)

    assert isinstance(result, CodeDiff)
    assert len(calls) == 1
    call = calls[0]
    assert call["repo"] == "keephq/keep-workflows"
    assert call["base"] == DISPATCH_EVENT["client_payload"]["base_sha"]
    assert call["head"] == DISPATCH_EVENT["client_payload"]["head_sha"]
    assert call["pr_number"] == 42
    assert call["pr_title"] == "Add foreach to actions"
    assert result.repo == "keephq/keep-workflows"
    assert result.pr_number == 42


def test_diff_from_event_push(tmp_path, monkeypatch):
    fake, calls = _recorder()
    monkeypatch.setattr(events, "diff_github", fake)

    path = tmp_path / "event.json"
    path.write_text(json.dumps(PUSH_EVENT))

    diff_from_event(path)

    call = calls[0]
    assert call["repo"] == "keephq/keep-api-gateway"
    assert call["base"] == PUSH_EVENT["before"]
    assert call["head"] == PUSH_EVENT["after"]
    assert call["pr_number"] is None
    assert call["pr_title"] == "fix: handle null base sha"


def test_diff_from_event_null_base_falls_back_to_head_parent(tmp_path, monkeypatch):
    """An all-zeros base must become ``<head>^`` so the first-push diff is meaningful."""
    fake, calls = _recorder()
    monkeypatch.setattr(events, "diff_github", fake)

    head = "7777777777777777777777777777777777777777"
    event = {
        "repository": {"full_name": "keephq/keep-event-handler"},
        "before": NULL_SHA,
        "after": head,
        "head_commit": {"message": "initial commit"},
    }
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event))

    diff_from_event(path)

    call = calls[0]
    assert call["base"] == f"{head}^"
    assert call["head"] == head


def test_diff_from_event_runner_injection(tmp_path):
    """The injected runner is honored without monkeypatching the module."""
    fake, calls = _recorder()

    path = tmp_path / "event.json"
    path.write_text(json.dumps(DISPATCH_EVENT))

    diff_from_event(path, runner=fake)

    assert len(calls) == 1
    assert calls[0]["repo"] == "keephq/keep-workflows"
