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
    diff_from_ci,
    diff_from_event,
    diff_from_gitlab_env,
    is_null_sha,
    load_event,
    parse_event,
    parse_gitlab_env,
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


# ---------------------------------------------------------------------------
# GitLab CI — env-var based (no event file)
# ---------------------------------------------------------------------------

# A push pipeline: only the commit-level CI_* vars are set.
GITLAB_PUSH_ENV = {
    "GITLAB_CI": "true",
    "CI_PROJECT_PATH": "keephq/keep-api-gateway",
    "CI_COMMIT_BEFORE_SHA": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "CI_COMMIT_SHA": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
}

# A merge-request pipeline: the richer CI_MERGE_REQUEST_* family is present and
# preferred over the commit-level vars.
GITLAB_MR_ENV = {
    "GITLAB_CI": "true",
    "CI_PIPELINE_SOURCE": "merge_request_event",
    "CI_PROJECT_PATH": "keephq/keep-workflows",
    # The ephemeral merge-result commit / branch tip — should be overridden by
    # the MR-specific vars below.
    "CI_COMMIT_BEFORE_SHA": "cccccccccccccccccccccccccccccccccccccccc",
    "CI_COMMIT_SHA": "dddddddddddddddddddddddddddddddddddddddd",
    "CI_MERGE_REQUEST_IID": "42",
    "CI_MERGE_REQUEST_TITLE": "Add foreach to actions",
    "CI_MERGE_REQUEST_DIFF_BASE_SHA": "1111111111111111111111111111111111111111",
    "CI_MERGE_REQUEST_SOURCE_BRANCH_SHA": "2222222222222222222222222222222222222222",
}


def test_parse_gitlab_push_env():
    info = parse_gitlab_env(GITLAB_PUSH_ENV)
    assert isinstance(info, EventInfo)
    assert info.repo == "keephq/keep-api-gateway"
    assert info.base_sha == GITLAB_PUSH_ENV["CI_COMMIT_BEFORE_SHA"]
    assert info.head_sha == GITLAB_PUSH_ENV["CI_COMMIT_SHA"]
    assert info.pr_number is None
    assert info.pr_title is None


def test_parse_gitlab_merge_request_env_prefers_mr_vars():
    info = parse_gitlab_env(GITLAB_MR_ENV)
    assert info.repo == "keephq/keep-workflows"
    # MR diff-base + source-branch tip win over the commit-level vars.
    assert info.base_sha == GITLAB_MR_ENV["CI_MERGE_REQUEST_DIFF_BASE_SHA"]
    assert info.head_sha == GITLAB_MR_ENV["CI_MERGE_REQUEST_SOURCE_BRANCH_SHA"]
    assert info.pr_number == 42
    assert isinstance(info.pr_number, int)
    assert info.pr_title == "Add foreach to actions"


def test_parse_gitlab_env_missing_vars_raises():
    with pytest.raises(ValueError):
        parse_gitlab_env({"GITLAB_CI": "true"})


def test_diff_from_gitlab_env_push():
    fake, calls = _recorder()

    diff_from_gitlab_env(GITLAB_PUSH_ENV, runner=fake)

    call = calls[0]
    assert call["repo"] == "keephq/keep-api-gateway"
    assert call["base"] == GITLAB_PUSH_ENV["CI_COMMIT_BEFORE_SHA"]
    assert call["head"] == GITLAB_PUSH_ENV["CI_COMMIT_SHA"]
    assert call["pr_number"] is None
    assert call["pr_title"] is None


def test_diff_from_gitlab_env_merge_request():
    fake, calls = _recorder()

    result = diff_from_gitlab_env(GITLAB_MR_ENV, runner=fake)

    assert isinstance(result, CodeDiff)
    call = calls[0]
    assert call["repo"] == "keephq/keep-workflows"
    assert call["base"] == GITLAB_MR_ENV["CI_MERGE_REQUEST_DIFF_BASE_SHA"]
    assert call["head"] == GITLAB_MR_ENV["CI_MERGE_REQUEST_SOURCE_BRANCH_SHA"]
    assert call["pr_number"] == 42
    assert call["pr_title"] == "Add foreach to actions"


def test_diff_from_gitlab_env_null_base_falls_back_to_head_parent():
    """GitLab's all-zeros CI_COMMIT_BEFORE_SHA (first push) must become ``<head>^``."""
    fake, calls = _recorder()

    head = "7777777777777777777777777777777777777777"
    env = {
        "GITLAB_CI": "true",
        "CI_PROJECT_PATH": "keephq/keep-event-handler",
        "CI_COMMIT_BEFORE_SHA": NULL_SHA,
        "CI_COMMIT_SHA": head,
    }

    diff_from_gitlab_env(env, runner=fake)

    call = calls[0]
    assert call["base"] == f"{head}^"
    assert call["head"] == head


def test_diff_from_gitlab_env_defaults_to_os_environ(monkeypatch):
    """With no env injected, it reads os.environ (here: a stubbed merge-request)."""
    fake, calls = _recorder()
    for key, value in GITLAB_MR_ENV.items():
        monkeypatch.setenv(key, value)

    diff_from_gitlab_env(runner=fake)

    assert calls[0]["repo"] == "keephq/keep-workflows"
    assert calls[0]["pr_number"] == 42


# ---------------------------------------------------------------------------
# diff_from_ci — platform auto-detection
# ---------------------------------------------------------------------------


def test_diff_from_ci_detects_gitlab():
    fake, calls = _recorder()

    diff_from_ci(GITLAB_PUSH_ENV, runner=fake)

    assert calls[0]["repo"] == "keephq/keep-api-gateway"
    assert calls[0]["base"] == GITLAB_PUSH_ENV["CI_COMMIT_BEFORE_SHA"]


def test_diff_from_ci_detects_github(tmp_path):
    fake, calls = _recorder()

    path = tmp_path / "event.json"
    path.write_text(json.dumps(PUSH_EVENT))
    env = {"GITHUB_EVENT_PATH": str(path)}

    diff_from_ci(env, runner=fake)

    assert calls[0]["repo"] == "keephq/keep-api-gateway"
    assert calls[0]["base"] == PUSH_EVENT["before"]
    assert calls[0]["head"] == PUSH_EVENT["after"]


def test_diff_from_ci_prefers_gitlab_when_both_present(tmp_path):
    """GITLAB_CI wins over a stray GITHUB_EVENT_PATH (GitLab is checked first)."""
    fake, calls = _recorder()

    path = tmp_path / "event.json"
    path.write_text(json.dumps(PUSH_EVENT))
    env = {**GITLAB_PUSH_ENV, "GITHUB_EVENT_PATH": str(path)}

    diff_from_ci(env, runner=fake)

    # Resolved via GitLab (keep-api-gateway here too, but base differs by source).
    assert calls[0]["base"] == GITLAB_PUSH_ENV["CI_COMMIT_BEFORE_SHA"]


def test_diff_from_ci_no_platform_raises():
    with pytest.raises(ValueError):
        diff_from_ci({})
