"""Stage 1.5 — event-native ingestion (the CI entry point).

Derive every diff parameter from the GitHub event JSON instead of requiring the
caller to pass ``--src-repo/--base/--head/--pr-number/--pr-title`` flags. This
serves the project's core goal: *triggered by commits, not many parameters*.

In CI, GitHub writes the triggering event to the file named by
``$GITHUB_EVENT_PATH``. :func:`diff_from_event` reads that file, extracts the
comparison parameters with the pure :func:`parse_event`, and delegates the
actual diffing to :func:`docsync.diff.diff_github` — so this module is
network-free and fully testable by monkeypatching ``diff_github``.

Two GitHub event shapes are understood:

* ``repository_dispatch`` — a hand-rolled trigger carrying an explicit
  ``client_payload`` (``repo``/``base_sha``/``head_sha``/``pr_number``/``pr_title``).
* ``push`` — the native push event (``repository.full_name``, ``before``,
  ``after``, ``head_commit.message``).

GitLab CI is also supported, but it does *not* write an event JSON file — it
exposes predefined environment variables instead. :func:`parse_gitlab_env`
recovers the same :class:`EventInfo` from those vars (``CI_PROJECT_PATH``,
``CI_COMMIT_BEFORE_SHA``, ``CI_COMMIT_SHA`` and the ``CI_MERGE_REQUEST_*``
family); :func:`diff_from_gitlab_env` mirrors :func:`diff_from_event`; and
:func:`diff_from_ci` auto-detects the platform (``GITLAB_CI`` env var vs the
GitHub event file).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .diff import diff_github
from .models import CodeDiff

__all__ = [
    "EventInfo",
    "NULL_SHA",
    "is_null_sha",
    "parse_event",
    "parse_gitlab_env",
    "load_event",
    "diff_from_event",
    "diff_from_gitlab_env",
    "diff_from_ci",
]

# GitHub uses an all-zeros SHA as the "base" of a branch's very first push (there
# is no prior commit to compare against).
NULL_SHA = "0" * 40


def is_null_sha(sha: str | None) -> bool:
    """True if ``sha`` is git's all-zeros null SHA (any length of only zeros).

    GitHub emits this for ``before`` on a branch's first push and for the base
    of a ``repository_dispatch`` payload that has no real ancestor.
    """
    return bool(sha) and set(sha) == {"0"}


@dataclass
class EventInfo:
    """The diff parameters recovered from a CI event, before null-base fixup."""

    repo: str
    base_sha: str
    head_sha: str
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None


def _coerce_pr_number(value: object) -> Optional[int]:
    """Best-effort int coercion for ``pr_number`` (JSON may carry it as a string)."""
    if value is None or value == "":
        return None
    return int(value)


def parse_event(event: dict) -> EventInfo:
    """Extract diff parameters from a GitHub event payload (pure; no I/O).

    Recognizes two shapes, in order:

    1. ``repository_dispatch`` — when ``event["client_payload"]`` is present, read
       ``repo``/``base_sha``/``head_sha``/``pr_number``/``pr_title`` from it.
    2. ``push`` — when ``event`` has ``before``/``after``, derive
       ``repo`` from ``repository.full_name``, ``base`` from ``before``,
       ``head`` from ``after``, and ``pr_title`` from ``head_commit.message``
       (``pr_number`` is ``None``).

    Raises:
        ValueError: if the event matches neither shape (no ``client_payload`` and
            no ``before``/``after``).
    """
    payload = event.get("client_payload")
    if payload is not None:
        return EventInfo(
            repo=payload["repo"],
            base_sha=payload["base_sha"],
            head_sha=payload["head_sha"],
            pr_number=_coerce_pr_number(payload.get("pr_number")),
            pr_title=payload.get("pr_title"),
        )

    if "before" in event and "after" in event:
        repository = event.get("repository") or {}
        head_commit = event.get("head_commit") or {}
        return EventInfo(
            repo=repository["full_name"],
            base_sha=event["before"],
            head_sha=event["after"],
            pr_number=None,
            pr_title=head_commit.get("message"),
        )

    raise ValueError(
        "unrecognized GitHub event shape: expected a `client_payload` "
        "(repository_dispatch) or `before`/`after` keys (push)"
    )


def parse_gitlab_env(env: Mapping[str, str]) -> EventInfo:
    """Extract diff parameters from GitLab CI environment variables (pure; no I/O).

    GitLab CI does not write an event JSON file the way GitHub does; it exposes
    `predefined variables <https://docs.gitlab.com/ci/variables/predefined_variables/>`_
    instead. The mapping is:

    * ``CI_PROJECT_PATH`` → ``repo`` (``"group/name"``, matching GitHub's
      ``owner/name`` shape).
    * ``CI_COMMIT_BEFORE_SHA`` → ``base_sha`` (all-zeros on a branch's first
      push, handled downstream by the null-base fallback).
    * ``CI_COMMIT_SHA`` → ``head_sha``.

    For **merge-request pipelines** (``CI_PIPELINE_SOURCE=merge_request_event``)
    the ``CI_MERGE_REQUEST_*`` family is richer and preferred when present:

    * ``CI_MERGE_REQUEST_IID`` → ``pr_number``.
    * ``CI_MERGE_REQUEST_TITLE`` → ``pr_title``.
    * ``CI_MERGE_REQUEST_DIFF_BASE_SHA`` → ``base_sha`` (the merge-base; better
      than ``CI_COMMIT_BEFORE_SHA`` for MR diffs).
    * ``CI_MERGE_REQUEST_SOURCE_BRANCH_SHA`` → ``head_sha`` (the real source-tip
      commit; ``CI_COMMIT_SHA`` is the ephemeral merge-result commit).

    Raises:
        ValueError: if neither ``CI_PROJECT_PATH`` nor the commit SHAs are set,
            i.e. this does not look like a GitLab CI environment.
    """
    repo = env.get("CI_PROJECT_PATH")
    head = env.get("CI_MERGE_REQUEST_SOURCE_BRANCH_SHA") or env.get("CI_COMMIT_SHA")
    if not repo or not head:
        raise ValueError(
            "unrecognized GitLab CI environment: expected `CI_PROJECT_PATH` and "
            "`CI_COMMIT_SHA` (or `CI_MERGE_REQUEST_SOURCE_BRANCH_SHA`) to be set"
        )

    pr_number = _coerce_pr_number(env.get("CI_MERGE_REQUEST_IID"))
    pr_title = env.get("CI_MERGE_REQUEST_TITLE")
    base = (
        env.get("CI_MERGE_REQUEST_DIFF_BASE_SHA")
        or env.get("CI_COMMIT_BEFORE_SHA")
        or NULL_SHA
    )

    return EventInfo(
        repo=repo,
        base_sha=base,
        head_sha=head,
        pr_number=pr_number,
        pr_title=pr_title,
    )


def load_event(event_path: str | Path) -> dict:
    """Read and JSON-parse the event file at ``event_path`` (``$GITHUB_EVENT_PATH``)."""
    return json.loads(Path(event_path).read_text())


def diff_from_event(
    event_path: str | Path,
    *,
    runner: Optional[Callable[..., CodeDiff]] = None,
) -> CodeDiff:
    """Build a :class:`~docsync.models.CodeDiff` from a GitHub event file.

    Reads the JSON at ``event_path`` (this is ``$GITHUB_EVENT_PATH`` in CI),
    parses it with :func:`parse_event`, applies the null-base fallback, and
    delegates to :func:`diff_github`.

    Null-base fallback: when the recovered base is git's all-zeros null SHA (a
    branch's first push has no prior commit), we compare against ``<head>^`` —
    the head commit's first parent — so the diff is still meaningful instead of
    against the empty tree.

    Args:
        event_path: path to the event JSON file.
        runner: the diffing callable; when ``None`` (the default) the module-level
            :func:`diff_github` is resolved at call time, so tests can either pass
            a network-free stub here or monkeypatch ``docsync.events.diff_github``.
    """
    return _diff_from_info(parse_event(load_event(event_path)), runner=runner)


def diff_from_gitlab_env(
    env: Optional[Mapping[str, str]] = None,
    *,
    runner: Optional[Callable[..., CodeDiff]] = None,
) -> CodeDiff:
    """Build a :class:`~docsync.models.CodeDiff` from GitLab CI environment vars.

    The GitLab analogue of :func:`diff_from_event`: there is no event file, so
    the parameters come from the predefined ``CI_*`` variables via
    :func:`parse_gitlab_env`. The same null-base fallback applies — GitLab sets
    ``CI_COMMIT_BEFORE_SHA`` to all-zeros on a branch's first push.

    Args:
        env: the environment mapping to read; defaults to ``os.environ`` when
            ``None``. Tests inject a plain dict so they stay offline.
        runner: the diffing callable; when ``None`` (the default) the module-level
            :func:`diff_github` is used.
    """
    if env is None:
        env = os.environ
    return _diff_from_info(parse_gitlab_env(env), runner=runner)


def diff_from_ci(
    env: Optional[Mapping[str, str]] = None,
    *,
    runner: Optional[Callable[..., CodeDiff]] = None,
) -> CodeDiff:
    """Auto-detect the CI platform and build a :class:`~docsync.models.CodeDiff`.

    Detection order:

    1. ``GITLAB_CI`` set → :func:`diff_from_gitlab_env` (GitLab exposes env vars,
       not an event file).
    2. ``GITHUB_EVENT_PATH`` set → :func:`diff_from_event` (the GitHub event JSON).

    GitHub behavior is unchanged: when ``GITLAB_CI`` is absent this falls through
    to the existing GitHub event-file path exactly as before.

    Args:
        env: the environment mapping to read; defaults to ``os.environ``.
        runner: the diffing callable threaded through to the platform helper.

    Raises:
        ValueError: if neither platform is detected.
    """
    if env is None:
        env = os.environ
    if env.get("GITLAB_CI"):
        return diff_from_gitlab_env(env, runner=runner)
    event_path = env.get("GITHUB_EVENT_PATH")
    if event_path:
        return diff_from_event(event_path, runner=runner)
    raise ValueError(
        "no CI event detected: expected `GITLAB_CI` (GitLab) or "
        "`GITHUB_EVENT_PATH` (GitHub) to be set"
    )


def _diff_from_info(
    info: EventInfo,
    *,
    runner: Optional[Callable[..., CodeDiff]] = None,
) -> CodeDiff:
    """Apply the null-base fallback to ``info`` and delegate to the diff runner.

    Null-base fallback: when the recovered base is git's all-zeros null SHA (a
    branch's first push has no prior commit), compare against ``<head>^`` — the
    head commit's first parent — so the diff is still meaningful instead of
    against the empty tree.
    """
    base = info.base_sha
    if is_null_sha(base):
        base = f"{info.head_sha}^"

    diff_fn = runner if runner is not None else diff_github
    return diff_fn(
        info.repo,
        base,
        info.head_sha,
        pr_number=info.pr_number,
        pr_title=info.pr_title,
    )
