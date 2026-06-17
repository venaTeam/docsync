"""Stage 1.5 — event-native ingestion (the CI entry point).

Derive every diff parameter from the GitHub event JSON instead of requiring the
caller to pass ``--src-repo/--base/--head/--pr-number/--pr-title`` flags. This
serves the project's core goal: *triggered by commits, not many parameters*.

In CI, GitHub writes the triggering event to the file named by
``$GITHUB_EVENT_PATH``. :func:`diff_from_event` reads that file, extracts the
comparison parameters with the pure :func:`parse_event`, and delegates the
actual diffing to :func:`docsync.diff.diff_github` — so this module is
network-free and fully testable by monkeypatching ``diff_github``.

Two event shapes are understood:

* ``repository_dispatch`` — a hand-rolled trigger carrying an explicit
  ``client_payload`` (``repo``/``base_sha``/``head_sha``/``pr_number``/``pr_title``).
* ``push`` — the native push event (``repository.full_name``, ``before``,
  ``after``, ``head_commit.message``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .diff import diff_github
from .models import CodeDiff

__all__ = [
    "EventInfo",
    "NULL_SHA",
    "is_null_sha",
    "parse_event",
    "load_event",
    "diff_from_event",
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
    """The diff parameters recovered from a GitHub event, before null-base fixup."""

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
    info = parse_event(load_event(event_path))

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
