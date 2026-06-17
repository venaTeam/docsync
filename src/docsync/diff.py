"""Stage 2 — diff extraction.

Turn a ``base..head`` comparison (from a local git checkout or a remote GitHub
repo) into the structured :class:`~docsync.models.CodeDiff` the rest of the
pipeline builds against.

Two public entry points, both returning a ``CodeDiff``:

* :func:`diff_local`  — primary CLI path; shells out to ``git diff`` and parses
  the unified diff with the ``unidiff`` library.
* :func:`diff_github` — CI path; calls ``gh api repos/{repo}/compare/...`` and
  parses GitHub's per-file ``patch`` fragments.

The load-bearing piece both share is :func:`extract_changed_symbols`, which
recovers the function/class/assignment names a set of hunks touch *without*
needing the full post-image file text — it works from the hunk text alone.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from unidiff import PatchSet

from docsync.models import ChangedFile, CodeDiff, FileStatus

__all__ = [
    "diff_local",
    "diff_github",
    "parse_patchset",
    "extract_changed_symbols",
]


# ---------------------------------------------------------------------------
# Symbol extraction (the cross-boundary signal; survives line-number churn)
# ---------------------------------------------------------------------------

# `def foo(` / `async def foo(` / `class Foo(` / `class Foo:` — capture NAME.
_DEF_OR_CLASS = re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
# Module-level assignment: NAME = ...  (no leading whitespace -> top level).
# Allows multiple targets / type annotations: `FOO: list = ...`, `FOO = BAR = ...`.
_MODULE_ASSIGN = re.compile(r"^([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)")
# The enclosing-scope heading git/unidiff append after the second `@@`.
_HUNK_HEADER = re.compile(r"^@@.*?@@\s*(.*)$")


def _add(out: list[str], seen: set[str], name: str | None) -> None:
    if name and name not in seen:
        seen.add(name)
        out.append(name)


def _symbols_from_text(line: str, out: list[str], seen: set[str]) -> None:
    """Pull def/class names (anywhere) and module-level assignments from a line."""
    for m in _DEF_OR_CLASS.finditer(line):
        _add(out, seen, m.group(1))
    # Module-level assignment: only when the line has no leading indentation.
    if line[:1] not in (" ", "\t"):
        m = _MODULE_ASSIGN.match(line)
        if m:
            _add(out, seen, m.group(1))


def extract_changed_symbols(file_path: str, hunks: list[str]) -> list[str]:
    """Best-effort recovery of the symbols a set of hunks touch.

    Python files only (``.py``); returns ``[]`` for anything else. Hunks-only:
    we never require the full file text, so this is robust when all we have is a
    diff fragment (e.g. GitHub's ``patch`` field).

    Two complementary signals are combined, in order:

    1. The enclosing-scope heading git puts after the second ``@@`` of each hunk
       (``@@ -2,6 +2,8 @@ def register_routes(app):``) — names the function or
       class whose body changed even when its own signature line is untouched.
    2. The added/removed lines themselves — ``def NAME``, ``class NAME``, and
       module-level ``NAME = ...`` assignments introduced or deleted by the hunk.

    Returns a de-duplicated, order-preserving list (heading symbols first).
    """
    if not file_path.endswith(".py"):
        return []

    out: list[str] = []
    seen: set[str] = set()

    for hunk in hunks:
        lines = hunk.splitlines()
        if not lines:
            continue

        # Signal 1: the `@@ ... @@ <heading>` enclosing-scope heading.
        header = _HUNK_HEADER.match(lines[0])
        if header:
            heading = header.group(1).strip()
            if heading:
                _symbols_from_text(heading, out, seen)

        # Signal 2: the changed lines themselves (added `+` / removed `-`),
        # excluding the `+++`/`---` file markers a raw patch may carry.
        for raw in lines[1:]:
            if not raw or raw[0] not in "+-":
                continue
            if raw.startswith(("+++", "---")):
                continue
            content = raw[1:]
            _symbols_from_text(content, out, seen)

    return out


# ---------------------------------------------------------------------------
# Local path — `git diff` + unidiff
# ---------------------------------------------------------------------------


def _strip_ab_prefix(path: str | None) -> str | None:
    """Drop git's ``a/`` / ``b/`` prefix (and handle ``/dev/null``)."""
    if path is None:
        return None
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _status_of(patched_file) -> FileStatus:
    if patched_file.is_added_file:
        return FileStatus.ADDED
    if patched_file.is_removed_file:
        return FileStatus.REMOVED
    if patched_file.is_rename:
        return FileStatus.RENAMED
    return FileStatus.MODIFIED


def parse_patchset(diff_text: str, repo: str, base: str, head: str,
                   pr_number: int | None = None, pr_title: str | None = None) -> CodeDiff:
    """Parse a unified-diff string into a ``CodeDiff``.

    Factored out of :func:`diff_local` so it can be exercised with an inline
    diff fixture (no git/subprocess). ``pipeline.py`` and the tests rely on this
    exact signature.
    """
    patch_set = PatchSet(diff_text)
    files: list[ChangedFile] = []

    for patched_file in patch_set:
        status = _status_of(patched_file)

        # unidiff's `.path` already strips the a/ b/ prefix and prefers the
        # target side; fall back to target/source if needed.
        path = patched_file.path
        if not path:
            path = (
                _strip_ab_prefix(patched_file.target_file)
                or _strip_ab_prefix(patched_file.source_file)
                or ""
            )

        previous_path = None
        if status is FileStatus.RENAMED:
            previous_path = _strip_ab_prefix(patched_file.source_file)

        hunks = [str(hunk) for hunk in patched_file]
        symbols = extract_changed_symbols(path, hunks)

        files.append(
            ChangedFile(
                path=path,
                status=status,
                previous_path=previous_path,
                hunks=hunks,
                changed_symbols=symbols,
            )
        )

    return CodeDiff(
        repo=repo,
        base_sha=base,
        head_sha=head,
        pr_number=pr_number,
        pr_title=pr_title,
        files=files,
    )


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    """Run a subprocess, returning stdout; raise a clear RuntimeError on failure."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:  # e.g. git / gh not installed
        raise RuntimeError(f"command not found: {cmd[0]!r} ({' '.join(cmd)})") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr.strip()}"
        )
    return proc.stdout


def diff_local(repo_path: str | Path, base: str, head: str,
               pr_number: int | None = None, pr_title: str | None = None) -> CodeDiff:
    """Compare ``base..head`` in a LOCAL git checkout via ``git diff``.

    Primary path for the CLI dogfood. ``-M`` enables rename detection so renames
    surface as :attr:`FileStatus.RENAMED` rather than add+remove pairs.
    """
    repo_path = str(repo_path)
    diff_text = _run(
        ["git", "-C", repo_path, "diff", "--unified=3", "-M", base, head]
    )
    return parse_patchset(
        diff_text,
        repo=repo_path,
        base=base,
        head=head,
        pr_number=pr_number,
        pr_title=pr_title,
    )


# ---------------------------------------------------------------------------
# Remote path — `gh api repos/{repo}/compare/...`
# ---------------------------------------------------------------------------

_GH_STATUS = {
    "added": FileStatus.ADDED,
    "modified": FileStatus.MODIFIED,
    "removed": FileStatus.REMOVED,
    "renamed": FileStatus.RENAMED,
    # GitHub also emits these for large/odd diffs; map them onto MODIFIED so we
    # still record the file (its `patch` may be empty).
    "changed": FileStatus.MODIFIED,
    "copied": FileStatus.ADDED,
}


def _split_github_patch(patch: str | None) -> list[str]:
    """Split a GitHub ``patch`` fragment into per-hunk strings.

    GitHub's ``patch`` has no ``diff --git``/``---``/``+++`` header — it starts
    at the first ``@@``. We split on lines beginning with ``@@`` and re-attach
    that header line to each hunk so the result looks like a standalone hunk
    (and so :func:`extract_changed_symbols` can read the heading).
    """
    if not patch:
        return []

    hunks: list[str] = []
    current: list[str] | None = None
    for line in patch.splitlines():
        if line.startswith("@@"):
            if current is not None:
                hunks.append("\n".join(current))
            current = [line]
        elif current is not None:
            current.append(line)
        # lines before the first @@ (shouldn't normally exist) are dropped.
    if current is not None:
        hunks.append("\n".join(current))
    return hunks


def diff_github(repo: str, base: str, head: str, *, token: str | None = None,
                pr_number: int | None = None, pr_title: str | None = None) -> CodeDiff:
    """Compare ``base..head`` in a REMOTE repo via the ``gh`` CLI.

    ``repo`` is ``"owner/name"``. Uses ``gh api repos/{repo}/compare/{base}...{head}``
    and parses the returned ``files`` array. The token is taken from the
    ``token`` argument, else ``GH_TOKEN`` / ``GITHUB_TOKEN`` in the environment.
    """
    env = dict(os.environ)
    token = token or env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")
    if token:
        # gh prefers GH_TOKEN; set both so it authenticates regardless.
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token

    endpoint = f"repos/{repo}/compare/{base}...{head}"
    # --paginate so the full files list is returned for large compares.
    raw = _run(["gh", "api", "--paginate", endpoint], env=env)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"could not parse `gh api {endpoint}` JSON response: {exc}"
        ) from exc

    # --paginate may concatenate multiple JSON objects; tolerate either shape.
    file_entries: list[dict] = []
    payloads = payload if isinstance(payload, list) else [payload]
    for page in payloads:
        file_entries.extend(page.get("files", []) or [])

    files: list[ChangedFile] = []
    for entry in file_entries:
        path = entry.get("filename", "")
        status = _GH_STATUS.get(entry.get("status", ""), FileStatus.MODIFIED)
        previous_path = entry.get("previous_filename") if status is FileStatus.RENAMED else None
        hunks = _split_github_patch(entry.get("patch"))
        symbols = extract_changed_symbols(path, hunks)
        files.append(
            ChangedFile(
                path=path,
                status=status,
                previous_path=previous_path,
                hunks=hunks,
                changed_symbols=symbols,
            )
        )

    return CodeDiff(
        repo=repo,
        base_sha=base,
        head_sha=head,
        pr_number=pr_number,
        pr_title=pr_title,
        files=files,
    )
