"""Stage 6 — open a docs PR (or emit a patch in dry-run).

The docs repo is a real git checkout. In `--open-pr` mode we branch, commit the
written page changes + the advanced cursor, push, and open a PR via `gh`. In
dry-run we just write a `.patch` next to the report so a human can inspect it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def branch_name(repo: str, head_sha: str) -> str:
    slug = repo.split("/")[-1]
    return f"docsync/{slug}-{head_sha[:8]}"


def write_patch(docs_repo: Path, out_path: Path) -> Path:
    """Write the working-tree diff (page changes) to a patch file for inspection."""
    diff = _git(Path(docs_repo), "diff")
    out_path.write_text(diff, encoding="utf-8")
    return out_path


def _existing_pr_url(repo: Path, branch: str) -> str | None:
    """Return the URL of an open PR whose head is `branch`, or None.

    Used so a re-run on the same head_sha (deterministic branch name) updates the
    existing PR in place — the force-push already refreshed its diff — instead of
    erroring on a duplicate `gh pr create`.
    """
    proc = subprocess.run(
        ["gh", "pr", "list", "--head", branch, "--state", "open", "--json", "url"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    return data[0]["url"] if data else None


def _ensure_labels(repo: Path, labels: list[str]) -> None:
    """Best-effort create each label so `gh pr create --label` can't fail on it.

    `gh label create` exits non-zero if the label already exists; we ignore that
    (and any other failure) — labels are a convenience, never a blocker.
    """
    for label in labels:
        subprocess.run(
            ["gh", "label", "create", label, "--color", "1f6feb",
             "--description", "Opened by docsync"],
            cwd=str(repo), capture_output=True, text=True,
        )


def open_pr(
    docs_repo: Path,
    *,
    branch: str,
    title: str,
    body: str,
    paths: list[str],
    base: str = "main",
    reviewers: list[str] | None = None,
    labels: list[str] | None = None,
    push: bool = True,
) -> str:
    """Create a branch, commit `paths` (+ the .docsync cursor), push, open/update a PR.

    Returns the PR URL (or the branch name if `gh` is unavailable / push disabled).
    If a PR already exists for `branch`, the force-push updates it in place and its
    existing URL is returned — no duplicate PR. Assumes the changed files are already
    written to the working tree.
    """
    repo = Path(docs_repo)
    _git(repo, "checkout", "-B", branch)
    for p in paths:
        _git(repo, "add", p)
    # Always include the advanced cursor if it changed.
    _git(repo, "add", "--", ".docsync/state/cursors.json")
    _git(repo, "commit", "-m", title, "-m", body)

    if not push:
        return branch

    try:
        _git(repo, "push", "-u", "origin", branch, "--force-with-lease")

        # An open PR for this branch means the push just updated it — don't re-create.
        existing = _existing_pr_url(repo, branch)
        if existing:
            return existing

        _ensure_labels(repo, labels or [])
        cmd = [
            "gh", "pr", "create", "--title", title, "--body", body, "--base", base,
            "--head", branch,
        ]
        for r in reviewers or []:
            cmd += ["--reviewer", r]
        for label in labels or []:
            cmd += ["--label", label]
        proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        if proc.returncode != 0:
            # Lost a race (PR created concurrently)? Surface it; else the branch.
            existing = _existing_pr_url(repo, branch)
            if existing:
                return existing
            return f"{branch} (gh: {proc.stderr.strip()})"
        return proc.stdout.strip()
    except (RuntimeError, FileNotFoundError) as exc:
        return f"{branch} (push/gh unavailable: {exc})"
