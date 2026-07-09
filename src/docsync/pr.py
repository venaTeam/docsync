"""Stage 6 — open a docs PR / MR (or emit a patch in dry-run).

The docs repo is a real git checkout. In `--open-pr` mode we branch, commit the
written page changes + the advanced cursor, push, and open a review request on the
host: a **GitHub** PR via `gh`, or a **GitLab** MR via `glab`. The host is chosen by
`forge` ("auto" detects it from the `origin` remote). In dry-run we just write a
`.patch` next to the report so a human can inspect it.
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
    """Build the docsync branch name for a repo's head commit.

    Derives a short repo slug from the last path segment of the repo identifier and
    combines it with the first eight characters of the head SHA.

    Args:
        repo: Repository identifier (e.g. "owner/name"); the segment after the last
            "/" is used as the slug.
        head_sha: Commit SHA of the head being processed; only its first eight
            characters are included.

    Returns:
        A branch name of the form "docsync/{slug}-{head_sha[:8]}".
    """
    slug = repo.split("/")[-1]
    return f"docsync/{slug}-{head_sha[:8]}"


def write_patch(docs_repo: Path, out_path: Path) -> Path | None:
    """Write the working-tree diff (page changes) to a patch file for inspection.

    Returns None when the docs repo isn't a git repo (e.g. a fresh from-scratch
    scaffold) — the pages are written regardless, so a missing patch must not crash
    the command.
    """
    try:
        diff = _git(Path(docs_repo), "diff")
    except RuntimeError:
        return None
    out_path.write_text(diff, encoding="utf-8")
    return out_path


def detect_forge(repo: Path) -> str:
    """Best-effort host detection from the `origin` remote URL: 'gitlab' or 'github'.

    GitLab.com and most self-managed instances carry "gitlab" in the URL; everything
    else (including GitHub Enterprise on a custom host) defaults to 'github' so the
    long-standing GitHub behavior is unchanged. Self-managed GitLab on an opaque
    hostname should set `forge: gitlab` in config rather than rely on this guess.
    """
    try:
        url = _git(repo, "remote", "get-url", "origin")
    except RuntimeError:
        return "github"
    return "gitlab" if "gitlab" in url.lower() else "github"


def _resolve_forge(repo: Path, forge: str) -> str:
    """An explicit 'github'/'gitlab' wins; 'auto' (or anything else) is detected."""
    return forge if forge in ("github", "gitlab") else detect_forge(repo)


def _last_url(text: str) -> str:
    """Pull the last http(s) URL out of CLI stdout (glab prints extra chatter)."""
    tokens = [t for t in text.split() if t.startswith("http://") or t.startswith("https://")]
    return tokens[-1] if tokens else text.strip()


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


def _existing_mr_url(repo: Path, branch: str) -> str | None:
    """Return the URL of an open GitLab MR whose source branch is `branch`, or None.

    The GitLab analogue of :func:`_existing_pr_url`: a re-run on the same head_sha
    force-pushes the deterministic branch (refreshing the MR), so we return the open
    MR's URL instead of letting `glab mr create` error on a duplicate.
    """
    proc = subprocess.run(
        ["glab", "mr", "list", "--source-branch", branch, "-F", "json"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not data:
        return None
    return data[0].get("web_url") or data[0].get("url")


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


def _open_github(
    repo: Path, *, branch: str, title: str, body: str, base: str,
    reviewers: list[str] | None, labels: list[str] | None,
) -> str:
    """Open/return a GitHub PR for an already-pushed `branch`, via `gh`."""
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


def _open_gitlab(
    repo: Path, *, branch: str, title: str, body: str, base: str,
    reviewers: list[str] | None, labels: list[str] | None,
) -> str:
    """Open/return a GitLab MR for an already-pushed `branch`, via `glab`.

    Mirrors :func:`_open_github`: dedup an open MR on the source branch (the
    force-push refreshed it), else create one. GitLab creates unknown labels on
    apply, so there's no label pre-create step (unlike `gh`).
    """
    existing = _existing_mr_url(repo, branch)
    if existing:
        return existing

    cmd = [
        "glab", "mr", "create", "--source-branch", branch, "--target-branch", base,
        "--title", title, "--description", body, "--yes",
    ]
    for r in reviewers or []:
        cmd += ["--reviewer", r]
    for label in labels or []:
        cmd += ["--label", label]
    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    if proc.returncode != 0:
        existing = _existing_mr_url(repo, branch)
        if existing:
            return existing
        return f"{branch} (glab: {proc.stderr.strip()})"
    return _last_url(proc.stdout)


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
    forge: str = "auto",
) -> str:
    """Create a branch, commit `paths` (+ the .docsync cursor), push, open/update a PR/MR.

    `forge` picks the host: 'github' (PR via `gh`), 'gitlab' (MR via `glab`), or 'auto'
    to detect it from the `origin` remote. Returns the PR/MR URL (or the branch name if
    the host CLI is unavailable / push disabled). If a review request already exists for
    `branch`, the force-push updates it in place and its URL is returned — no duplicate.
    Assumes the changed files are already written to the working tree.
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
        platform = _resolve_forge(repo, forge)
        opener = _open_gitlab if platform == "gitlab" else _open_github
        return opener(
            repo, branch=branch, title=title, body=body, base=base,
            reviewers=reviewers, labels=labels,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        return f"{branch} (push/host CLI unavailable: {exc})"
