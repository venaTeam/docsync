"""PR stage tests — branch/commit/push + dedup + labels, with subprocess faked.

`open_pr` shells out to `git` and `gh`; here a single fake `subprocess.run` scripts
both, so we assert the *decisions* (create vs. update-in-place, label handling)
without touching a real repo or network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from docsync import pr


class FakeRun:
    """Scriptable stand-in for subprocess.run; records every argv it sees."""

    def __init__(self, *, pr_list="[]", create_rc=0, create_url="https://gh/o/r/pull/7"):
        self.calls: list[list[str]] = []
        self.pr_list = pr_list
        self.create_rc = create_rc
        self.create_url = create_url

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        head = cmd[:3]
        if cmd[0] == "git":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if head == ["gh", "pr", "list"]:
            return SimpleNamespace(returncode=0, stdout=self.pr_list, stderr="")
        if head == ["gh", "label", "create"]:
            # Mimic "already exists" — open_pr must ignore this.
            return SimpleNamespace(returncode=1, stdout="", stderr="already exists")
        if head == ["gh", "pr", "create"]:
            return SimpleNamespace(
                returncode=self.create_rc, stdout=self.create_url, stderr="boom"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def cmds_starting(self, *prefix) -> list[list[str]]:
        return [c for c in self.calls if c[: len(prefix)] == list(prefix)]


def _open(repo: Path, fake: FakeRun, **kw):
    return pr.open_pr(
        repo, branch="docsync/svc-abc12345", title="docs: sync",
        body="b", paths=["page.mdx"], labels=["docsync"], **kw,
    )


def test_branch_name_is_deterministic_per_head_sha():
    a = pr.branch_name("keephq/keep-api-gateway", "abcdef1234567890")
    assert a == "docsync/keep-api-gateway-abcdef12"
    assert a == pr.branch_name("forkowner/keep-api-gateway", "abcdef1234567890")  # slug only


def test_open_pr_creates_when_none_exists(tmp_path: Path, monkeypatch):
    fake = FakeRun(pr_list="[]")
    monkeypatch.setattr(pr.subprocess, "run", fake)

    url = _open(tmp_path, fake)

    assert url == "https://gh/o/r/pull/7"
    creates = fake.cmds_starting("gh", "pr", "create")
    assert len(creates) == 1
    assert "--label" in creates[0] and "docsync" in creates[0]


def test_open_pr_updates_in_place_when_pr_exists(tmp_path: Path, monkeypatch):
    fake = FakeRun(pr_list='[{"url": "https://gh/o/r/pull/3"}]')
    monkeypatch.setattr(pr.subprocess, "run", fake)

    url = _open(tmp_path, fake)

    # Returns the existing PR (the force-push refreshed it) and never re-creates.
    assert url == "https://gh/o/r/pull/3"
    assert fake.cmds_starting("gh", "pr", "create") == []
    # The branch was still force-pushed (update in place).
    assert any("push" in c and "--force-with-lease" in c for c in fake.calls)


def test_open_pr_ensures_labels_before_create(tmp_path: Path, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(pr.subprocess, "run", fake)

    _open(tmp_path, fake)

    # Label is best-effort created (idempotent) ahead of the PR create.
    assert fake.cmds_starting("gh", "label", "create")


def test_open_pr_create_failure_falls_back_to_existing(tmp_path: Path, monkeypatch):
    # create fails, but a concurrent PR now exists -> surface it instead of an error.
    fake = FakeRun(create_rc=1, pr_list='[{"url": "https://gh/o/r/pull/9"}]')
    # First pr-list (pre-create) must be empty so we attempt create; flip after.
    calls = {"n": 0}
    real = fake.__call__

    def wrapped(cmd, *a, **k):
        if cmd[:3] == ["gh", "pr", "list"]:
            calls["n"] += 1
            stdout = "[]" if calls["n"] == 1 else '[{"url": "https://gh/o/r/pull/9"}]'
            fake.calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        return real(cmd, *a, **k)

    monkeypatch.setattr(pr.subprocess, "run", wrapped)

    url = _open(tmp_path, fake)
    assert url == "https://gh/o/r/pull/9"


def test_open_pr_no_push_returns_branch(tmp_path: Path, monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(pr.subprocess, "run", fake)

    url = _open(tmp_path, fake, push=False)

    assert url == "docsync/svc-abc12345"
    assert fake.cmds_starting("gh", "pr", "create") == []
    assert not any("push" in c for c in fake.calls)


def test_write_patch_returns_none_on_non_git_dir(tmp_path):
    # A fresh from-scratch scaffold isn't a git repo; write_patch must degrade to
    # None (not crash) so already-written pages aren't lost to a failed patch step.
    assert pr.write_patch(tmp_path, tmp_path / "out.patch") is None
    assert not (tmp_path / "out.patch").exists()
