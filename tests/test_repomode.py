"""Repo-topology resolution + detection (`config.resolve_repo_mode`, `scaffold.detect_repo_mode`).

mono = docs and code in one checkout; single = code repo + separate docs repo;
poly = many code repos + one docs repo. No network, no LLM.
"""

from __future__ import annotations

from pathlib import Path

from docsync.config import resolve_repo_mode
from docsync.models import DocsyncConfig, Manifest, ManifestPage, ManifestSource
from docsync.scaffold import detect_repo_mode


def _manifest(*repos: str) -> Manifest:
    return Manifest(
        pages=[
            ManifestPage(path=f"p{i}.mdx", sources=[ManifestSource(repo=r, globs=["src/*.py"])])
            for i, r in enumerate(repos)
        ]
    )


# --- resolve_repo_mode --------------------------------------------------------


def test_explicit_mode_short_circuits(tmp_path: Path):
    config = DocsyncConfig(repo_mode="single")
    # Even though the diff repo == docs repo (would auto-detect mono), explicit wins.
    diff_repo = f"owner/{tmp_path.resolve().name}"
    assert resolve_repo_mode(config, tmp_path, diff_repo, _manifest("a")) == "single"


def test_auto_mono_when_diff_repo_is_docs_repo(tmp_path: Path):
    config = DocsyncConfig(repo_mode="auto")
    # An owner/name whose bare name matches the docs checkout dir → mono.
    diff_repo = f"venaTeam/{tmp_path.resolve().name}"
    assert resolve_repo_mode(config, tmp_path, diff_repo, _manifest("a")) == "mono"


def test_auto_mono_when_diff_repo_is_local_docs_path(tmp_path: Path):
    config = DocsyncConfig(repo_mode="auto")
    # A local --src-repo pointing at the docs checkout itself → mono.
    assert resolve_repo_mode(config, tmp_path, str(tmp_path), _manifest("a")) == "mono"


def test_auto_single_for_separate_repo(tmp_path: Path):
    config = DocsyncConfig(repo_mode="auto")
    # A diff from a different repo than the docs checkout → not mono; one repo → single.
    assert resolve_repo_mode(config, tmp_path, "owner/service", _manifest("owner/service")) == "single"


def test_auto_poly_for_multi_repo_manifest(tmp_path: Path):
    config = DocsyncConfig(repo_mode="auto")
    mode = resolve_repo_mode(config, tmp_path, "owner/a", _manifest("owner/a", "owner/b"))
    assert mode == "poly"


def test_auto_ignores_empty_repos_for_poly_detection(tmp_path: Path):
    config = DocsyncConfig(repo_mode="auto")
    # Wildcard (empty) sources don't count as distinct repos.
    mode = resolve_repo_mode(config, tmp_path, "owner/a", _manifest("", "", "owner/a"))
    assert mode == "single"


# --- detect_repo_mode ---------------------------------------------------------


def test_detect_mono_when_code_beside_docs(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "intro.mdx").write_text("# Intro\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main():\n    pass\n")
    assert detect_repo_mode(tmp_path, "docs") == "mono"


def test_detect_single_when_docs_only(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "intro.mdx").write_text("# Intro\n")
    assert detect_repo_mode(tmp_path, "docs") == "single"


def test_detect_ignores_code_inside_docs_tree(tmp_path: Path):
    # A code snippet committed under the docs tree shouldn't read as a code surface.
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "intro.mdx").write_text("# Intro\n")
    (tmp_path / "docs" / "example.py").write_text("print('hi')\n")
    assert detect_repo_mode(tmp_path, "docs") == "single"


def test_detect_skips_noise_dirs(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "intro.mdx").write_text("# Intro\n")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = {}\n")
    assert detect_repo_mode(tmp_path, "docs") == "single"
