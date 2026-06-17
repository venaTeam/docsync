"""Tests for Stage-B1 whole-repo ingest (`docsync.ingest`).

Read-only walk + symbol extraction. Everything is inline / `tmp_path`; no network
and no real source checkout is required.
"""

from __future__ import annotations

from pathlib import Path

from docsync.ingest import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_INCLUDE,
    extract_symbols,
    read_excerpt,
    walk_repo,
)
from docsync.models import RepoDigest, SourceUnit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_constants() -> None:
    assert DEFAULT_INCLUDE == ("*.py", "*.ts", "*.tsx")
    for d in ("node_modules", ".git", "tests", "__pycache__", ".venv", ".docsync"):
        assert d in DEFAULT_EXCLUDE_DIRS


# ---------------------------------------------------------------------------
# extract_symbols — Python (AST)
# ---------------------------------------------------------------------------


def test_extract_symbols_python_top_level_only() -> None:
    text = (
        "def f():\n"
        "    helper_inner = 1\n"
        "    def nested():\n"
        "        pass\n"
        "\n"
        "async def g():\n"
        "    pass\n"
        "\n"
        "class C:\n"
        "    def method(self):\n"
        "        pass\n"
        "\n"
        "X = 1\n"
        "Y: int = 2\n"
    )
    symbols = extract_symbols("mod.py", text)
    # Top-level defs/classes and module-level assignments ARE returned.
    assert "f" in symbols
    assert "g" in symbols
    assert "C" in symbols
    assert "X" in symbols
    assert "Y" in symbols
    # Nested defs, methods and indented assignments are NOT.
    assert "nested" not in symbols
    assert "method" not in symbols
    assert "helper_inner" not in symbols


def test_extract_symbols_python_syntax_error_fallback() -> None:
    # The body is broken (so ast.parse fails) but a parseable `def foo(` line
    # remains; the regex fallback must still surface `foo`.
    text = "def foo(:\n    this is not valid python @@@\n"
    symbols = extract_symbols("broken.py", text)
    assert "foo" in symbols


# ---------------------------------------------------------------------------
# extract_symbols — TypeScript (export regex)
# ---------------------------------------------------------------------------


def test_extract_symbols_typescript_exports() -> None:
    text = (
        "export function a() {}\n"
        "export const b = 1;\n"
        "export default class C {}\n"
        "export interface I {}\n"
        "export type T = string;\n"
        "function notExported() {}\n"
        "const alsoNot = 2;\n"
    )
    symbols = extract_symbols("mod.ts", text)
    assert "a" in symbols
    assert "b" in symbols
    assert "C" in symbols
    assert "I" in symbols
    assert "T" in symbols
    assert "notExported" not in symbols
    assert "alsoNot" not in symbols


def test_extract_symbols_other_kind_is_empty() -> None:
    assert extract_symbols("x.md", "# Heading\nsome **markdown**\n") == []


# ---------------------------------------------------------------------------
# walk_repo
# ---------------------------------------------------------------------------


def _build_tree(root: Path) -> None:
    (root / "src" / "routes").mkdir(parents=True)
    (root / "src" / "routes" / "alerts.py").write_text("def get_app():\n    return 1\n")
    (root / "src" / "client.ts").write_text("export const api = 1;\n")
    (root / "README.md").write_text("# readme\n")  # not in include globs
    # Excluded dirs that must be pruned.
    (root / "node_modules").mkdir()
    (root / "node_modules" / "junk.py").write_text("def junk():\n    pass\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    (root / ".git").mkdir()
    (root / ".git" / "x.py").write_text("def gitthing():\n    pass\n")


def test_walk_repo_prunes_excluded_and_populates(tmp_path: Path) -> None:
    _build_tree(tmp_path)

    digest = walk_repo(tmp_path)
    assert isinstance(digest, RepoDigest)

    paths = [u.path for u in digest.units]
    # Only included-glob files under non-excluded dirs appear.
    assert paths == ["src/client.ts", "src/routes/alerts.py"]
    # Sorted by path.
    assert paths == sorted(paths)
    # Excluded dirs pruned.
    assert "node_modules/junk.py" not in paths
    assert "tests/test_x.py" not in paths
    assert ".git/x.py" not in paths
    # README.md excluded by include globs.
    assert "README.md" not in paths

    by_path = {u.path: u for u in digest.units}
    assert by_path["src/routes/alerts.py"].kind == "python"
    assert "get_app" in by_path["src/routes/alerts.py"].symbols
    assert by_path["src/client.ts"].kind == "typescript"
    assert "api" in by_path["src/client.ts"].symbols


def test_walk_repo_never_writes(tmp_path: Path) -> None:
    _build_tree(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}

    walk_repo(tmp_path)

    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    # No new files appeared and nothing was rewritten.
    assert set(before) == set(after)
    assert before == after


def test_walk_repo_max_files_cap(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 2\n")
    (tmp_path / "c.py").write_text("C = 3\n")

    digest = walk_repo(tmp_path, max_files=2)
    assert len(digest.units) <= 2
    assert all(isinstance(u, SourceUnit) for u in digest.units)


# ---------------------------------------------------------------------------
# read_excerpt
# ---------------------------------------------------------------------------


def test_read_excerpt_truncates(tmp_path: Path) -> None:
    big = "x" * 5000
    (tmp_path / "big.py").write_text(big)

    excerpt = read_excerpt(tmp_path, "big.py", max_chars=100)
    assert len(excerpt) < len(big)
    assert "truncated" in excerpt
    assert excerpt.startswith("x" * 100)


def test_read_excerpt_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_excerpt(tmp_path, "does-not-exist.py") == ""
