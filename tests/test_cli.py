"""CLI tests — the bits with branching worth guarding (the doctor pre-flight gate).

Uses Typer's CliRunner; no network and no LLM (the pre-flight aborts before either).
"""

from __future__ import annotations

import json

import pytest
from pathlib import Path

import typer
from typer.testing import CliRunner

from docsync.cli import _base_or_cursor, _format_symbol_list, app

runner = CliRunner()


def _docs_repo(tmp_path: Path, manifest_yml: str) -> Path:
    root = tmp_path / "docs"
    (root / ".docsync").mkdir(parents=True)
    (root / ".docsync" / "manifest.yml").write_text(manifest_yml, encoding="utf-8")
    return root


def test_run_preflight_aborts_on_missing_manifest_page(tmp_path: Path):
    # Manifest references a page that doesn't exist -> abort before any LLM spend.
    docs = _docs_repo(tmp_path, "pages:\n  - path: ghost.mdx\n")
    result = runner.invoke(app, ["run", "--docs-repo", str(docs)])
    assert result.exit_code == 2
    assert "preflight failed" in result.output
    assert "ghost.mdx" in result.output


def test_run_no_preflight_bypasses_the_gate(tmp_path: Path):
    # With --no-preflight the missing page no longer aborts at the gate; the run gets
    # past it and fails later for an unrelated reason (no diff source provided).
    docs = _docs_repo(tmp_path, "pages:\n  - path: ghost.mdx\n")
    result = runner.invoke(app, ["run", "--docs-repo", str(docs), "--no-preflight"])
    assert "preflight failed" not in result.output
    assert result.exit_code != 0  # still errors, just not at the pre-flight gate


# ---------------------------------------------------------------------------
# map/run ergonomics — quiet symbol output + cursor-aware base
# ---------------------------------------------------------------------------


def test_format_symbol_list_truncates_with_tail():
    assert _format_symbol_list([]) == "—"
    assert _format_symbol_list(["a", "b"]) == "a, b"
    many = [f"s{i}" for i in range(20)]
    out = _format_symbol_list(many, limit=12)
    assert out.startswith("s0, s1,")
    assert out.endswith("(+8 more)")
    assert out.count(",") == 12  # 12 shown names + the tail separator


def _cursored_docs_repo(tmp_path: Path, cursors: dict) -> Path:
    docs = _docs_repo(tmp_path, "pages: []\n")
    state = docs / ".docsync" / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "cursors.json").write_text(json.dumps(cursors), encoding="utf-8")
    return docs


def test_base_or_cursor_resolves_by_repo_key(tmp_path: Path):
    # The cursor is keyed by an owner/name repo; a bare-name --src-repo still matches
    # via _repo_key normalization.
    docs = _cursored_docs_repo(tmp_path, {"venaTeam/docsync": "abc1234"})
    assert _base_or_cursor(docs, "docsync") == "abc1234"
    assert _base_or_cursor(docs, "venaTeam/docsync") == "abc1234"


def test_base_or_cursor_errors_without_a_cursor(tmp_path: Path):
    docs = _cursored_docs_repo(tmp_path, {})
    with pytest.raises(typer.BadParameter):
        _base_or_cursor(docs, "docsync")
