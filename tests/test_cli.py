"""CLI tests — the bits with branching worth guarding (the doctor pre-flight gate).

Uses Typer's CliRunner; no network and no LLM (the pre-flight aborts before either).
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from docsync.cli import app

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
