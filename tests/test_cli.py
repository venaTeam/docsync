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
# infer command (offline — fake client + patched run_infer / encoder)
# ---------------------------------------------------------------------------


def _infer_repo(tmp_path: Path) -> tuple[Path, Path]:
    docs = tmp_path / "docs"
    (docs / "reference").mkdir(parents=True)
    (docs / "reference" / "alerts.mdx").write_text(
        "---\ntitle: A\ndescription: d\n---\n\n# A\n", encoding="utf-8"
    )
    src = tmp_path / "svc"
    (src / "src").mkdir(parents=True)
    (src / "src" / "alerts.py").write_text("def get_alerts():\n    return []\n", encoding="utf-8")
    return docs, src


def test_infer_missing_embeddings_extra_exits_clean(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    docs, src = _infer_repo(tmp_path)
    # A client shaped like the real one (has .messages) so MeteredClient can wrap it;
    # the run still aborts cleanly when the encoder import fails.
    monkeypatch.setattr(
        "docsync.llm_backends.get_client", lambda backend: SimpleNamespace(messages=object())
    )
    monkeypatch.setattr(
        "docsync.embeddings.default_encoder",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no extra")),
    )
    result = runner.invoke(app, ["infer", "--docs-repo", str(docs), "--src-repo", f"svc={src}"])
    assert result.exit_code == 0
    assert "embeddings extra" in result.output


def test_infer_dry_run_reports_without_writing(tmp_path: Path, monkeypatch):
    from docsync.models import InferredPage, InferResult, ManifestSource

    docs, src = _infer_repo(tmp_path)
    canned = InferResult(pages=[InferredPage(
        page_path="reference/alerts.mdx", kind="reference", confidence=0.9, status="anchored",
        sources=[ManifestSource(repo="svc", globs=["src/alerts.py"], symbols=["get_alerts"])],
    )])
    monkeypatch.setattr("docsync.llm_backends.get_client", lambda backend: object())
    monkeypatch.setattr("docsync.infer.run_infer", lambda *a, **k: canned)

    result = runner.invoke(app, ["infer", "--docs-repo", str(docs), "--src-repo", f"svc={src}"])
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert not (docs / ".docsync" / "manifest.yml").exists()  # nothing written


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


# ---------------------------------------------------------------------------
# --polish flag wiring (config.readability_pass) on bootstrap
# ---------------------------------------------------------------------------


def _polish_smoke(tmp_path: Path, monkeypatch, extra_args: list[str]) -> dict:
    from docsync.models import BootstrapResult

    docs = tmp_path / "docs"
    docs.mkdir()
    src = tmp_path / "svc"
    src.mkdir()
    captured: dict = {}

    def _fake_run_bootstrap(repos, docs_repo, config, **kwargs):
        captured["readability_pass"] = config.readability_pass
        return BootstrapResult(repo="svc")  # no authored pages -> clean early exit

    monkeypatch.setattr("docsync.llm_backends.get_client", lambda backend: object())
    monkeypatch.setattr("docsync.bootstrap.run_bootstrap", _fake_run_bootstrap)
    result = runner.invoke(
        app, ["bootstrap", "--docs-repo", str(docs), "--src-repo", f"svc={src}", *extra_args]
    )
    assert result.exit_code == 0, result.output
    return captured


def test_bootstrap_polish_flag_enables_readability_pass(tmp_path: Path, monkeypatch):
    assert _polish_smoke(tmp_path, monkeypatch, ["--polish"])["readability_pass"] is True


def test_bootstrap_defaults_to_no_polish(tmp_path: Path, monkeypatch):
    assert _polish_smoke(tmp_path, monkeypatch, [])["readability_pass"] is False


# ---------------------------------------------------------------------------
# --backend flag vs config.backend precedence (on bootstrap)
# ---------------------------------------------------------------------------


def _backend_smoke(
    tmp_path: Path, monkeypatch, extra_args: list[str], config_yml: str | None = None
):
    from docsync.models import BootstrapResult

    docs = tmp_path / "docs"
    docs.mkdir()
    if config_yml is not None:
        (docs / ".docsync").mkdir()
        (docs / ".docsync" / "config.yml").write_text(config_yml, encoding="utf-8")
    src = tmp_path / "svc"
    src.mkdir()
    captured: dict = {}

    def _fake_get_client(backend):
        captured["backend"] = backend
        return object()

    monkeypatch.setattr("docsync.llm_backends.get_client", _fake_get_client)
    monkeypatch.setattr(
        "docsync.bootstrap.run_bootstrap",
        lambda repos, docs_repo, config, **kwargs: BootstrapResult(repo="svc"),
    )
    result = runner.invoke(
        app, ["bootstrap", "--docs-repo", str(docs), "--src-repo", f"svc={src}", *extra_args]
    )
    return result, captured


def test_backend_defaults_to_api(tmp_path: Path, monkeypatch):
    result, captured = _backend_smoke(tmp_path, monkeypatch, [])
    assert result.exit_code == 0, result.output
    assert captured["backend"] == "api"


def test_backend_from_config(tmp_path: Path, monkeypatch):
    result, captured = _backend_smoke(tmp_path, monkeypatch, [], config_yml="backend: cursor\n")
    assert result.exit_code == 0, result.output
    assert captured["backend"] == "cursor"


def test_backend_flag_overrides_config(tmp_path: Path, monkeypatch):
    result, captured = _backend_smoke(
        tmp_path, monkeypatch, ["--backend", "cursor"], config_yml="backend: claude-code\n"
    )
    assert result.exit_code == 0, result.output
    assert captured["backend"] == "cursor"


def test_backend_flag_rejects_unknown_value(tmp_path: Path, monkeypatch):
    result, captured = _backend_smoke(tmp_path, monkeypatch, ["--backend", "bogus"])
    assert result.exit_code != 0
    assert "must be one of" in result.output
    assert "backend" not in captured  # never reached get_client
