"""Config validation + the friendly error surfaces (Phase 3 ease-of-use).

Covers extra='forbid' framing in load_config, the missing-manifest hint in
run/map/eval, and the `docsync explain` reference command. No network, no LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from docsync.cli import app
from docsync.config import ConfigError, DOCSYNC_DIR, docsync_dir, load_config
from docsync.models import DocsyncConfig

runner = CliRunner()


def _write_config(tmp_path: Path, body: str) -> Path:
    docs = tmp_path / "docs"
    (docs / DOCSYNC_DIR).mkdir(parents=True)
    (docsync_dir(docs) / "config.yml").write_text(body, encoding="utf-8")
    return docs


# --- extra='forbid' + framed errors -------------------------------------------


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError):
        DocsyncConfig.model_validate({"docs_rootx": "./mydocs"})


def test_load_config_frames_unknown_key(tmp_path: Path):
    docs = _write_config(tmp_path, "docs_rootx: ./mydocs\n")
    with pytest.raises(ConfigError) as exc:
        load_config(docs)
    msg = str(exc.value)
    assert "config.yml" in msg
    assert "docs_rootx" in msg
    assert "unknown field" in msg


def test_load_config_frames_bad_type(tmp_path: Path):
    docs = _write_config(tmp_path, "reviewers: not-a-list\n")
    with pytest.raises(ConfigError) as exc:
        load_config(docs)
    assert "reviewers" in str(exc.value)


def test_load_config_valid_still_loads(tmp_path: Path):
    docs = _write_config(tmp_path, "docs_root: docs\nthoroughness: high\n")
    config = load_config(docs)
    assert config.docs_root == "docs"
    assert config.thoroughness == "high"


def test_run_surfaces_config_error_not_traceback(tmp_path: Path):
    docs = _write_config(tmp_path, "typoed_field: 1\n")
    (docsync_dir(docs) / "manifest.yml").write_text("pages: []\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--docs-repo", str(docs), "--no-preflight"])
    assert result.exit_code == 2
    assert "typoed_field" in result.output
    assert "Traceback" not in result.output


# --- friendly missing-manifest hint -------------------------------------------


@pytest.mark.parametrize("command", ["run", "map", "eval"])
def test_missing_manifest_gives_init_hint(tmp_path: Path, command: str):
    docs = tmp_path / "docs"
    (docs / DOCSYNC_DIR).mkdir(parents=True)  # config dir but no manifest.yml
    args = {
        "run": ["run", "--docs-repo", str(docs), "--no-preflight"],
        "map": ["map", "--docs-repo", str(docs), "--src-repo", "owner/x", "--head", "abc"],
        "eval": ["eval", "--docs-repo", str(docs), "--golden", str(tmp_path / "g.json")],
    }[command]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "docsync init" in result.output
    assert "Traceback" not in result.output


# --- explain command ----------------------------------------------------------


def test_explain_lists_all_config_fields():
    result = runner.invoke(app, ["explain"])
    assert result.exit_code == 0
    # Every config field should appear in the listing.
    for name in DocsyncConfig.model_fields:
        assert name in result.output
    assert "repo_mode" in result.output
    assert "thoroughness" in result.output


def test_explain_single_field():
    result = runner.invoke(app, ["explain", "repo_mode"])
    assert result.exit_code == 0
    assert "repo_mode" in result.output
    assert "topology" in result.output.lower()


def test_explain_unknown_field_errors():
    result = runner.invoke(app, ["explain", "nonsense"])
    assert result.exit_code != 0
    assert "unknown config field" in result.output


def test_explain_manifest_schema():
    result = runner.invoke(app, ["explain", "manifest"])
    assert result.exit_code == 0
    assert "ManifestPage" in result.output
    assert "ManifestSource" in result.output
    assert "max_diff_lines" in result.output
