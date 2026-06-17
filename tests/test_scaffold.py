"""Unit tests for onboarding helpers (`docsync.scaffold`).

Two operations under test: scaffolding a starter `.docsync/` (and proving the
output is valid input to the real loaders) and `doctor`'s manifest-drift report
against real (temp) checkouts.
"""

from __future__ import annotations

from pathlib import Path

from docsync.config import (
    CONFIG_FILE,
    CURSORS_FILE,
    MANIFEST_FILE,
    docsync_dir,
    load_config,
    load_manifest,
)
from docsync.models import ModelConfig
from docsync.scaffold import DoctorReport, doctor, init_docs_repo


# ---------------------------------------------------------------------------
# init_docs_repo
# ---------------------------------------------------------------------------


def test_init_creates_three_artifacts(tmp_path: Path) -> None:
    created = init_docs_repo(tmp_path)

    base = docsync_dir(tmp_path)
    config_path = base / CONFIG_FILE
    manifest_path = base / MANIFEST_FILE
    cursors_path = base / CURSORS_FILE

    assert config_path.exists()
    assert manifest_path.exists()
    assert cursors_path.exists()
    assert set(created) == {config_path, manifest_path, cursors_path}
    assert cursors_path.read_text().strip() == "{}"


def test_init_is_idempotent_without_force(tmp_path: Path) -> None:
    first = init_docs_repo(tmp_path)
    assert first  # created something the first time

    # Mutate a file so we can prove the rerun didn't clobber it.
    config_path = docsync_dir(tmp_path) / CONFIG_FILE
    sentinel = config_path.read_text() + "\n# user edit\n"
    config_path.write_text(sentinel)

    second = init_docs_repo(tmp_path)
    assert second == []  # nothing written
    assert config_path.read_text() == sentinel  # preserved


def test_init_force_overwrites(tmp_path: Path) -> None:
    init_docs_repo(tmp_path)
    config_path = docsync_dir(tmp_path) / CONFIG_FILE
    config_path.write_text("# user edit\n")

    created = init_docs_repo(tmp_path, force=True)

    assert config_path in created
    assert "# user edit" not in config_path.read_text()


def test_scaffolded_config_roundtrips_through_loader(tmp_path: Path) -> None:
    init_docs_repo(tmp_path)
    config = load_config(tmp_path)

    # Defaults are seeded from the real ModelConfig, not invented.
    assert config.models.edit_model == ModelConfig().edit_model
    assert config.models.judge_model == ModelConfig().judge_model
    assert config.docs_root == "."
    assert config.reviewers == []


def test_scaffolded_manifest_roundtrips_through_loader(tmp_path: Path) -> None:
    init_docs_repo(tmp_path)
    manifest = load_manifest(tmp_path)

    assert len(manifest.pages) == 1
    page = manifest.pages[0]
    assert page.path == "example-page.mdx"
    assert page.max_diff_lines == 60
    assert page.sources[0].globs == ["src/routes/*.py"]
    assert page.sources[0].symbols == ["get_app"]


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _write_manifest(docs_repo: Path, body: str) -> None:
    path = docsync_dir(docs_repo) / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _make_checkout(tmp_path: Path) -> Path:
    """A fake source checkout with one file containing a known symbol."""
    checkout = tmp_path / "keep-api-gateway"
    (checkout / "src" / "routes").mkdir(parents=True)
    (checkout / "src" / "routes" / "alerts.py").write_text(
        "def get_app():\n    return 'app'\n"
    )
    return checkout


def test_doctor_ok_when_glob_and_symbol_resolve(tmp_path: Path) -> None:
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    (docs_repo / "alerts.mdx").write_text("# Alerts\n")
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: alerts.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs: ['src/routes/*.py']\n"
        "        symbols: ['get_app']\n",
    )
    checkout = _make_checkout(tmp_path)

    report = doctor(docs_repo, {"keephq/keep-api-gateway": checkout})

    assert isinstance(report, DoctorReport)
    assert report.ok is True
    assert report.dead_globs == []
    assert report.missing_pages == []
    assert report.missing_symbols == []
    assert report.unmapped_repos == []


def test_doctor_reports_dead_glob(tmp_path: Path) -> None:
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    (docs_repo / "alerts.mdx").write_text("# Alerts\n")
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: alerts.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs: ['src/nonexistent/*.py']\n",
    )
    checkout = _make_checkout(tmp_path)

    report = doctor(docs_repo, {"keephq/keep-api-gateway": checkout})

    assert report.ok is False
    assert len(report.dead_globs) == 1
    assert report.dead_globs[0].glob == "src/nonexistent/*.py"
    assert report.dead_globs[0].page == "alerts.mdx"


def test_doctor_reports_missing_page(tmp_path: Path) -> None:
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    # Note: no gone.mdx file written.
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: gone.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs: ['src/routes/*.py']\n",
    )
    checkout = _make_checkout(tmp_path)

    report = doctor(docs_repo, {"keephq/keep-api-gateway": checkout})

    assert report.ok is False
    assert "gone.mdx" in report.missing_pages


def test_doctor_reports_unmapped_repo(tmp_path: Path) -> None:
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    (docs_repo / "alerts.mdx").write_text("# Alerts\n")
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: alerts.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-workflows\n"
        "        globs: ['src/routes/*.py']\n",
    )

    # No checkout provided for keep-workflows.
    report = doctor(docs_repo, {})

    assert "keephq/keep-workflows" in report.unmapped_repos
    # Unmapped sources are skipped for glob/symbol checks, so no dead globs.
    assert report.dead_globs == []


def test_doctor_reports_missing_symbol_as_warning(tmp_path: Path) -> None:
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    (docs_repo / "alerts.mdx").write_text("# Alerts\n")
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: alerts.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs: ['src/routes/*.py']\n"
        "        symbols: ['vanished_symbol']\n",
    )
    checkout = _make_checkout(tmp_path)

    report = doctor(docs_repo, {"keephq/keep-api-gateway": checkout})

    # Missing symbol is a warning: it does not flip ok (glob still resolves).
    assert report.ok is True
    assert len(report.missing_symbols) == 1
    assert report.missing_symbols[0].symbol == "vanished_symbol"


def test_doctor_matches_repo_on_bare_name(tmp_path: Path) -> None:
    """A bare-name checkout key reconciles with an owner/name manifest repo."""
    docs_repo = tmp_path / "docs"
    docs_repo.mkdir()
    (docs_repo / "alerts.mdx").write_text("# Alerts\n")
    _write_manifest(
        docs_repo,
        "pages:\n"
        "  - path: alerts.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs: ['src/routes/*.py']\n",
    )
    checkout = _make_checkout(tmp_path)

    # Key is the bare last path segment, not owner/name.
    report = doctor(docs_repo, {"keep-api-gateway": checkout})

    assert report.ok is True
    assert report.unmapped_repos == []
