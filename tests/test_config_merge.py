"""Tests for `docsync.config.merge_manifest_pages` (bootstrap manifest append).

Exercises a fresh-repo create, comment preservation on an existing manifest, and
idempotency. All on `tmp_path`; no network.
"""

from __future__ import annotations

from pathlib import Path

from docsync.config import (
    DOCSYNC_DIR,
    MANIFEST_FILE,
    docsync_dir,
    load_manifest,
    merge_manifest_pages,
)
from docsync.models import ManifestPage, ManifestSource


def _manifest_path(docs_repo: Path) -> Path:
    return docsync_dir(docs_repo) / MANIFEST_FILE


def _page(path: str, *, repo: str, globs: list[str], symbols: list[str]) -> ManifestPage:
    return ManifestPage(
        path=path,
        sources=[ManifestSource(repo=repo, globs=globs, symbols=symbols)],
    )


# ---------------------------------------------------------------------------
# Fresh repo
# ---------------------------------------------------------------------------


def test_merge_creates_manifest_and_roundtrips(tmp_path: Path) -> None:
    page = _page(
        "reference/alerts.mdx",
        repo="keephq/keep-api-gateway",
        globs=["src/routes/*.py"],
        symbols=["get_app"],
    )

    added = merge_manifest_pages(tmp_path, [page])

    assert added == ["reference/alerts.mdx"]
    assert _manifest_path(tmp_path).exists()

    manifest = load_manifest(tmp_path)
    assert len(manifest.pages) == 1
    loaded = manifest.pages[0]
    assert loaded.path == "reference/alerts.mdx"
    assert loaded.sources[0].repo == "keephq/keep-api-gateway"
    assert loaded.sources[0].globs == ["src/routes/*.py"]
    assert loaded.sources[0].symbols == ["get_app"]


# ---------------------------------------------------------------------------
# Comment preservation
# ---------------------------------------------------------------------------


def test_merge_preserves_existing_comments(tmp_path: Path) -> None:
    base = docsync_dir(tmp_path)
    base.mkdir(parents=True, exist_ok=True)
    _manifest_path(tmp_path).write_text(
        "# CURATED COMMENT\n"
        "pages:\n"
        "  - path: existing.mdx\n"
        "    sources:\n"
        "      - repo: keephq/keep-api-gateway\n"
        "        globs:\n"
        "          - src/routes/*.py\n"
    )

    new_page = _page(
        "reference/new.mdx",
        repo="keephq/keep-workflows",
        globs=["src/workflows/*.py"],
        symbols=["WorkflowManager"],
    )
    added = merge_manifest_pages(tmp_path, [new_page])

    assert added == ["reference/new.mdx"]
    content = _manifest_path(tmp_path).read_text()
    # Curated comment survives the round-trip dump.
    assert "# CURATED COMMENT" in content

    manifest = load_manifest(tmp_path)
    paths = {p.path for p in manifest.pages}
    assert paths == {"existing.mdx", "reference/new.mdx"}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_merge_is_idempotent_on_path(tmp_path: Path) -> None:
    page = _page(
        "reference/alerts.mdx",
        repo="keephq/keep-api-gateway",
        globs=["src/routes/*.py"],
        symbols=["get_app"],
    )
    first = merge_manifest_pages(tmp_path, [page])
    assert first == ["reference/alerts.mdx"]

    # Same path again -> nothing added, no duplicate.
    second = merge_manifest_pages(tmp_path, [page])
    assert second == []

    manifest = load_manifest(tmp_path)
    assert [p.path for p in manifest.pages] == ["reference/alerts.mdx"]
    assert DOCSYNC_DIR in str(_manifest_path(tmp_path))
