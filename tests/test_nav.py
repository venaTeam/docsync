"""Tests for `MintlifyAdapter` navigation registration.

Inline `docs.json` / `mint.json` written into `tmp_path`; no network, no CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

from docsync.adapters.mintlify import MintlifyAdapter

_GROUP = "Reference (docsync)"


def _adapter() -> MintlifyAdapter:
    return MintlifyAdapter()


def _write_docs_json(root: Path, groups: list[dict]) -> Path:
    path = root / "docs.json"
    path.write_text(json.dumps({"navigation": {"groups": groups}}, indent=2) + "\n")
    return path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# register_pages_in_nav
# ---------------------------------------------------------------------------


def test_register_adds_pages_in_new_group(tmp_path: Path) -> None:
    path = _write_docs_json(
        tmp_path, [{"group": "Existing", "pages": ["guides/intro"]}]
    )

    modified = _adapter().register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx", "reference/incidents.mdx"], group=_GROUP
    )

    assert modified == ["docs.json"]
    data = _load(path)
    groups = {g["group"]: g for g in data["navigation"]["groups"]}
    # Existing group untouched.
    assert groups["Existing"]["pages"] == ["guides/intro"]
    # New group created with extension-stripped, no-leading-slash refs.
    assert _GROUP in groups
    assert groups[_GROUP]["pages"] == ["reference/alerts", "reference/incidents"]


def test_register_is_idempotent(tmp_path: Path) -> None:
    path = _write_docs_json(tmp_path, [{"group": "Existing", "pages": []}])
    adapter = _adapter()

    first = adapter.register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )
    assert first == ["docs.json"]
    after_first = _load(path)

    second = adapter.register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )
    assert second == []
    # File content unchanged on the no-op re-run.
    assert _load(path) == after_first


def test_register_skips_page_present_in_another_group(tmp_path: Path) -> None:
    path = _write_docs_json(
        tmp_path, [{"group": "Other", "pages": ["reference/alerts"]}]
    )

    modified = _adapter().register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )

    # Already present elsewhere (set-union across groups) -> nothing to do.
    assert modified == []
    data = _load(path)
    group_names = {g["group"] for g in data["navigation"]["groups"]}
    assert _GROUP not in group_names


# ---------------------------------------------------------------------------
# nav_routes
# ---------------------------------------------------------------------------


def test_nav_routes_returns_all_refs(tmp_path: Path) -> None:
    _write_docs_json(
        tmp_path,
        [
            {"group": "A", "pages": ["a/one", "a/two"]},
            {"group": "B", "pages": ["b/three"]},
        ],
    )

    routes = _adapter().nav_routes(tmp_path)
    assert set(routes) == {"a/one", "a/two", "b/three"}


# ---------------------------------------------------------------------------
# docs.json + legacy mint.json
# ---------------------------------------------------------------------------


def test_register_mirrors_into_both_nav_files(tmp_path: Path) -> None:
    docs_path = _write_docs_json(tmp_path, [{"group": "Existing", "pages": []}])
    mint_path = tmp_path / "mint.json"
    # Legacy mint.json: navigation is a flat list of {group, pages}.
    mint_path.write_text(
        json.dumps({"navigation": [{"group": "Existing", "pages": []}]}, indent=2) + "\n"
    )

    modified = _adapter().register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )

    assert set(modified) == {"docs.json", "mint.json"}
    docs_groups = {g["group"]: g for g in _load(docs_path)["navigation"]["groups"]}
    assert docs_groups[_GROUP]["pages"] == ["reference/alerts"]
    mint_groups = {g["group"]: g for g in _load(mint_path)["navigation"]}
    assert mint_groups[_GROUP]["pages"] == ["reference/alerts"]


def test_register_works_with_only_mint_json(tmp_path: Path) -> None:
    mint_path = tmp_path / "mint.json"
    mint_path.write_text(
        json.dumps({"navigation": [{"group": "Existing", "pages": []}]}, indent=2) + "\n"
    )

    modified = _adapter().register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )

    assert modified == ["mint.json"]
    mint_groups = {g["group"]: g for g in _load(mint_path)["navigation"]}
    assert mint_groups[_GROUP]["pages"] == ["reference/alerts"]


def test_register_with_no_nav_file_returns_empty(tmp_path: Path) -> None:
    modified = _adapter().register_pages_in_nav(
        tmp_path, ["reference/alerts.mdx"], group=_GROUP
    )
    assert modified == []


# ---------------------------------------------------------------------------
# set_nav_sections — ordered, idempotent
# ---------------------------------------------------------------------------


def test_set_nav_sections_writes_groups_in_order(tmp_path):
    _write_docs_json(tmp_path, [{"group": "Existing", "pages": ["intro"]}])
    a = _adapter()
    modified = a.set_nav_sections(tmp_path, [
        ("Getting Started", ["getting-started/intro"]),
        ("Architecture", ["architecture/flow"]),
        ("Reference", ["reference/alerts"]),
    ])
    assert "docs.json" in modified
    nav = json.loads((tmp_path / "docs.json").read_text())
    order = [g["group"] for g in nav["navigation"]["groups"]]
    # Existing group stays first; new sections appended in the given reading-flow order.
    assert order == ["Existing", "Getting Started", "Architecture", "Reference"]
    pages = {g["group"]: g["pages"] for g in nav["navigation"]["groups"]}
    assert pages["Architecture"] == ["architecture/flow"]


def test_set_nav_sections_is_idempotent(tmp_path):
    _write_docs_json(tmp_path, [])
    a = _adapter()
    sections = [("Reference", ["reference/a", "reference/b"])]
    a.set_nav_sections(tmp_path, sections)
    first = (tmp_path / "docs.json").read_text()
    assert a.set_nav_sections(tmp_path, sections) == []  # no change on re-run
    assert (tmp_path / "docs.json").read_text() == first


def test_set_nav_sections_skips_route_already_in_another_group(tmp_path):
    _write_docs_json(tmp_path, [{"group": "Existing", "pages": ["reference/a"]}])
    a = _adapter()
    a.set_nav_sections(tmp_path, [("Reference", ["reference/a", "reference/b"])])
    nav = json.loads((tmp_path / "docs.json").read_text())
    pages = {g["group"]: g["pages"] for g in nav["navigation"]["groups"]}
    assert pages["Existing"] == ["reference/a"]          # untouched
    assert pages["Reference"] == ["reference/b"]         # only the new route added


# ---------------------------------------------------------------------------
# ensure_valid_docs_json
# ---------------------------------------------------------------------------


def test_ensure_valid_docs_json_creates_when_missing(tmp_path):
    a = _adapter()
    assert a.ensure_valid_docs_json(tmp_path) is True
    nav = json.loads((tmp_path / "docs.json").read_text())
    assert "colors" in nav and "navigation" in nav and nav["navigation"]["groups"] == []
    # Idempotent: a valid file is left unchanged.
    assert a.ensure_valid_docs_json(tmp_path) is False


def test_ensure_valid_docs_json_adds_missing_required_field(tmp_path):
    # A minimal docs.json lacking `colors` (the error we hit live) gets it added.
    (tmp_path / "docs.json").write_text(
        json.dumps({"name": "X", "navigation": {"groups": []}}, indent=2) + "\n"
    )
    a = _adapter()
    assert a.ensure_valid_docs_json(tmp_path) is True
    assert "colors" in json.loads((tmp_path / "docs.json").read_text())
