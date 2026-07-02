"""Tests for the Docusaurus adapter — admonition structure, on-disk links, sidebars nav.

No network, no LLM, no Docusaurus CLI: pure adapter behavior plus the validation gates
driven through it. Mirrors test_adapters.py / test_validate.py / test_nav.py.
"""

from __future__ import annotations

from pathlib import Path

from docsync.adapters import DocusaurusAdapter, make_adapter
from docsync.validate import get_adapter, validate_page


def _adapter() -> DocusaurusAdapter:
    return DocusaurusAdapter()


def _page(body: str) -> str:
    return f"---\ntitle: T\ndescription: D\n---\n{body}"


# --- registry / ownership --------------------------------------------------


def test_make_adapter_and_ownership():
    a = make_adapter("docusaurus")
    assert isinstance(a, DocusaurusAdapter)
    assert a.name == "docusaurus"
    assert a.page_extension == ".md"
    assert a.owns("a/b.md") and a.owns("c.mdx") and a.owns("d.markdown")
    assert not a.owns("e.json")


def test_get_adapter_threads_docusaurus():
    assert isinstance(get_adapter("guides/x.md", "docusaurus"), DocusaurusAdapter)
    assert isinstance(get_adapter("guides/x.mdx", "docusaurus"), DocusaurusAdapter)


def test_frontmatter_freeze_keys_and_split():
    a = _adapter()
    meta, body = a.split_frontmatter(_page("# Body\n\ntext\n"))
    assert meta["title"] == "T" and meta["description"] == "D"
    assert body.strip().startswith("# Body")
    assert a.frontmatter_keys_to_freeze() == ["title", "description"]


# --- structural signature --------------------------------------------------


def test_signature_counts_admonitions_with_matched_close_keys():
    a = _adapter()
    sig = a.structural_signature(_page(":::note\nhi\n:::\n\n:::tip\nyo\n:::\n"))
    assert sig["note"] == 1 and sig["/note"] == 1
    assert sig["tip"] == 1 and sig["/tip"] == 1


def test_signature_counts_components_and_fences():
    a = _adapter()
    text = _page("<Tabs>\n<TabItem value=\"a\">x</TabItem>\n</Tabs>\n\n```py\nc\n```\n")
    sig = a.structural_signature(text)
    assert sig["Tabs"] == 1 and sig["/Tabs"] == 1
    assert sig["TabItem"] == 1 and sig["/TabItem"] == 1
    assert sig["fence_count"] == 2 and sig["mermaid_count"] == 0


def test_signature_counts_mermaid():
    a = _adapter()
    sig = a.structural_signature(_page("```mermaid\ngraph TD\n```\n"))
    assert sig["mermaid_count"] == 1 and sig["fence_count"] == 2


def test_signature_ignores_admonitions_and_tags_in_code():
    a = _adapter()
    # A `:::note` and a `<Tabs>` shown as example text inside a fence are not structure.
    text = _page("```\n:::note\nexample\n:::\n<Tabs></Tabs>\n```\n")
    sig = a.structural_signature(text)
    assert "note" not in sig and "Tabs" not in sig
    assert sig["fence_count"] == 2


def test_signature_unknown_admonition_keyword_ignored():
    a = _adapter()
    # `:::foo` is not a recognized admonition — don't fingerprint it, and don't
    # misread its balancing closer as stray.
    sig = a.structural_signature(_page(":::foo\nx\n:::\n"))
    assert "foo" not in sig and "/foo" not in sig
    assert "/_unmatched" not in sig


def test_custom_admonition_closer_attributed_positionally():
    a = _adapter()
    # A custom block nested in a known one: the inner closer belongs to the custom
    # kind (uncounted), the outer to :::note — the known pair stays balanced.
    sig = a.structural_signature(_page(":::note\n:::release\nx\n:::\n:::\n"))
    assert sig["note"] == 1 and sig["/note"] == 1
    assert "/_unmatched" not in sig


# --- well-formedness / repair ----------------------------------------------


def test_problems_flag_unclosed_admonition():
    a = _adapter()
    assert a.structural_problems(_page(":::warning\noops\n")) == [
        "unclosed :::warning (no matching ::: close)"
    ]


def test_problems_flag_stray_admonition_closer():
    a = _adapter()
    assert a.structural_problems(_page("text\n:::\n")) == [
        "stray closing ::: with no matching admonition"
    ]


def test_problems_flag_component_misnesting():
    a = _adapter()
    problems = a.structural_problems(_page("<Tabs><TabItem></Tabs></TabItem>\n"))
    assert any("mismatched nesting" in p for p in problems)


def test_well_formed_admonitions_and_components_have_no_problems():
    a = _adapter()
    text = _page(":::note\nhi\n:::\n\n<Tabs>\n<TabItem>x</TabItem>\n</Tabs>\n")
    assert a.structural_problems(text) == []


def test_repair_closes_trailing_admonition():
    a = _adapter()
    repaired = a.repair_structure(_page(":::tip\nuse this\n"))
    assert a.structural_problems(repaired) == []
    assert repaired.rstrip().endswith(":::")


def test_repair_drops_stray_admonition_closer():
    a = _adapter()
    repaired = a.repair_structure(_page("body\n:::\n"))
    assert a.structural_problems(repaired) == []


def test_repair_is_noop_on_well_formed_text():
    a = _adapter()
    text = _page(":::note\nhi\n:::\n")
    assert a.repair_structure(text) == text


def test_repair_bails_on_component_misnesting():
    a = _adapter()
    text = _page("<Tabs><TabItem></Tabs></TabItem>\n")
    assert a.repair_structure(text) == text  # ambiguous — left for the validator


def test_custom_admonition_is_balanced_not_stray():
    a = _adapter()
    # Custom keywords (Docusaurus `admonitions.keywords` config) are legitimate:
    # a balanced custom block is well-formed and repair must leave it alone.
    text = _page(":::release\nShipped in 2.1.\n:::\n")
    assert a.structural_problems(text) == []
    assert a.repair_structure(text) == text


def test_repair_closes_unclosed_custom_admonition():
    a = _adapter()
    text = _page(":::release\nunclosed\n")
    assert a.structural_problems(text) == ["unclosed :::release (no matching ::: close)"]
    repaired = a.repair_structure(text)
    assert a.structural_problems(repaired) == []


# --- validation gates driven through the adapter ---------------------------


def test_adding_a_balanced_admonition_passes_structure_gate():
    a = _adapter()
    orig = _page("Body line one.\n")
    new = _page("Body line one.\n\n:::note\nA new caveat.\n:::\n")
    result = validate_page("x.md", orig, new, None, a)
    assert result.passed, result.failures


def test_removing_an_admonition_fails_structure_gate():
    a = _adapter()
    orig = _page("Body.\n\n:::note\ncaveat\n:::\n")
    new = _page("Body.\n")
    result = validate_page("x.md", orig, new, None, a)
    assert not result.passed
    assert any("count changed" in f for f in result.failures)


def test_adding_an_unbalanced_admonition_fails():
    a = _adapter()
    orig = _page("Body.\n")
    new = _page("Body.\n\n:::note\nunclosed caveat\n")
    result = validate_page("x.md", orig, new, None, a)
    assert not result.passed


def test_editing_page_with_custom_admonition_passes_gates():
    a = _adapter()
    # A page using a custom admonition must stay editable — the block is not
    # misread as a stray closer by the well-formedness gate.
    orig = _page("Body.\n\n:::release\nShipped in 2.0.\n:::\n")
    new = _page("Body, updated.\n\n:::release\nShipped in 2.0.\n:::\n")
    result = validate_page("x.md", orig, new, None, a)
    assert result.passed, result.failures


def test_adding_a_whole_tabs_container_fails():
    a = _adapter()
    orig = _page("Body.\n")
    new = _page("Body.\n\n<Tabs>\n<TabItem>x</TabItem>\n</Tabs>\n")
    result = validate_page("x.md", orig, new, None, a)
    # Tabs is a container, not in the additive-safe set — a fresh one is rejected.
    assert not result.passed


# --- link check (self-contained, on-disk) ----------------------------------


def _docusaurus_tree(tmp_path: Path) -> Path:
    (tmp_path / "docusaurus.config.js").write_text("module.exports = {};\n", encoding="utf-8")
    docs = tmp_path / "docs"
    (docs / "guides").mkdir(parents=True)
    (docs / "guides" / "setup.md").write_text(_page("setup\n"), encoding="utf-8")
    return docs


def test_check_links_flags_missing_relative_target_only(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    (docs / "intro.md").write_text(
        _page(
            "See [setup](./guides/setup.md), [gone](./guides/nope.md), "
            "[ext](https://example.com), [slug](/api/x), [anchor](#foo).\n"
        ),
        encoding="utf-8",
    )
    problems = _adapter().check_links(docs)
    assert len(problems) == 1
    assert "nope.md" in problems[0]


def test_check_links_resolves_extensionless_relative_link(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    (docs / "intro.md").write_text(
        _page("See [setup](./guides/setup).\n"), encoding="utf-8"
    )
    assert _adapter().check_links(docs) == []


def test_check_links_never_raises_on_empty_tree(tmp_path: Path):
    assert _adapter().check_links(tmp_path) == []


# --- navigation (sidebars.js) ----------------------------------------------


def test_set_nav_sections_writes_sidebars_and_reads_back(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    a = _adapter()
    assert a.ensure_valid_docs_json(docs) is True
    modified = a.set_nav_sections(
        docs, [("Getting Started", ["intro", "guides/setup"]), ("API", ["api/ref"])]
    )
    assert modified  # the sidebar file, path relative to docs_root
    sidebars = tmp_path / "sidebars.js"
    assert sidebars.exists()
    assert "Getting Started" in sidebars.read_text()
    assert sorted(a.nav_routes(docs)) == ["api/ref", "guides/setup", "intro"]


def test_set_nav_sections_is_idempotent(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    a = _adapter()
    a.set_nav_sections(docs, [("Getting Started", ["intro"])])
    assert a.set_nav_sections(docs, [("Getting Started", ["intro"])]) == []


def test_register_pages_in_nav_adds_to_group(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    a = _adapter()
    a.register_pages_in_nav(docs, ["guides/setup.md"], group="Guides")
    assert a.nav_routes(docs) == ["guides/setup"]


def test_ensure_valid_docs_json_seeds_empty_sidebar(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    a = _adapter()
    assert a.ensure_valid_docs_json(docs) is True
    assert (tmp_path / "sidebars.js").exists()
    assert a.ensure_valid_docs_json(docs) is False  # already present


def test_nav_routes_empty_when_no_sidebar(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    assert _adapter().nav_routes(docs) == []


def test_route_of_strips_number_prefixes_like_docusaurus():
    a = _adapter()
    # Docusaurus's default numberPrefixParser: `01-intro.md` has doc id `intro`,
    # and directory segments are stripped too. A sidebar entry with the raw
    # prefixed path would fail the build with an unknown-document error.
    assert a._route_of("01-intro.md") == "intro"
    assert a._route_of("guides/02-setup.mdx") == "guides/setup"
    assert a._route_of("1. getting-started/3 - install.md") == "getting-started/install"
    # No digit prefix or no separator: left alone.
    assert a._route_of("v2-migration.md") == "v2-migration"
    assert a._route_of("guides/setup.md") == "guides/setup"


def test_register_pages_in_nav_uses_stripped_doc_ids(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    a = _adapter()
    a.register_pages_in_nav(docs, ["guides/01-setup.md"], group="Guides")
    assert a.nav_routes(docs) == ["guides/setup"]


def test_nav_routes_and_dedup_cover_all_sidebars(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    sidebars = tmp_path / "sidebars.js"
    sidebars.write_text(
        "module.exports = "
        '{"docsSidebar": ["intro"], '
        '"apiSidebar": [{"type": "category", "label": "API", "items": ["api/ref"]}]};\n',
        encoding="utf-8",
    )
    a = _adapter()
    # Reads ids from every sidebar, not just the first.
    assert sorted(a.nav_routes(docs)) == ["api/ref", "intro"]
    # A page already navigable from the second sidebar is not re-registered...
    assert a.register_pages_in_nav(docs, ["api/ref.md"], group="API") == []
    # ...and a new page for a category living in the second sidebar lands there.
    a.register_pages_in_nav(docs, ["api/errors.md"], group="API")
    assert sorted(a.nav_routes(docs)) == ["api/errors", "api/ref", "intro"]
    assert '"api/errors"' in sidebars.read_text()


def test_set_nav_sections_refuses_to_clobber_handwritten_sidebar(tmp_path: Path):
    docs = _docusaurus_tree(tmp_path)
    # A real, hand-written JS sidebar with a require() the JSON reader can't parse.
    handwritten = "module.exports = require('./sidebars.config.js');\n"
    (tmp_path / "sidebars.js").write_text(handwritten, encoding="utf-8")
    a = _adapter()
    assert a.set_nav_sections(docs, [("Guides", ["intro"])]) == []
    assert (tmp_path / "sidebars.js").read_text() == handwritten  # untouched


# --- authoring hint --------------------------------------------------------


def test_authoring_hint_mentions_admonitions_not_mdx_components():
    hint = _adapter().authoring_components_hint()
    assert ":::note" in hint and ":::warning" in hint
    assert "<Note>" not in hint


def test_plan_prompt_uses_adapter_page_extension():
    from docsync.bootstrap import build_plan_prompt

    system, _ = build_plan_prompt(
        [], existing_routes=[], existing_pages=set(), page_extension=".md"
    )
    assert "introduction.md" in system and "introduction.mdx" not in system
