"""Tests for the adapter registry + the plain-Markdown adapter (Phase 3).

No network, no LLM: pure adapter behavior and the config-driven resolution seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync.adapters import (
    ADAPTERS,
    DocusaurusAdapter,
    MarkdownAdapter,
    MintlifyAdapter,
    make_adapter,
)
from docsync.validate import get_adapter


# --- registry --------------------------------------------------------------


def test_registry_has_both_adapters():
    assert ADAPTERS["mintlify"] is MintlifyAdapter
    assert ADAPTERS["markdown"] is MarkdownAdapter
    assert ADAPTERS["docusaurus"] is DocusaurusAdapter


def test_make_adapter_resolves_by_name():
    assert isinstance(make_adapter("mintlify"), MintlifyAdapter)
    assert isinstance(make_adapter("markdown"), MarkdownAdapter)
    assert isinstance(make_adapter("docusaurus"), DocusaurusAdapter)


def test_make_adapter_unknown_raises_with_known_names():
    with pytest.raises(ValueError, match="unknown adapter 'gitbook'"):
        make_adapter("gitbook")


# --- get_adapter threads the configured adapter ----------------------------


def test_get_adapter_defaults_to_mintlify():
    assert isinstance(get_adapter("reference/x.mdx"), MintlifyAdapter)


def test_get_adapter_uses_named_adapter():
    assert isinstance(get_adapter("reference/x.md", "markdown"), MarkdownAdapter)


def test_get_adapter_raises_when_named_adapter_does_not_own_page():
    # The markdown adapter does NOT own .mdx — selecting it for an .mdx page must fail
    # loudly rather than silently mis-handle the page.
    with pytest.raises(ValueError, match="does not own"):
        get_adapter("reference/x.mdx", "markdown")


# --- MarkdownAdapter behavior ----------------------------------------------


def test_markdown_owns_md_not_mdx():
    md = MarkdownAdapter()
    assert md.owns("a/b.md") and md.owns("c.markdown")
    assert not md.owns("a/b.mdx")


def test_markdown_page_extension_is_md():
    assert MarkdownAdapter().page_extension == ".md"
    assert MintlifyAdapter().page_extension == ".mdx"


def test_markdown_splits_yaml_frontmatter():
    text = '---\ntitle: "Hi"\ndescription: "d"\n---\n\n# Body\n\ntext\n'
    meta, body = MarkdownAdapter().split_frontmatter(text)
    assert meta["title"] == "Hi"
    assert body.strip().startswith("# Body")


def test_markdown_structural_signature_counts_fences():
    md = MarkdownAdapter()
    text = "intro\n\n```python\ncode\n```\n\n```mermaid\ngraph TD\n```\n"
    sig = md.structural_signature(text)
    assert sig["fence_count"] == 4  # two open + two close fence markers
    assert sig["mermaid_count"] == 1


def test_markdown_no_components_no_structural_problems():
    md = MarkdownAdapter()
    # Bare angle-bracket prose isn't a component — plain markdown reports no problems.
    assert md.structural_problems("a <Thing> that is not a component\n") == []


def test_markdown_check_links_is_noop(tmp_path: Path):
    assert MarkdownAdapter().check_links(tmp_path) == []
