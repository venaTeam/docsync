"""Tests for Stage-5 validation gates and the Mintlify adapter.

All inline MDX strings — no network, no mintlify CLI invocation.
"""

from __future__ import annotations

import pytest

from docsync.adapters.mintlify import MintlifyAdapter
from docsync.models import ManifestPage
from docsync.validate import get_adapter, validate_page

PAGE = "docs/example.mdx"


def _page_with_frontmatter(title: str, body: str) -> str:
    return f"---\ntitle: {title}\ndescription: An example page\n---\n\n{body}"


# A representative MDX body with components, a code fence, and a mermaid block.
_BODY = """\
# Heading

Some intro prose explaining the feature.

<CardGroup cols={2}>
  <Card title="One" icon="bolt">First card body.</Card>
  <Card title="Two" icon="star">Second card body.</Card>
</CardGroup>

<Note>Remember to configure the thing.</Note>

```python
print("hello")
```

```mermaid
graph TD
  A --> B
```
"""


def _adapter() -> MintlifyAdapter:
    return MintlifyAdapter()


# ---------------------------------------------------------------------------
# get_adapter registry
# ---------------------------------------------------------------------------


def test_get_adapter_returns_mintlify_for_mdx():
    adapter = get_adapter("docs/page.mdx")
    assert isinstance(adapter, MintlifyAdapter)
    assert adapter.name == "mintlify"


def test_get_adapter_returns_mintlify_for_md():
    assert isinstance(get_adapter("docs/page.md"), MintlifyAdapter)


def test_get_adapter_raises_for_unknown_extension():
    with pytest.raises(ValueError):
        get_adapter("docs/page.rst")


# ---------------------------------------------------------------------------
# structural_signature
# ---------------------------------------------------------------------------


def test_structural_signature_counts():
    sig = _adapter().structural_signature(_BODY)
    assert sig["CardGroup"] == 1
    assert sig["/CardGroup"] == 1
    assert sig["Card"] == 2
    assert sig["/Card"] == 2
    assert sig["Note"] == 1
    assert sig["/Note"] == 1
    # 2 python/plain fences + 2 mermaid fences = 4 ``` markers total.
    assert sig["fence_count"] == 4
    assert sig["mermaid_count"] == 1
    # Flat int dict that compares by ==.
    assert all(isinstance(v, int) for v in sig.values())


def test_structural_signature_ignores_non_components():
    # A capitalised prose word in `<...>` that is not a known component is ignored.
    sig = _adapter().structural_signature("Plain text with <Bogus> not counted.")
    assert "Bogus" not in sig
    assert sig["fence_count"] == 0


# ---------------------------------------------------------------------------
# split_frontmatter
# ---------------------------------------------------------------------------


def test_split_frontmatter_parses_metadata_and_body():
    meta, body = _adapter().split_frontmatter(_page_with_frontmatter("Hello", "body text"))
    assert meta["title"] == "Hello"
    assert meta["description"] == "An example page"
    assert "body text" in body
    assert "title:" not in body


def test_split_frontmatter_handles_no_frontmatter():
    meta, body = _adapter().split_frontmatter("just a body, no fm")
    assert meta == {}
    assert body == "just a body, no fm"


# ---------------------------------------------------------------------------
# Hard gate 1 — frontmatter freeze
# ---------------------------------------------------------------------------


def test_body_edit_passes_frontmatter_gate():
    original = _page_with_frontmatter("Hello", "old body sentence.")
    new = _page_with_frontmatter("Hello", "new body sentence, slightly changed.")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


def test_changing_frozen_title_fails():
    original = _page_with_frontmatter("Hello", "body.")
    new = _page_with_frontmatter("Goodbye", "body.")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("title" in f for f in result.failures)


def test_changing_title_passes_when_allowed():
    original = _page_with_frontmatter("Hello", "body.")
    new = _page_with_frontmatter("Goodbye", "body.")
    manifest = ManifestPage(path=PAGE, allow_frontmatter_edit=True)
    result = validate_page(PAGE, original, new, manifest, _adapter())
    assert result.passed, result.failures


# ---------------------------------------------------------------------------
# Hard gate 2 — component integrity
# ---------------------------------------------------------------------------


def test_removing_a_card_fails():
    original = _page_with_frontmatter("Hello", _BODY)
    # Drop one <Card>...</Card> pair, unbalancing the component counts.
    mutilated = _BODY.replace(
        '  <Card title="Two" icon="star">Second card body.</Card>\n', "", 1
    )
    new = _page_with_frontmatter("Hello", mutilated)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("Card" in f for f in result.failures)


def test_benign_body_edit_keeps_components_passing():
    original = _page_with_frontmatter("Hello", _BODY)
    # Edit prose only; every component tag stays in place.
    edited = _BODY.replace(
        "Some intro prose explaining the feature.",
        "Some intro prose that explains the feature in more detail.",
    )
    new = _page_with_frontmatter("Hello", edited)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


# ---------------------------------------------------------------------------
# Hard gate 2 — fence / mermaid balance
# ---------------------------------------------------------------------------


def test_deleting_one_fence_line_fails():
    original = _page_with_frontmatter("Hello", _BODY)
    # Remove the closing ``` of the python block -> odd fence count + count change.
    broken = _BODY.replace('print("hello")\n```', 'print("hello")', 1)
    new = _page_with_frontmatter("Hello", broken)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("fence" in f.lower() for f in result.failures)


# ---------------------------------------------------------------------------
# Hard gate 2b — component well-formedness (nesting / balance)
# ---------------------------------------------------------------------------


def test_count_balanced_misnesting_fails_wellformed_gate():
    # Counts stay equal (CardGroup 1/1, Card 1/1) but the nesting is broken — the
    # signature gate passes, the new well-formedness gate must catch it.
    original = _page_with_frontmatter("Hello", "<CardGroup><Card>x</Card></CardGroup>")
    new = _page_with_frontmatter("Hello", "<CardGroup></Card><Card></CardGroup>")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("malformed MDX" in f for f in result.failures)


def test_structural_problems_ignores_tags_in_code():
    a = _adapter()
    assert a.structural_problems("```jsx\n<Card>\n```\nprose") == []
    assert a.structural_problems("inline `<Card>` reference") == []


def test_structural_problems_flags_unclosed_and_self_closing_ok():
    a = _adapter()
    assert a.structural_problems("<CardGroup><Card>x</CardGroup>")  # mis-nested/unclosed
    assert a.structural_problems("<CardGroup><Card/></CardGroup>") == []  # self-closing ok


# ---------------------------------------------------------------------------
# Hard gate 3 — diff-size guardrail
# ---------------------------------------------------------------------------


def test_tiny_edit_passes_diff_size():
    body = "\n".join(f"Line number {i} of prose." for i in range(40))
    original = _page_with_frontmatter("Hello", body)
    new = _page_with_frontmatter("Hello", body.replace("Line number 5", "Line number five"))
    manifest = ManifestPage(path=PAGE, max_diff_lines=5, max_diff_pct=0.1)
    result = validate_page(PAGE, original, new, manifest, _adapter())
    assert result.passed, result.failures


def test_replacing_most_of_page_exceeds_budget():
    body = "\n".join(f"Original line {i} with some content." for i in range(40))
    new_body = "\n".join(f"Rewritten line {i} entirely different now." for i in range(40))
    original = _page_with_frontmatter("Hello", body)
    new = _page_with_frontmatter("Hello", new_body)
    manifest = ManifestPage(path=PAGE, max_diff_lines=5, max_diff_pct=0.1)
    result = validate_page(PAGE, original, new, manifest, _adapter())
    assert not result.passed
    assert any("diff too large" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Hard gate 4 — non-empty / not-truncated
# ---------------------------------------------------------------------------


def test_empty_new_text_fails():
    original = _page_with_frontmatter("Hello", _BODY)
    result = validate_page(PAGE, original, "   ", ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("empty" in f for f in result.failures)


def test_drastically_shorter_new_text_fails():
    body = "\n".join(f"A fairly long line of documentation prose, number {i}." for i in range(60))
    original = _page_with_frontmatter("Hello", body)
    # Keep frontmatter so the freeze gate passes; gut the body so length collapses.
    new = _page_with_frontmatter("Hello", "tiny")
    # Generous diff budget so truncation (not diff-size) is the failing gate.
    manifest = ManifestPage(path=PAGE, max_diff_lines=500, max_diff_pct=1.0)
    result = validate_page(PAGE, original, new, manifest, _adapter())
    assert not result.passed
    assert any("truncated" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Soft gate — links never hard-fail
# ---------------------------------------------------------------------------


def test_link_check_is_soft(tmp_path, monkeypatch):
    original = _page_with_frontmatter("Hello", _BODY)
    new = _page_with_frontmatter("Hello", _BODY)

    # Force the adapter to report a broken link; it must surface as a warning,
    # not a failure, and must not flip `passed`.
    monkeypatch.setattr(
        MintlifyAdapter,
        "check_links",
        lambda self, docs_root: ["/docs/missing-page is broken"],
    )
    result = validate_page(
        PAGE, original, new, ManifestPage(path=PAGE), _adapter(),
        check_links=True, docs_root=tmp_path,
    )
    assert result.passed, result.failures
    assert any("broken" in w.lower() for w in result.warnings)


def test_check_links_never_raises(tmp_path):
    # No mintlify CLI in the test env -> must return [] without raising.
    assert _adapter().check_links(tmp_path) == []
