"""Tests for `docsync.validate.validate_new_page` (from-scratch / bootstrap gates).

Inline MDX strings only; no network and no Mintlify CLI is invoked. The broken-link
gate is exercised by monkeypatching the adapter so the soft behaviour is testable.
"""

from __future__ import annotations

from docsync.adapters.mintlify import MintlifyAdapter
from docsync.validate import _NEW_PAGE_MIN_CHARS, validate_new_page

PAGE = "reference/example.mdx"


def _adapter() -> MintlifyAdapter:
    return MintlifyAdapter()


def _page(frontmatter: str, body: str) -> str:
    return f"---\n{frontmatter}\n---\n\n{body}"


# A balanced, long-enough body that satisfies every gate.
_GOOD_BODY = (
    "# Example\n\n"
    + ("Some real documentation prose explaining the feature in detail. " * 6)
    + "\n\n"
    "<CardGroup cols={2}>\n"
    '  <Card title="One">First card body.</Card>\n'
    '  <Card title="Two">Second card body.</Card>\n'
    "</CardGroup>\n"
)


def _good_frontmatter() -> str:
    return "title: Example Page\ndescription: A real description"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_well_formed_new_page_passes() -> None:
    text = _page(_good_frontmatter(), _GOOD_BODY)
    assert len(text) >= _NEW_PAGE_MIN_CHARS  # guard the fixture itself
    result = validate_new_page(PAGE, text, _adapter())
    assert result.passed, result.failures


# ---------------------------------------------------------------------------
# Frontmatter gates
# ---------------------------------------------------------------------------


def test_missing_title_fails() -> None:
    text = _page("description: A real description", _GOOD_BODY)
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("title" in f for f in result.failures)


def test_empty_title_fails() -> None:
    text = _page('title: ""\ndescription: A real description', _GOOD_BODY)
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("title" in f for f in result.failures)


def test_missing_description_fails() -> None:
    text = _page("title: Example Page", _GOOD_BODY)
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("description" in f for f in result.failures)


def test_unparseable_frontmatter_fails() -> None:
    # Broken YAML inside the frontmatter fence.
    text = "---\ntitle: [unterminated\n  : : :\n---\n\n" + _GOOD_BODY
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("frontmatter" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Length gates
# ---------------------------------------------------------------------------


def test_empty_body_fails() -> None:
    text = _page(_good_frontmatter(), "   \n  \n")
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    # The page is non-empty as a whole (has frontmatter) but short -> stub.
    assert any("stub" in f for f in result.failures)


def test_whitespace_only_text_fails_empty() -> None:
    result = validate_new_page(PAGE, "   \n\t\n  ", _adapter())
    assert not result.passed
    assert any("empty" in f for f in result.failures)


def test_short_page_fails_as_stub() -> None:
    text = _page(_good_frontmatter(), "tiny")
    assert len(text) < _NEW_PAGE_MIN_CHARS
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("stub" in f for f in result.failures)


# ---------------------------------------------------------------------------
# Component well-formedness / fences
# ---------------------------------------------------------------------------


def test_misnested_components_fail() -> None:
    # Counts balance (CardGroup 1/1, Card 1/1) but nesting is broken.
    body = "<CardGroup></Card><Card></CardGroup>\n" + ("filler prose. " * 30)
    text = _page(_good_frontmatter(), body)
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("malformed MDX" in f for f in result.failures)


def test_odd_fences_fail() -> None:
    body = "An opening fence with no close.\n\n```python\nprint('x')\n" + (
        "trailing prose. " * 20
    )
    text = _page(_good_frontmatter(), body)
    result = validate_new_page(PAGE, text, _adapter())
    assert not result.passed
    assert any("fence" in f.lower() for f in result.failures)


# ---------------------------------------------------------------------------
# Soft link gate
# ---------------------------------------------------------------------------


def test_broken_link_check_is_soft(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        MintlifyAdapter,
        "check_links",
        lambda self, docs_root: ["x is broken"],
    )
    text = _page(_good_frontmatter(), _GOOD_BODY)
    result = validate_new_page(
        PAGE, text, _adapter(), check_links=True, docs_root=tmp_path
    )
    assert result.passed, result.failures
    assert any("broken" in w.lower() for w in result.warnings)
