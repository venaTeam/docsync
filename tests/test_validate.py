"""Tests for Stage-5 validation gates and the Mintlify adapter.

All inline MDX strings — no network, no mintlify CLI invocation.
"""

from __future__ import annotations

import pytest

from docsync.adapters.mintlify import MintlifyAdapter
from docsync.models import ManifestPage
from docsync.validate import get_adapter, validate_new_page, validate_page

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


def test_structural_signature_counts_self_closing_as_balanced():
    # `<Card/>` is its own close — it must register an open AND a close so the additive
    # gate sees balanced growth (regression: it used to count opens +1, closes +0).
    sig = _adapter().structural_signature("<CardGroup><Card/></CardGroup>")
    assert sig["Card"] == 1
    assert sig["/Card"] == 1


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
# Hard gate 2 — additive component growth (recovers the recall gap: documenting a
# new field/stage often means one more balanced <Note>/<Step>/<Card>)
# ---------------------------------------------------------------------------

_STEPS_BODY = (
    "# Pipeline\n\n<Steps>\n"
    '  <Step title="A">do a</Step>\n'
    '  <Step title="B">do b</Step>\n'
    "</Steps>\n"
)


def test_adding_a_balanced_note_passes():
    original = _page_with_frontmatter("Hello", _BODY)
    new = _page_with_frontmatter("Hello", _BODY + "\n<Note>A new caveat the diff added.</Note>\n")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


def test_adding_a_step_to_existing_steps_passes():
    original = _page_with_frontmatter("Hello", _STEPS_BODY)
    grown = _STEPS_BODY.replace(
        "</Steps>", '  <Step title="C">do c (a newly documented stage)</Step>\n</Steps>'
    )
    new = _page_with_frontmatter("Hello", grown)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


def test_adding_a_card_to_existing_group_passes():
    original = _page_with_frontmatter("Hello", _BODY)
    grown = _BODY.replace(
        "</CardGroup>", '  <Card title="Three" icon="gear">Third card.</Card>\n</CardGroup>'
    )
    new = _page_with_frontmatter("Hello", grown)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


def test_adding_a_whole_container_still_fails():
    # A new <Steps> container (not a safe leaf) is a real structural change — reject.
    original = _page_with_frontmatter("Hello", _BODY)
    new = _page_with_frontmatter("Hello", _BODY + "\n<Steps>\n  <Step>x</Step>\n</Steps>\n")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("Steps" in f for f in result.failures)


def test_adding_a_self_closing_component_passes():
    # A new self-closing <Frame/>-style leaf (here <Card/>) is balanced by definition;
    # the additive gate must not reject it as opens +1, closes +0.
    original = _page_with_frontmatter("Hello", _BODY)
    grown = _BODY.replace(
        "</CardGroup>", '  <Card title="Quick" icon="bolt"/>\n</CardGroup>'
    )
    new = _page_with_frontmatter("Hello", grown)
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert result.passed, result.failures


def test_adding_unbalanced_open_tag_fails():
    # An extra <Note> with no closing </Note> must not slip through the additive carve-out.
    original = _page_with_frontmatter("Hello", _BODY)
    new = _page_with_frontmatter("Hello", _BODY + "\n<Note>dangling open tag\n")
    result = validate_page(PAGE, original, new, ManifestPage(path=PAGE), _adapter())
    assert not result.passed
    assert any("Note" in f for f in result.failures)


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
# repair_structure — safe auto-repair before the hard gate
# ---------------------------------------------------------------------------


def test_repair_closes_trailing_unclosed_components():
    a = _adapter()
    broken = "# T\n\n<Steps>\n<Step>do it</Step>\n<Step>more\n"
    fixed = a.repair_structure(broken)
    assert a.structural_problems(fixed) == []  # now well-formed
    assert fixed.rstrip().endswith("</Step>\n</Steps>")  # innermost closed first


def test_repair_drops_stray_closer():
    a = _adapter()
    broken = "# T\n\nbody text here</Note>\n\nmore body\n"
    fixed = a.repair_structure(broken)
    assert a.structural_problems(fixed) == []
    assert "</Note>" not in fixed


def test_repair_is_noop_on_well_formed_text():
    a = _adapter()
    good = "# T\n\n<Note>all good</Note>\n"
    assert a.repair_structure(good) == good


def test_repair_bails_on_ambiguous_misnesting():
    a = _adapter()
    # A closer matching an *inner* open is ambiguous — leave it for the validator.
    misnested = "<Note><Tip>x</Note></Tip>"
    assert a.repair_structure(misnested) == misnested


def test_repair_ignores_tags_inside_code():
    a = _adapter()
    in_code = "# T\n\n```\n<Steps>\n```\n\nreal body\n"
    assert a.repair_structure(in_code) == in_code  # the <Steps> is example text


def test_repair_then_validate_recovers_a_dropped_page():
    # The exact failure mode bootstrap hit: an unclosed <Steps>/<Step> that the
    # validator rejects — repair makes it pass the hard gate.
    a = _adapter()
    body = "Guide content. " * 30
    broken = f"---\ntitle: T\ndescription: d\n---\n\n# T\n\n{body}\n\n<Steps>\n<Step>one\n"
    assert not validate_new_page(PAGE, broken, a).passed
    assert validate_new_page(PAGE, a.repair_structure(broken), a).passed


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
    # Whether or not a mintlify CLI is on PATH, an empty dir has no broken links,
    # so this must return [] without raising (a success line is not a finding).
    assert _adapter().check_links(tmp_path) == []


def test_parse_link_output_ignores_success_line():
    # Mintlify's "no broken links" success line contains "broken" but is NOT a
    # finding — only genuine problem lines should survive.
    output = (
        "checking for broken links...\n"
        "success no broken links found\n"
    )
    assert MintlifyAdapter._parse_link_output(output) == []


def test_parse_link_output_keeps_real_problems():
    output = "✓ /docs/intro\n/docs/missing — 404 not found\n"
    problems = MintlifyAdapter._parse_link_output(output)
    assert problems == ["/docs/missing — 404 not found"]
