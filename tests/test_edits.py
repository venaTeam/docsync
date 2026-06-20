"""Unit tests for Stage 4 LLM edit generation (`docsync.edits`).

No real API calls — `generate_page_edit` is exercised with a fake client whose
`.messages.parse(...)` returns an object exposing `.parsed_output`.
"""

from __future__ import annotations

import pytest

from docsync.edits import (
    EditApplicationError,
    apply_edits,
    build_edit_prompt,
    generate_page_edit,
    should_cache_diff,
)
from docsync.models import (
    CandidateSource,
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    EditOp,
    FileStatus,
    ImpactedPage,
    ManifestPage,
    PageEdit,
)

REPO = "keephq/keep-api-gateway"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_diff(
    *,
    repo: str = REPO,
    pr_title: str | None = "Rename foo to bar",
    path: str = "src/routes/alerts.py",
    symbols: list[str] | None = None,
    hunks: list[str] | None = None,
) -> CodeDiff:
    return CodeDiff(
        repo=repo,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title=pr_title,
        files=[
            ChangedFile(
                path=path,
                status=FileStatus.MODIFIED,
                changed_symbols=list(symbols or ["get_alerts"]),
                hunks=list(hunks or ["@@ -1,2 +1,2 @@\n-old\n+new"]),
            )
        ],
    )


def make_impacted(
    page_path: str = "api/alerts.mdx",
    *,
    reason: str = "route signature changed",
) -> ImpactedPage:
    return ImpactedPage(
        page_path=page_path,
        source=CandidateSource.ANCHOR,
        confidence=0.9,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# apply_edits
# ---------------------------------------------------------------------------


def test_apply_edits_single_replace():
    text = "The endpoint is GET /alerts and returns alerts."
    edit = PageEdit(
        edits=[EditOp(find="GET /alerts", replace="POST /alerts", rationale="method changed")]
    )
    assert apply_edits(text, edit) == "The endpoint is POST /alerts and returns alerts."


def test_apply_edits_multiple_sequential_ops():
    text = "alpha then beta then gamma"
    edit = PageEdit(
        edits=[
            EditOp(find="alpha", replace="ALPHA", rationale="1"),
            EditOp(find="gamma", replace="GAMMA", rationale="2"),
        ]
    )
    assert apply_edits(text, edit) == "ALPHA then beta then GAMMA"


def test_apply_edits_sequential_op_sees_prior_result():
    # The second op only matches because the first one created its target.
    text = "value = 1"
    edit = PageEdit(
        edits=[
            EditOp(find="value = 1", replace="value = 2", rationale="bump"),
            EditOp(find="value = 2", replace="value = 3", rationale="bump again"),
        ]
    )
    assert apply_edits(text, edit) == "value = 3"


def test_apply_edits_empty_returns_text_unchanged():
    text = "unchanged content"
    assert apply_edits(text, PageEdit(edits=[])) == text
    # A default PageEdit also has no edits.
    assert apply_edits(text, PageEdit()) == text


def test_apply_edits_zero_match_raises():
    text = "nothing to see here"
    edit = PageEdit(edits=[EditOp(find="ABSENT", replace="x", rationale="r")])
    with pytest.raises(EditApplicationError):
        apply_edits(text, edit)


def test_apply_edits_multi_match_is_ambiguous_and_raises():
    text = "foo and foo again"
    edit = PageEdit(edits=[EditOp(find="foo", replace="bar", rationale="r")])
    with pytest.raises(EditApplicationError):
        apply_edits(text, edit)


def test_apply_edits_classifies_not_found_as_non_ambiguous():
    text = "nothing to see here"
    edit = PageEdit(edits=[EditOp(find="ABSENT", replace="x", rationale="r")])
    with pytest.raises(EditApplicationError) as exc:
        apply_edits(text, edit)
    assert exc.value.ambiguous is False


def test_apply_edits_classifies_multi_match_as_ambiguous():
    text = "foo and foo again"
    edit = PageEdit(edits=[EditOp(find="foo", replace="bar", rationale="r")])
    with pytest.raises(EditApplicationError) as exc:
        apply_edits(text, edit)
    assert exc.value.ambiguous is True


def test_apply_edits_never_replace_all_partial_failure_message():
    text = "a duplicated line\na duplicated line\n"
    edit = PageEdit(
        edits=[EditOp(find="a duplicated line", replace="x", rationale="dedupe")]
    )
    with pytest.raises(EditApplicationError) as exc:
        apply_edits(text, edit)
    assert "ambiguous" in str(exc.value)


def test_apply_edits_later_op_becomes_ambiguous_aborts():
    # First op makes the text contain two copies of "X"; second op targeting "X" fails.
    text = "X here and Y there"
    edit = PageEdit(
        edits=[
            EditOp(find="Y", replace="X", rationale="introduce duplicate"),
            EditOp(find="X", replace="Z", rationale="now ambiguous"),
        ]
    )
    with pytest.raises(EditApplicationError):
        apply_edits(text, edit)


# ---------------------------------------------------------------------------
# build_edit_prompt
# ---------------------------------------------------------------------------


def test_build_edit_prompt_system_forbids_frontmatter_and_components():
    diff = make_diff(symbols=["get_alerts"])
    impacted = make_impacted()
    system_prompt, _ = build_edit_prompt(
        "api/alerts.mdx", "# page\n", diff, impacted, manifest_page=None
    )
    assert "NEVER change the YAML frontmatter" in system_prompt
    # The component invariant is stated as the validated rule: add balanced leaf
    # components, but never remove one or alter a container.
    assert "NEVER remove a component" in system_prompt
    assert "alter a container component" in system_prompt
    assert "NEVER rewrite the whole file" in system_prompt


def test_build_edit_prompt_user_contains_page_text_and_symbol():
    diff = make_diff(symbols=["get_alerts"])
    impacted = make_impacted(reason="the GET /alerts handler was renamed")
    page_text = "## Alerts\nThe `get_alerts` endpoint lists alerts."
    _, user_prompt = build_edit_prompt(
        "api/alerts.mdx", page_text, diff, impacted, manifest_page=None
    )
    assert page_text in user_prompt
    assert "get_alerts" in user_prompt  # changed symbol rendered
    assert "the GET /alerts handler was renamed" in user_prompt
    assert "Rename foo to bar" in user_prompt  # pr_title rendered


def test_build_edit_prompt_allows_frontmatter_when_manifest_permits():
    diff = make_diff()
    impacted = make_impacted()
    mp = ManifestPage(path="api/alerts.mdx", allow_frontmatter_edit=True)
    system_prompt, _ = build_edit_prompt(
        "api/alerts.mdx", "# page\n", diff, impacted, manifest_page=mp
    )
    assert "MAY edit the YAML frontmatter" in system_prompt
    assert "NEVER change the YAML frontmatter" not in system_prompt


def test_build_edit_prompt_diff_rendering_is_capped():
    big_hunk = "X" * 50_000
    diff = make_diff(hunks=[big_hunk])
    impacted = make_impacted()
    _, user_prompt = build_edit_prompt(
        "api/alerts.mdx", "page", diff, impacted, manifest_page=None
    )
    assert "diff truncated" in user_prompt
    # The full 50k hunk must not have been included verbatim.
    assert big_hunk not in user_prompt


# ---------------------------------------------------------------------------
# generate_page_edit (fake client)
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, page_edit: PageEdit):
        self._page_edit = page_edit
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        class _Resp:
            parsed_output = self._page_edit

        return _Resp()


class FakeClient:
    """Stub whose `.messages.parse(...)` returns an object with `.parsed_output`."""

    def __init__(self, page_edit: PageEdit):
        self.messages = _FakeMessages(page_edit)


def test_generate_page_edit_returns_parsed_output_and_calls_model():
    expected = PageEdit(
        edits=[EditOp(find="old", replace="new", rationale="renamed in diff")]
    )
    client = FakeClient(expected)
    config = DocsyncConfig()
    diff = make_diff()
    impacted = make_impacted()

    result = generate_page_edit(
        "api/alerts.mdx",
        "page with old text",
        diff,
        impacted,
        manifest_page=None,
        config=config,
        client=client,
    )

    assert result is expected
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    # Opus 4.8 by default, adaptive thinking, effort + structured output.
    assert call["model"] == "claude-opus-4-8"
    assert call["model"] == config.models.edit_model
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": config.models.edit_effort}
    assert call["output_format"] is PageEdit
    # System prompt cached (invariant across pages in a run).
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Diff prompt-caching (gated, edit-stage only)
# ---------------------------------------------------------------------------


def test_should_cache_diff_gates_on_pages_and_size():
    small = make_diff(hunks=["@@ -1 +1 @@\n-a\n+b"])
    big = make_diff(hunks=["X" * 20_000])
    assert should_cache_diff(small, n_pages=3) is False  # diff too small to cache
    assert should_cache_diff(big, n_pages=1) is False    # single page: no reads
    assert should_cache_diff(big, n_pages=2) is True     # big diff + multiple pages


def test_generate_page_edit_default_keeps_single_uncached_diff():
    client = FakeClient(PageEdit(edits=[]))
    generate_page_edit(
        "p.mdx", "page text", make_diff(symbols=["get_alerts"]), make_impacted(),
        manifest_page=None, config=DocsyncConfig(), client=client,  # cache_diff defaults False
    )
    call = client.messages.calls[0]
    assert len(call["system"]) == 1  # instructions only
    assert "get_alerts" in call["messages"][0]["content"]  # diff inline in the user msg


def test_generate_page_edit_caches_diff_in_system_block():
    client = FakeClient(PageEdit(edits=[]))
    big = make_diff(symbols=["get_alerts"], hunks=["X" * 20_000])
    generate_page_edit(
        "p.mdx", "page text", big, make_impacted(),
        manifest_page=None, config=DocsyncConfig(), cache_diff=True, client=client,
    )
    call = client.messages.calls[0]
    system = call["system"]
    assert len(system) == 2
    assert system[1]["cache_control"] == {"type": "ephemeral"}
    assert "Code change" in system[1]["text"]
    # The diff is in the cached block, NOT re-sent in the per-page user message.
    user = call["messages"][0]["content"]
    assert "system context" in user
    assert "get_alerts" not in user


def test_build_edit_prompt_can_omit_diff_for_caching():
    diff = make_diff(symbols=["get_alerts"])
    _, user = build_edit_prompt(
        "p.mdx", "page", diff, make_impacted(), manifest_page=None, include_diff=False
    )
    assert "get_alerts" not in user
    assert "system context" in user


def test_generate_page_edit_passes_no_change_through():
    expected = PageEdit(edits=[], no_change_reason="page does not document this symbol")
    client = FakeClient(expected)
    result = generate_page_edit(
        "api/alerts.mdx",
        "unrelated page",
        make_diff(),
        make_impacted(),
        manifest_page=None,
        config=DocsyncConfig(),
        client=client,
    )
    assert result is expected
    assert result.edits == []
    assert result.no_change_reason == "page does not document this symbol"
