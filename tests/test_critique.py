"""Unit tests for the adversarial self-critique gate (`docsync.critique`).

No real API calls — `critique_page_edit` is exercised with a scripted fake
client whose `.messages.parse(...)` returns an object exposing `.parsed_output`
(the same pattern as test_pipeline / test_edits).
"""

from __future__ import annotations

from docsync.critique import (
    CritiqueVerdict,
    apply_critique,
    build_critique_prompt,
    critique_page_edit,
)
from docsync.models import (
    ChangedFile,
    CodeDiff,
    EditOp,
    FileStatus,
    PageEdit,
)

REPO = "keephq/keep-api-gateway"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_diff(
    *,
    path: str = "src/routes/alerts.py",
    symbols: list[str] | None = None,
    hunks: list[str] | None = None,
) -> CodeDiff:
    return CodeDiff(
        repo=REPO,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="Rename foo to bar",
        files=[
            ChangedFile(
                path=path,
                status=FileStatus.MODIFIED,
                changed_symbols=list(symbols or ["get_alerts"]),
                hunks=list(hunks or ["@@ -1,2 +1,2 @@\n-old\n+new"]),
            )
        ],
    )


def make_page_edit() -> PageEdit:
    return PageEdit(
        edits=[
            EditOp(find="GET /alerts", replace="POST /alerts", rationale="method changed"),
            EditOp(find="hallucinated bit", replace="x", rationale="not in diff"),
        ]
    )


# ---------------------------------------------------------------------------
# Fake client (scripted, no network)
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, verdict: CritiqueVerdict):
        self._verdict = verdict
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)

        class _Resp:
            parsed_output = self._verdict

        return _Resp()


class FakeClient:
    """Stub whose `.messages.parse(...)` returns an object with `.parsed_output`."""

    def __init__(self, verdict: CritiqueVerdict):
        self.messages = _FakeMessages(verdict)


# ---------------------------------------------------------------------------
# build_critique_prompt
# ---------------------------------------------------------------------------


def test_build_critique_prompt_includes_symbols_path_and_ops():
    diff = make_diff(symbols=["get_alerts", "setup_routers"])
    page_edit = make_page_edit()
    prompt = build_critique_prompt(diff, "api/alerts.mdx", page_edit)

    # changed symbols rendered
    assert "get_alerts" in prompt
    assert "setup_routers" in prompt
    # page path rendered
    assert "api/alerts.mdx" in prompt
    # changed path rendered
    assert "src/routes/alerts.py" in prompt
    # each op's find/replace text rendered
    assert "GET /alerts" in prompt
    assert "POST /alerts" in prompt
    assert "hallucinated bit" in prompt
    # rationale rendered
    assert "method changed" in prompt


# ---------------------------------------------------------------------------
# critique_page_edit (fake client)
# ---------------------------------------------------------------------------


def test_critique_page_edit_returns_scripted_verdict_and_uses_judge_model():
    expected = CritiqueVerdict(
        faithful=False,
        rejected_finds=["hallucinated bit"],
        reason="op #1 references content absent from the diff",
    )
    client = FakeClient(expected)

    result = critique_page_edit(
        client,
        diff=make_diff(),
        page_path="api/alerts.mdx",
        page_edit=make_page_edit(),
    )

    assert result is expected
    assert len(client.messages.calls) == 1
    call = client.messages.calls[0]
    # Default judge model (Haiku).
    assert call["model"] == "claude-haiku-4-5"
    assert call["output_format"] is CritiqueVerdict
    # System prefix cached (invariant across pages in a run).
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_critique_page_edit_model_override():
    client = FakeClient(CritiqueVerdict(faithful=True))
    critique_page_edit(
        client,
        diff=make_diff(),
        page_path="api/alerts.mdx",
        page_edit=make_page_edit(),
        model="claude-custom-judge",
    )
    assert client.messages.calls[0]["model"] == "claude-custom-judge"


# ---------------------------------------------------------------------------
# apply_critique
# ---------------------------------------------------------------------------


def test_apply_critique_drops_only_rejected_finds_preserving_order():
    page_edit = PageEdit(
        edits=[
            EditOp(find="keep-1", replace="a", rationale="1"),
            EditOp(find="drop-me", replace="b", rationale="2"),
            EditOp(find="keep-2", replace="c", rationale="3"),
        ]
    )
    verdict = CritiqueVerdict(faithful=False, rejected_finds=["drop-me"], reason="r")

    result = apply_critique(page_edit, verdict)

    assert [op.find for op in result.edits] == ["keep-1", "keep-2"]
    # input not mutated
    assert len(page_edit.edits) == 3


def test_apply_critique_dropping_all_yields_empty_edits():
    page_edit = make_page_edit()
    verdict = CritiqueVerdict(
        faithful=False,
        rejected_finds=["GET /alerts", "hallucinated bit"],
        reason="all hallucinated",
    )
    result = apply_critique(page_edit, verdict)
    assert result.edits == []


def test_apply_critique_faithful_verdict_leaves_page_edit_unchanged():
    page_edit = make_page_edit()
    verdict = CritiqueVerdict(faithful=True, rejected_finds=[], reason="all justified")

    result = apply_critique(page_edit, verdict)

    assert [op.find for op in result.edits] == [op.find for op in page_edit.edits]
    assert [op.replace for op in result.edits] == [op.replace for op in page_edit.edits]
