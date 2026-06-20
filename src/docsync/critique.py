"""Stage 4.5 — adversarial self-critique gate for generated edits (critique.py).

After the Opus editor proposes a `PageEdit` for a page, run a cheap second LLM
call (Haiku) that checks each edit op against the actual code diff and asks:
"does this edit faithfully reflect THIS diff, and touch only what the diff
changed?" — flagging hallucinated or over-reaching ops so they can be dropped
before validation. This cuts false positives at low cost.

The contract is intentionally flat (`CritiqueVerdict`) so the structured-output
backend has nothing nested to validate. `critique_page_edit` mirrors the
`messages.parse(...)` idiom used in edits.py and is injectable with a fake
`client` for tests; `apply_critique` is a pure helper that drops the flagged ops.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm
from .diffrender import render_diff
from .models import CodeDiff, EditOp, PageEdit

# Default judge model — the cheap second-opinion model (Haiku).
_JUDGE_MODEL = "claude-haiku-4-5"

# Max tokens for the parse call. The verdict is tiny (a bool, a short list of
# find strings, and a reason), so a small budget keeps this call cheap and
# non-streaming.
_MAX_TOKENS = 2000


# ---------------------------------------------------------------------------
# Verdict model
# ---------------------------------------------------------------------------


class CritiqueVerdict(BaseModel):
    """The judge's flat verdict over a proposed `PageEdit`.

    `faithful` is True iff every KEPT op is justified by the diff (i.e. there is
    nothing left to reject). `rejected_finds` holds the exact `find` strings of
    the ops the judge wants dropped, and `reason` is a short human-readable
    explanation of the call.
    """

    faithful: bool
    rejected_finds: list[str] = Field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _render_ops(page_edit: PageEdit) -> str:
    """Render each proposed edit op (find/replace/rationale) for the prompt."""
    if not page_edit.edits:
        return "(no edit ops proposed)"
    blocks: list[str] = []
    for index, op in enumerate(page_edit.edits):
        blocks.append(
            f"### op #{index}\n"
            f"find:\n```\n{op.find}\n```\n"
            f"replace:\n```\n{op.replace}\n```\n"
            f"rationale: {op.rationale}"
        )
    return "\n\n".join(blocks)


def build_critique_prompt(
    diff: CodeDiff,
    page_path: str,
    page_edit: PageEdit,
) -> str:
    """Build the user message for the critique call.

    Split out from `critique_page_edit` so prompt content is unit-testable
    without a client. Contains the diff's changed paths + symbols + hunks and
    every proposed edit op (find/replace/rationale).
    """
    changed_paths = ", ".join(diff.changed_paths()) or "(none)"
    changed_symbols = ", ".join(diff.all_symbols()) or "(none)"
    return (
        f"# Documentation page: {page_path}\n\n"
        f"# Code change\n\n"
        f"changed paths: {changed_paths}\n"
        f"changed symbols: {changed_symbols}\n\n"
        f"{render_diff(diff)}\n\n"
        f"# Proposed edit ops\n\n"
        f"{_render_ops(page_edit)}\n\n"
        "For each op, decide whether it is DIRECTLY justified by the diff above. "
        "List the exact `find` string of every op that is NOT justified in "
        "`rejected_finds`. Set `faithful` to true only if `rejected_finds` is "
        "empty, and give a short `reason`."
    )


def _build_system_prompt() -> str:
    """The invariant system prefix stating the critique invariants."""
    return "\n".join(
        [
            "You are an adversarial reviewer of proposed documentation edits. For "
            "each edit op you are given the code diff that supposedly motivated it.",
            "Your ONLY job is to judge FAITHFULNESS to the diff — not writing "
            "quality, not style, not whether the docs could be better.",
            "KEEP an op if it reflects something the diff actually changed. An op "
            "that merely ADDS undocumented-but-correct information about a real diff "
            "change is acceptable and must be kept.",
            "REJECT an op only if it is about something the diff did NOT change — a "
            "hallucinated symbol, an unrelated section, or an over-reaching rewrite "
            "of content the diff never touched.",
            "Put the exact `find` string of each rejected op in `rejected_finds`. "
            "Set `faithful` true iff `rejected_finds` is empty. Provide a concise "
            "`reason`.",
        ]
    )


# ---------------------------------------------------------------------------
# Critique call
# ---------------------------------------------------------------------------


def critique_page_edit(
    client,
    *,
    diff: CodeDiff,
    page_path: str,
    page_edit: PageEdit,
    model: str | None = None,
    system_extra: str = "",
) -> CritiqueVerdict:
    """Ask the judge model whether each edit op is faithful to `diff`.

    Mirrors the `messages.parse(...)` idiom from edits.py: a cached invariant
    system prefix plus a per-page user message, returning the SDK's
    `.parsed_output` (a `CritiqueVerdict`). Defaults to the Haiku judge model;
    `model` overrides it. `system_extra`, if given, is appended to the cached
    system prefix.
    """
    system_text = _build_system_prompt()
    if system_extra:
        system_text = f"{system_text}\n{system_extra}"

    user_prompt = build_critique_prompt(diff, page_path, page_edit)

    return llm.parse(
        client,
        stage="critique",
        model=model or _JUDGE_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system_text,
        user=user_prompt,
        output_format=CritiqueVerdict,
    )


# ---------------------------------------------------------------------------
# Applying the verdict
# ---------------------------------------------------------------------------


def apply_critique(page_edit: PageEdit, verdict: CritiqueVerdict) -> PageEdit:
    """Return a new `PageEdit` with the verdict's `rejected_finds` ops removed.

    Ops are matched on their exact `find` string and kept in their original
    order. If every op is rejected, the returned `PageEdit` has an empty edits
    list (the pipeline then treats the page as no-change). The input is not
    mutated.
    """
    rejected = set(verdict.rejected_finds)
    kept: list[EditOp] = [op for op in page_edit.edits if op.find not in rejected]
    return PageEdit(edits=kept, no_change_reason=page_edit.no_change_reason)
