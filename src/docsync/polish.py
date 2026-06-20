"""Opt-in readability polish — a fact-frozen second pass (polish.py).

After a page is authored (`bootstrap`) or edited (`run`), optionally run one more LLM
pass that revises it for READABILITY ONLY — lead-with-the-summary, scannable structure,
no wall of text — **without changing any facts**. The pass returns find/replace `EditOp`s
(reusing the `PageEdit` contract and the strict apply path in :mod:`docsync.edits`), so it
can only make surgical, reviewable changes, and the result is gated before it replaces the
original: frontmatter must be untouched, structure must stay well-formed, and the body
must not be gutted. Any failure falls back to the input text, so a bad polish never
regresses a valid page.

Gated entirely behind ``config.readability_pass`` / the ``--polish`` CLI flag — default
off, so standard runs add no extra spend. The craft rubric is sourced from
:mod:`docsync.style` so polish, authoring, and editing share one definition of "reads
well".
"""

from __future__ import annotations

from . import llm, style
from .adapters.base import DocAdapter
from .edits import EditApplicationError, apply_edits
from .models import DocsyncConfig, PageEdit
from .validate import validate_new_page

# Polish output is a handful of find/replace ops; a modest budget keeps it cheap.
_MAX_TOKENS = 4000
# Lighter than authoring — restructuring finished prose needs less reasoning than writing
# it. Kept separate from edit_effort ("high") so polish stays cheap.
_POLISH_EFFORT = "medium"
# Reject a polish that drops below this fraction of the original body length — a legitimate
# "condense" trims a little; gutting the page means it ate facts.
_MIN_BODY_RATIO = 0.5


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(kind: str) -> str:
    return "\n".join(
        [
            "You are a documentation editor improving the READABILITY of a finished MDX "
            "page. You return find/replace edit operations — you NEVER rewrite the whole "
            "file.",
            "Each `find` must be a VERBATIM, UNIQUE substring of the page (copy it exactly, "
            "including whitespace), as small as possible while staying unique.",
            "HARD CONSTRAINT: do not add, remove, or change any FACT — no new APIs, "
            "parameters, values, routes, or behavior. Only restructure, condense, clarify, "
            "or re-order content already on the page. You MAY add a short lead sentence, but "
            "only if it paraphrases information already present.",
            "NEVER change the YAML frontmatter. NEVER alter MDX component tag structure "
            "(<CardGroup>, <Card>, <Steps>, <Note>, etc.) or break code fences.",
            style.INVERTED_PYRAMID,
            style.SCANNABILITY,
            style.kind_structure(kind),
            "If the page already reads well, return an empty edits list with a "
            "no_change_reason.",
        ]
    )


def build_polish_prompt(page_path: str, page_text: str, kind: str) -> tuple[str, str]:
    """Return (system, user) for polishing one page. Split out so it's unit-testable."""
    system = _build_system_prompt(kind)
    user = (
        f"# Documentation page: {page_path}  (kind: {kind})\n\n"
        f"```mdx\n{page_text}\n```\n\n"
        "Propose find/replace edits that make this page lead with its key point and read "
        "more scannably, WITHOUT changing any facts. Return an empty edits list with a "
        "no_change_reason if it already reads well."
    )
    return system, user


# ---------------------------------------------------------------------------
# Polish call + safe application
# ---------------------------------------------------------------------------


def polish_page(page_path: str, page_text: str, kind: str, config: DocsyncConfig, *, client) -> PageEdit:
    """Ask the edit model for readability-only find/replace ops over `page_text`.

    Mirrors the `messages.parse(...)` idiom in edits.py (cached system block, structured
    `PageEdit` output) and meters under the `"polish"` stage. Injectable `client` for tests.
    """
    system, user = build_polish_prompt(page_path, page_text, kind)
    # Polish is fact-frozen (it can only restructure, never add), so it takes no
    # thoroughness *directive* — but a higher level means longer pages, so scale the
    # token budget by the per-kind level to leave room to restructure them.
    max_tokens = style.tokens_for(config.thoroughness_for(kind), _MAX_TOKENS)
    return llm.parse(
        client,
        stage="polish",
        model=config.models.edit_model,
        max_tokens=max_tokens,
        system=system,
        user=user,
        output_format=PageEdit,
        thinking=True,
        effort=_POLISH_EFFORT,
    )


def _frozen_frontmatter_ok(adapter: DocAdapter, original: str, polished: str) -> bool:
    """True iff the polished page's frontmatter metadata is unchanged."""
    return adapter.split_frontmatter(original)[0] == adapter.split_frontmatter(polished)[0]


def _body_not_gutted(adapter: DocAdapter, original: str, polished: str) -> bool:
    """True iff the polished body keeps at least `_MIN_BODY_RATIO` of the original length."""
    orig_body = adapter.split_frontmatter(original)[1]
    new_body = adapter.split_frontmatter(polished)[1]
    if not new_body.strip():
        return False
    return len(new_body) >= _MIN_BODY_RATIO * len(orig_body)


def polish_text(
    page_path: str,
    text: str,
    kind: str,
    config: DocsyncConfig,
    adapter: DocAdapter,
    *,
    client,
    check_links: bool = False,
    docs_root=None,
) -> tuple[str, bool, str]:
    """Run the readability pass and return ``(text, polished, note)``.

    Falls back to the *input* `text` (with ``polished=False``) on any failure — an LLM
    error, ops that don't apply, changed frontmatter, a gutted body, or a structural-gate
    failure — so a bad polish can never regress an already-valid page. `note` explains the
    outcome for the run report.
    """
    try:
        edit = polish_page(page_path, text, kind, config, client=client)
    except Exception as exc:  # noqa: BLE001 - polish must never block a page
        return text, False, f"polish skipped: {exc}"
    if not edit.edits:
        return text, False, edit.no_change_reason or "polish: no change"

    try:
        candidate = apply_edits(text, edit)
    except EditApplicationError as exc:
        return text, False, f"polish not applicable: {exc}"

    candidate = adapter.repair_structure(candidate)
    if not _frozen_frontmatter_ok(adapter, text, candidate):
        return text, False, "polish rejected: changed frontmatter"
    if not _body_not_gutted(adapter, text, candidate):
        return text, False, "polish rejected: body too short (possible content loss)"

    validation = validate_new_page(
        page_path, candidate, adapter, check_links=check_links, docs_root=docs_root
    )
    if not validation.passed:
        return text, False, "polish dropped by validation: " + "; ".join(validation.failures)

    return candidate, True, f"polished ({len(edit.edits)} edit(s))"
