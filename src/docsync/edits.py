"""Stage 4 — LLM edit generation (edits.py).

Given an impacted `.mdx` page and the code change that invalidated it, ask Claude
Opus 4.8 to produce a small list of *str-replace edit ops* — never a full rewrite —
then apply them with a strict uniqueness check.

The contract (find/replace `EditOp`s wrapped in a `PageEdit`) lives in models.py.
This module is the only place that talks to the Anthropic SDK for edit generation;
`generate_page_edit` is injectable with a fake `client` for tests.
"""

from __future__ import annotations

from . import cost
from .diffrender import render_diff
from .models import (
    CodeDiff,
    DocsyncConfig,
    ImpactedPage,
    ManifestPage,
    PageEdit,
)

# Max tokens for the parse call. Edits are small (a handful of find/replace ops),
# and 8000 stays below the streaming threshold so a non-streaming parse is fine.
_MAX_TOKENS = 8000

# The edit model (Opus) has a large window, and more diff context improves edit
# accuracy — render the diff with a generous cap, also so a big multi-file diff can
# clear the cacheable-prefix floor (see _CACHE_DIFF_MIN_CHARS).
_EDIT_DIFF_MAX_CHARS = 40_000

# Only cache the shared diff block when the rendered diff is at least this many chars
# (~4k tokens — the Opus/Haiku cacheable-prefix minimum). Below it, the API silently
# won't cache, and a cache *write* without reads is a net loss.
_CACHE_DIFF_MIN_CHARS = 16_000


def should_cache_diff(diff: CodeDiff, n_pages: int) -> bool:
    """Whether to cache the shared diff as a prompt block for this run.

    Worth it only when (a) more than one page will be edited (so there are reads to
    amortize the one cache write) and (b) the rendered diff clears the model's
    ~4096-token cacheable-prefix floor. The pipeline primes the cache on page 1 before
    fanning out the rest, so reads land within the 5-minute ephemeral window.
    """
    if n_pages <= 1:
        return False
    return len(render_diff(diff, max_chars=_EDIT_DIFF_MAX_CHARS)) >= _CACHE_DIFF_MIN_CHARS


class EditApplicationError(Exception):
    """Raised when an edit op cannot be applied safely (not found, or ambiguous)."""


# ---------------------------------------------------------------------------
# Applying edits
# ---------------------------------------------------------------------------


def apply_edits(text: str, edit: PageEdit) -> str:
    """Apply each `EditOp` as an EXACT, single-occurrence string replacement.

    For every op, `find` must occur in the current working text exactly once. If it
    occurs zero times or more than once, raise `EditApplicationError` — never
    fuzzy-match, never replace-all. Ops are applied sequentially against the evolving
    text, so a later op sees the result of the earlier ones. An empty edit list
    returns `text` unchanged.
    """
    working = text
    for index, op in enumerate(edit.edits):
        count = working.count(op.find)
        if count == 0:
            raise EditApplicationError(
                f"edit op #{index} not applicable: `find` not found in the current "
                f"text (rationale: {op.rationale!r})"
            )
        if count > 1:
            raise EditApplicationError(
                f"edit op #{index} ambiguous: `find` occurs {count} times in the "
                f"current text; it must be unique (rationale: {op.rationale!r})"
            )
        working = working.replace(op.find, op.replace, 1)
    return working


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_system_prompt(allow_frontmatter_edit: bool) -> str:
    lines = [
        "You update an existing MDX documentation page to reflect a code change. "
        "You return a list of find/replace edit operations — you NEVER rewrite the "
        "whole file.",
        "Each `find` must be a VERBATIM, UNIQUE substring of the current page (copy "
        "it exactly, including whitespace). Make `find` long enough to be unique but "
        "as small as possible — ideally a single table row, sentence, or code-fence "
        "line.",
        "Edit ONLY the specific rows, prose, or code-fence lines the diff "
        "invalidates. Do not touch unrelated content.",
    ]
    if allow_frontmatter_edit:
        lines.append(
            "You MAY edit the YAML frontmatter `title` or `description` for this page "
            "if the code change requires it, but only when genuinely necessary."
        )
    else:
        lines.append(
            "NEVER change the YAML frontmatter `title` or `description`."
        )
    lines.extend(
        [
            "NEVER alter MDX component tag structure (<CardGroup>, <Card>, <Warning>, "
            "<Note>, etc.) or break mermaid code fences.",
            "Preserve inline backtick code references unless the underlying symbol was "
            "renamed in the diff.",
            "If the page is NOT actually invalidated by this diff, return an empty "
            "edits list and set no_change_reason explaining why.",
        ]
    )
    return "\n".join(lines)


def build_edit_prompt(
    page_path: str,
    page_text: str,
    diff: CodeDiff,
    impacted: ImpactedPage,
    manifest_page: ManifestPage | None,
    *,
    rendered_diff: str | None = None,
    include_diff: bool = True,
) -> tuple[str, str]:
    """Return `(system_prompt, user_prompt)`.

    Split out from `generate_page_edit` so tests can assert prompt content without
    making an API call. `rendered_diff` reuses a pre-rendered diff (so it isn't
    rendered twice); `include_diff=False` omits the diff from the user message — used
    when it's supplied in a cached system block shared across the run's pages.
    """
    allow_frontmatter_edit = bool(
        manifest_page is not None and manifest_page.allow_frontmatter_edit
    )
    system_prompt = _build_system_prompt(allow_frontmatter_edit)
    rendered = (
        rendered_diff if rendered_diff is not None
        else render_diff(diff, max_chars=_EDIT_DIFF_MAX_CHARS)
    )

    if include_diff:
        code_change = f"# Code change\n\n{rendered}\n\n"
    else:
        code_change = (
            "# Code change\n\nThe code change for this run is provided in the system "
            "context above; edit this page to reflect it.\n\n"
        )

    user_prompt = (
        f"# Documentation page: {page_path}\n\n"
        f"This page was flagged as impacted ({impacted.source.value}, "
        f"confidence {impacted.confidence:.2f}).\n"
        f"Reason: {impacted.reason}\n\n"
        f"# Current page content\n\n"
        f"```mdx\n{page_text}\n```\n\n"
        f"{code_change}"
        "Produce surgical find/replace edits to bring the page in line with this "
        "change, or an empty edits list with a no_change_reason if the page is not "
        "actually invalidated."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Edit generation
# ---------------------------------------------------------------------------


def generate_page_edit(
    page_path: str,
    page_text: str,
    diff: CodeDiff,
    impacted: ImpactedPage,
    manifest_page: ManifestPage | None,
    config: DocsyncConfig,
    cache_diff: bool = False,
    client=None,
) -> PageEdit:
    """Ask Claude Opus 4.8 for surgical edits to align `page_text` with `diff`.

    Returns a `PageEdit` validated via the SDK's structured-output helper
    (`messages.parse`). `client` is injectable for tests; it defaults to
    `anthropic.Anthropic()`.

    Opus 4.8 uses adaptive thinking only — depth is controlled via
    `output_config.effort` (no `budget_tokens`, no sampling params). The invariant
    system prompt is cached (`cache_control: ephemeral`) since it's identical across
    every page in a run.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    rendered = render_diff(diff, max_chars=_EDIT_DIFF_MAX_CHARS)
    system_prompt, user_prompt = build_edit_prompt(
        page_path, page_text, diff, impacted, manifest_page,
        rendered_diff=rendered, include_diff=not cache_diff,
    )

    system_blocks: list[dict] = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]
    if cache_diff:
        # The diff is identical for every page in a run; put it in its own cached
        # system block (byte-identical across pages) so pages 2..N read it instead of
        # re-sending it. The pipeline primes this on page 1 before fanning out.
        system_blocks.append(
            {
                "type": "text",
                "text": f"# Code change (shared across all pages in this run)\n\n{rendered}",
                "cache_control": {"type": "ephemeral"},
            }
        )

    with cost.stage("edit"):
        resp = client.messages.parse(
            model=config.models.edit_model,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": config.models.edit_effort},
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=PageEdit,
        )
    return resp.parsed_output
