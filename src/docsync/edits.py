"""Stage 4 — LLM edit generation (edits.py).

Given an impacted `.mdx` page and the code change that invalidated it, ask Claude
Opus 4.8 to produce a small list of *str-replace edit ops* — never a full rewrite —
then apply them with a strict uniqueness check.

The contract (find/replace `EditOp`s wrapped in a `PageEdit`) lives in models.py.
This module is the only place that talks to the Anthropic SDK for edit generation;
`generate_page_edit` is injectable with a fake `client` for tests.
"""

from __future__ import annotations

from .models import (
    CodeDiff,
    DocsyncConfig,
    ImpactedPage,
    ManifestPage,
    PageEdit,
)

# Keep the rendered diff bounded so a giant PR can't blow up the prompt. The
# whole per-file diff section is capped at this many characters.
_MAX_DIFF_CHARS = 12_000

# Max tokens for the parse call. Edits are small (a handful of find/replace ops),
# and 8000 stays below the streaming threshold so a non-streaming parse is fine.
_MAX_TOKENS = 8000


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


def _render_diff(diff: CodeDiff) -> str:
    """A compact rendering of the diff, capped at ~_MAX_DIFF_CHARS characters."""
    header = f"repo: {diff.repo}\npr_title: {diff.pr_title or '(none)'}\n"
    parts: list[str] = []
    used = len(header)
    for f in diff.files:
        symbols = ", ".join(f.changed_symbols) or "(none)"
        section_lines = [
            f"\n## file: {f.path} ({f.status.value})",
            f"changed symbols: {symbols}",
        ]
        for hunk in f.hunks:
            section_lines.append(hunk)
        section = "\n".join(section_lines)
        if used + len(section) > _MAX_DIFF_CHARS:
            remaining = _MAX_DIFF_CHARS - used
            if remaining > 0:
                parts.append(section[:remaining])
            parts.append("\n... (diff truncated)")
            break
        parts.append(section)
        used += len(section)
    return header + "".join(parts)


def build_edit_prompt(
    page_path: str,
    page_text: str,
    diff: CodeDiff,
    impacted: ImpactedPage,
    manifest_page: ManifestPage | None,
) -> tuple[str, str]:
    """Return `(system_prompt, user_prompt)`.

    Split out from `generate_page_edit` so tests can assert prompt content without
    making an API call.
    """
    allow_frontmatter_edit = bool(
        manifest_page is not None and manifest_page.allow_frontmatter_edit
    )
    system_prompt = _build_system_prompt(allow_frontmatter_edit)

    user_prompt = (
        f"# Documentation page: {page_path}\n\n"
        f"This page was flagged as impacted ({impacted.source.value}, "
        f"confidence {impacted.confidence:.2f}).\n"
        f"Reason: {impacted.reason}\n\n"
        f"# Current page content\n\n"
        f"```mdx\n{page_text}\n```\n\n"
        f"# Code change\n\n"
        f"{_render_diff(diff)}\n\n"
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

    system_prompt, user_prompt = build_edit_prompt(
        page_path, page_text, diff, impacted, manifest_page
    )

    resp = client.messages.parse(
        model=config.models.edit_model,
        max_tokens=_MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": config.models.edit_effort},
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
        output_format=PageEdit,
    )
    return resp.parsed_output
