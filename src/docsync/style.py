"""Shared documentation-craft guidance — one voice for every generation stage.

These are plain prompt fragments (no logic, no imports beyond the stdlib) consumed by
the author prompt (``bootstrap.py``), the edit prompt (``edits.py``), and the readability
polish prompt (``polish.py``). Keeping the "what makes docs good" rules here means they
live in exactly one place: edit one constant and every stage that writes prose moves in
step, instead of three prompts drifting apart.

The rules distil four well-worn documentation principles:

* **Inverted pyramid / BLUF** — lead with the point; a skimmer who reads only the first
  line of each section should still come away with the basics.
* **Scannability** — short paragraphs, descriptive headings, tables/bullets over walls of
  prose, callouts used sparingly.
* **Grounding** — every statement traceable to the source; never invent API surface.
* **Diátaxis discipline** — keep a page to its one job (its *kind*); don't fold a how-to
  into a reference or dump API tables into an explanation.
"""

from __future__ import annotations

INVERTED_PYRAMID = (
    "Lead with the point. Open the body with a 1-2 sentence summary stating what this "
    "page is and why a reader would reach for it, then a short orientation, then the "
    "detail. Apply the same shape to every section: its first sentence carries its single "
    "most important fact. A reader who skims only the first line of each section must "
    "still come away with the basics — never bury the key fact below setup or caveats."
)

SCANNABILITY = (
    "Write to be scanned, not read top-to-bottom. Keep paragraphs short (about four "
    "sentences or fewer); break a wall of prose into bullets, a table, or sub-headings. "
    "Use descriptive headings that say what the section answers. Prefer a table for any "
    "set of fields/parameters/options and a bullet list for any enumeration. Reserve a "
    "<Note>/<Warning> callout for the one caveat that genuinely matters — don't sprinkle "
    "them."
)

GROUNDING = (
    "Ground every statement in the provided source code — names, signatures, routes, env "
    "vars, defaults, and behavior must match the code exactly. Never invent an API, "
    "parameter, or behavior; when the source doesn't say, say less rather than guess."
)

DIATAXIS_DISCIPLINE = (
    "Keep the page to its one job (its kind). Don't fold a how-to procedure into a "
    "reference, or dump API tables into a conceptual explanation — mixing what a page is "
    "for is the most common cause of docs that are hard to use. Point to a sibling page "
    "for the adjacent need instead of absorbing it."
)

# Per-kind section skeleton. Keyed by the PageKind literal values ("reference" | "guide" |
# "concept") — kept as plain strings to avoid importing models (and a cycle).
KIND_STRUCTURE: dict[str, str] = {
    "reference": (
        "Structure (REFERENCE — consulted while working): a one-line summary of what this "
        "module/API does; then one section per public route/function/class/data-model, "
        "each with a fields/parameters TABLE (name · type · required · meaning) and the "
        "return or response shape; collect enums and env vars into their own tables. Be "
        "exhaustive and precise; skip motivation and narrative."
    ),
    "guide": (
        "Structure (GUIDE — get a task done): what this lets you do and when you'd want "
        "it; prerequisites; the procedure as numbered <Steps>/<Step>; how to verify it "
        "worked; and a short 'next steps' with links. Stay concrete and task-first — "
        "explain only what the reader needs to finish the task."
    ),
    "concept": (
        "Structure (CONCEPT — understand how and why): what this is and the problem it "
        "solves; how it works end to end (a mermaid diagram when it clarifies a flow); the "
        "key design decisions and trade-offs; and where it lives in the code so a reader "
        "can go deeper. Favor clear prose and a worked example over exhaustive API tables."
    ),
}


def kind_structure(kind: str) -> str:
    """The section skeleton for a page *kind*, defaulting to reference for unknowns."""
    return KIND_STRUCTURE.get(kind, KIND_STRUCTURE["reference"])


# ---------------------------------------------------------------------------
# Thoroughness — how much to write (the light/medium/high dial)
# ---------------------------------------------------------------------------
#
# Orthogonal to a page's *kind* (which says how to shape it): the same reference page
# can be covered lightly or exhaustively. Consumed by the author (bootstrap), edit, and
# polish prompts; the matching token budgets come from `tokens_for`.
_THOROUGHNESS: dict[str, str] = {
    "light": (
        "Thoroughness: LIGHT — cover only the most important surface. Keep the page "
        "short: a concise summary and the essential fields/steps, not an exhaustive "
        "enumeration. Omit secondary options and edge cases; document the common path. "
        "When updating, make the smallest change that restores accuracy."
    ),
    "medium": (
        "Thoroughness: MEDIUM — balanced depth. Cover the primary surface and the common "
        "options; note the edge cases that genuinely matter, but don't aim for exhaustive "
        "completeness. When updating, document the surface the change introduces without "
        "expanding into adjacent material."
    ),
    "high": (
        "Thoroughness: HIGH — be exhaustive within what the source supports. Document "
        "every public symbol, parameter, option, return shape, and edge case, with a "
        "worked example where it aids understanding. When updating, fully document the "
        "surface the change introduces (add the rows/sections/fields it warrants), still "
        "as surgical additions that match the page's existing structure."
    ),
}

# Output-token budget multipliers per level (applied to each stage's base budget).
_TOKEN_MULTIPLIER: dict[str, float] = {"light": 0.6, "medium": 1.0, "high": 1.6}


def thoroughness_directive(level: str) -> str:
    """The coverage/length directive for a thoroughness *level* (unknown ⇒ medium)."""
    return _THOROUGHNESS.get(level, _THOROUGHNESS["medium"])


def tokens_for(level: str, base: int) -> int:
    """Scale a stage's base output-token budget by the thoroughness *level*.

    light shrinks the budget, high grows it; medium is the unchanged base. Floored at a
    small minimum so a light page can still hold a real edit.
    """
    scaled = int(base * _TOKEN_MULTIPLIER.get(level, 1.0))
    return max(1024, scaled)
