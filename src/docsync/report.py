"""Render a human-readable report + PR body from a PipelineResult."""

from __future__ import annotations

import difflib

from .cost import render_usage_console, render_usage_md
from .models import BootstrapResult, InferResult, PipelineResult


def _unified(page_path: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{page_path}",
        tofile=f"b/{page_path}",
    )
    return "".join(diff)


def pr_body(result: PipelineResult, *, original_texts: dict[str, str] | None = None) -> str:
    """Markdown PR body: per-page rationale + dropped-edit notes.

    `original_texts` maps page_path -> original content; when provided, a
    collapsed diff preview is embedded per changed page.
    """
    d = result.diff
    src = f"`{d.repo}`"
    pr_ref = f" (PR #{d.pr_number})" if d.pr_number else ""
    lines: list[str] = []
    lines.append(f"## docsync — documentation updates for {src}{pr_ref}")
    lines.append("")
    lines.append(
        f"Triggered by `{d.base_sha[:8]}..{d.head_sha[:8]}`"
        + (f" — _{d.pr_title}_" if d.pr_title else "")
    )
    lines.append("")

    changed = result.changed()
    if changed:
        lines.append(f"### Updated {len(changed)} page(s)")
        for o in changed:
            n_edits = len(o.edit.edits) if o.edit else 0
            lines.append(f"- **`{o.page_path}`** — {n_edits} edit(s)")
            for op in o.edit.edits if o.edit else []:
                lines.append(f"  - {op.rationale}")
            warns = o.validation.warnings if o.validation else []
            for w in warns:
                lines.append(f"  - ⚠️ {w}")
        lines.append("")
        if original_texts:
            lines.append("<details><summary>Diff preview</summary>\n")
            for o in changed:
                before = original_texts.get(o.page_path)
                if before is not None and o.new_content is not None:
                    lines.append(f"\n```diff\n{_unified(o.page_path, before, o.new_content)}```")
            lines.append("\n</details>")
            lines.append("")

    dropped = [o for o in result.outcomes if not o.applied]
    if dropped:
        lines.append("### Considered but not changed")
        for o in dropped:
            lines.append(f"- `{o.page_path}` — {o.note}")
        lines.append("")

    usage_lines = render_usage_md(result.usage)
    if usage_lines:
        lines.extend(usage_lines)
        lines.append("")

    lines.append("---")
    lines.append("_Opened by [docsync](https://github.com/keephq/docsync). Review the edits "
                 "before merging — docsync edits existing pages only; new pages / nav changes "
                 "are flagged here but not auto-applied._")
    return "\n".join(lines)


def console_summary(result: PipelineResult) -> str:
    """Short terminal summary for the CLI."""
    changed = result.changed()
    out = [f"docsync: {len(result.outcomes)} page(s) impacted, {len(changed)} updated."]
    for o in result.outcomes:
        mark = "✓" if o.applied else "·"
        out.append(f"  {mark} {o.page_path} — {o.note}")
    cost_line = render_usage_console(result.usage)
    if cost_line:
        out.append(cost_line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Bootstrap (from-scratch generation)
# ---------------------------------------------------------------------------


def bootstrap_console_summary(result: BootstrapResult) -> str:
    """Short terminal summary for `docsync bootstrap`, grouped by IA section.

    Lists pages under their section (in reading-flow order) with status: `✓` authored,
    `·` dropped, or `•` planned-but-not-yet-authored (the plan-only case).
    """
    authored = result.authored()
    out = [
        f"docsync bootstrap: {len(result.plan.pages)} page(s) planned, "
        f"{len(authored)} authored, {len(result.skipped)} skipped (collisions)."
    ]
    by_outcome = {o.page_path: o for o in result.outcomes}
    for section, pages in result.plan.ordered_sections():
        out.append(f"  [{section}]")
        for p in pages:
            o = by_outcome.get(p.page_path)
            if o is None:  # plan-only: planned but not authored
                out.append(f"    • {p.page_path} — {p.title} ({p.kind})")
            else:
                mark = "✓" if o.applied else "·"
                out.append(f"    {mark} {p.page_path} — {o.note}")
    for path in result.skipped:
        out.append(f"  ⤫ {path} — skipped: already exists")
    cost_line = render_usage_console(result.usage)
    if cost_line:
        out.append(cost_line)
    return "\n".join(out)


def bootstrap_pr_body(result: BootstrapResult) -> str:
    """Markdown PR body for a bootstrap run: the authored site, by section."""
    authored = result.authored()
    authored_paths = {o.page_path for o in authored}
    lines: list[str] = []
    lines.append(f"## docsync — bootstrapped documentation site for `{result.repo}`")
    lines.append("")
    lines.append(
        f"Generated {len(authored)} new page(s) across a sequenced information "
        "architecture, with manifest anchors (narrative pages judge-gated) so "
        "`docsync run` keeps the whole site in sync with the code going forward."
    )
    lines.append("")
    if authored:
        for section, pages in result.plan.ordered_sections():
            sec = [p for p in pages if p.page_path in authored_paths]
            if not sec:
                continue
            lines.append(f"### {section}")
            for p in sec:
                lines.append(f"- **`{p.page_path}`** — {p.title} ({p.kind})")
            lines.append("")

    dropped = [o for o in result.outcomes if not o.applied]
    if dropped:
        lines.append("### Planned but not authored")
        for o in dropped:
            lines.append(f"- `{o.page_path}` — {o.note}")
        lines.append("")
    if result.skipped:
        lines.append("### Skipped (page/route already existed)")
        for path in result.skipped:
            lines.append(f"- `{path}`")
        lines.append("")

    usage_lines = render_usage_md(result.usage)
    if usage_lines:
        lines.extend(usage_lines)
        lines.append("")

    lines.append("---")
    lines.append(
        "_Opened by [docsync](https://github.com/keephq/docsync). These are "
        "newly-generated pages — review for accuracy before merging._"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference (anchoring an existing docs site)
# ---------------------------------------------------------------------------


_INFER_MARKS = {"anchored": "✓", "low_confidence": "~", "no_match": "·"}


def infer_console_summary(result: InferResult) -> str:
    """Short terminal summary for `docsync infer`: per-page anchor status.

    `✓` anchored (will be written), `~` low-confidence (reported, not written), `·`
    no match. Already-anchored pages are noted as skipped.
    """
    anchored = result.anchored()
    out = [
        f"docsync infer: {len(result.pages)} page(s) examined, {len(anchored)} anchored, "
        f"{len(result.skipped_already_anchored)} already in the manifest."
    ]
    for p in result.pages:
        mark = _INFER_MARKS.get(p.status, "·")
        if p.status == "anchored":
            srcs = "; ".join(
                f"{s.repo}:{','.join(s.globs)}"
                + (f" [{','.join(s.symbols)}]" if s.symbols else "")
                for s in p.sources
            )
            out.append(
                f"  {mark} {p.page_path} — {p.kind}"
                + (" (judged)" if p.judge_required else "")
                + f", conf {p.confidence:.2f} → {srcs}"
            )
        else:
            out.append(f"  {mark} {p.page_path} — {p.status}: {p.reason}")
    cost_line = render_usage_console(result.usage)
    if cost_line:
        out.append(cost_line)
    return "\n".join(out)
