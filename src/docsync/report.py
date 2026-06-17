"""Render a human-readable report + PR body from a PipelineResult."""

from __future__ import annotations

import difflib

from .cost import render_usage_console, render_usage_md
from .models import BootstrapResult, PipelineResult


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
    """Short terminal summary for `docsync bootstrap`.

    Lists every planned page with its status: `✓` authored, `·` dropped, or `•`
    planned-but-not-yet-authored (the plan-only case, where outcomes are empty).
    """
    authored = result.authored()
    out = [
        f"docsync bootstrap: {len(result.plan.pages)} page(s) planned, "
        f"{len(authored)} authored, {len(result.skipped)} skipped (collisions)."
    ]
    by_outcome = {o.page_path: o for o in result.outcomes}
    for p in result.plan.pages:
        o = by_outcome.get(p.page_path)
        if o is None:  # plan-only: planned but not authored
            srcs = f"  ← {', '.join(p.source_paths[:3])}" if p.source_paths else ""
            out.append(f"  • {p.page_path} — {p.title}{srcs}")
        else:
            mark = "✓" if o.applied else "·"
            out.append(f"  {mark} {p.page_path} — {o.note}")
    for path in result.skipped:
        out.append(f"  ⤫ {path} — skipped: already exists")
    cost_line = render_usage_console(result.usage)
    if cost_line:
        out.append(cost_line)
    return "\n".join(out)


def bootstrap_pr_body(result: BootstrapResult, *, group: str) -> str:
    """Markdown PR body for a bootstrap run: pages authored + nav + manifest notes."""
    authored = result.authored()
    lines: list[str] = []
    lines.append(f"## docsync — bootstrapped documentation for `{result.repo}`")
    lines.append("")
    lines.append(
        f"Generated {len(authored)} new page(s) from a read-only scan of the source "
        f"repo, registered under the **{group}** nav group, with manifest anchors so "
        "`docsync run` can keep them in sync going forward."
    )
    lines.append("")
    if authored:
        lines.append(f"### Authored {len(authored)} page(s)")
        for o in authored:
            planned = next((p for p in result.plan.pages if p.page_path == o.page_path), None)
            title = f" — {planned.title}" if planned else ""
            lines.append(f"- **`{o.page_path}`**{title}")
            for w in (o.validation.warnings if o.validation else []):
                lines.append(f"  - ⚠️ {w}")
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
