"""Render a human-readable report + PR body from a PipelineResult."""

from __future__ import annotations

import difflib

from .models import PipelineResult


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
    return "\n".join(out)
