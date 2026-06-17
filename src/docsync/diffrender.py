"""Shared compact diff renderer for LLM prompts.

Both the Opus editor (`edits.py`) and the Haiku critic (`critique.py`) feed the
model the same compact rendering of a `CodeDiff` — per-file sections with status,
changed symbols, and hunks — capped so a giant PR can't blow up the prompt. Keeping
it in one place stops the two stages from drifting, and gives a single home for a
future cached-diff prompt block.
"""

from __future__ import annotations

from .models import CodeDiff

# Cap the rendered diff so a giant PR can't blow up the prompt.
MAX_DIFF_CHARS = 12_000


def render_diff(diff: CodeDiff, *, max_chars: int = MAX_DIFF_CHARS) -> str:
    """A compact rendering of `diff`, truncated at ~`max_chars` characters."""
    header = f"repo: {diff.repo}\npr_title: {diff.pr_title or '(none)'}\n"
    parts: list[str] = []
    used = len(header)
    for f in diff.files:
        symbols = ", ".join(f.changed_symbols) or "(none)"
        section_lines = [
            f"\n## file: {f.path} ({f.status.value})",
            f"changed symbols: {symbols}",
        ]
        section_lines.extend(f.hunks)
        section = "\n".join(section_lines)
        if used + len(section) > max_chars:
            remaining = max_chars - used
            if remaining > 0:
                parts.append(section[:remaining])
            parts.append("\n... (diff truncated)")
            break
        parts.append(section)
        used += len(section)
    return header + "".join(parts)
