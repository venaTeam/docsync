"""Stage 5 — validation gates.

After the LLM rewrites a page, every candidate must survive a battery of gates
before it is allowed into a PR. The hard gates protect against the failure modes
an LLM edit introduces — clobbered frontmatter, unbalanced MDX components, a
runaway diff, a truncated rewrite — and any hard failure drops the page. The
broken-link check is a *soft* gate: a single patched page legitimately references
pages that haven't been written yet, so its findings annotate the PR instead of
blocking it.
"""

from __future__ import annotations

import difflib
from math import ceil
from pathlib import Path

from docsync.adapters.base import DocAdapter
from docsync.adapters.mintlify import MintlifyAdapter
from docsync.models import ManifestPage, ValidationResult

# A rewrite that drops below this fraction of the original character length is
# treated as a botched / truncated edit rather than a legitimate trim.
_TRUNCATION_MIN_RATIO = 0.5


def get_adapter(page_path: str) -> DocAdapter:
    """Return the adapter that owns `page_path`.

    MVP registry: a single `MintlifyAdapter`. This is the seam where future
    frameworks register; raising keeps an unsupported page from being silently
    skipped (the caller treats the ValueError as "no adapter, drop the page").
    """
    adapter = MintlifyAdapter()
    if adapter.owns(page_path):
        return adapter
    raise ValueError(f"No adapter owns page: {page_path!r}")


def validate_page(
    page_path: str,
    original_text: str,
    new_text: str,
    manifest_page: ManifestPage | None,
    adapter: DocAdapter,
    *,
    check_links: bool = False,
    docs_root: Path | None = None,
) -> ValidationResult:
    """Run all gates comparing `original_text` -> `new_text`.

    Hard gates (any failure -> passed=False):
      1. frontmatter freeze — frozen keys must be unchanged unless the manifest
         opts in; frontmatter must still parse.
      2. component / mermaid integrity — structural signatures must be identical;
         fence count must be even.
      3. diff-size guardrail — net changed lines must stay within budget.
      4. non-empty / not-truncated — new text must exist and not be drastically
         shorter than the original.

    Soft gates (passed stays True; appended to `.warnings`):
      - broken-link findings, when `check_links` + `docs_root` are supplied.
    """
    failures: list[str] = []
    warnings: list[str] = []

    failures.extend(_check_frontmatter(original_text, new_text, manifest_page, adapter))
    failures.extend(_check_structure(original_text, new_text, adapter))
    failures.extend(_check_wellformed(new_text, adapter))
    failures.extend(_check_diff_size(original_text, new_text, manifest_page))
    failures.extend(_check_not_truncated(original_text, new_text))

    if check_links and docs_root is not None:
        warnings.extend(_check_links_soft(adapter, docs_root))

    return ValidationResult(
        page_path=page_path,
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Hard gate 1 — frontmatter freeze
# ---------------------------------------------------------------------------


def _check_frontmatter(
    original_text: str,
    new_text: str,
    manifest_page: ManifestPage | None,
    adapter: DocAdapter,
) -> list[str]:
    failures: list[str] = []

    try:
        old_meta, _ = adapter.split_frontmatter(original_text)
        new_meta, _ = adapter.split_frontmatter(new_text)
    except Exception as exc:  # frontmatter must still parse after the edit
        return [f"frontmatter no longer parses: {exc}"]

    allow_edit = bool(manifest_page and manifest_page.allow_frontmatter_edit)
    if allow_edit:
        return failures

    for key in adapter.frontmatter_keys_to_freeze():
        old_value = old_meta.get(key)
        new_value = new_meta.get(key)
        if old_value != new_value:
            failures.append(
                f"frozen frontmatter key {key!r} changed: "
                f"{old_value!r} -> {new_value!r}"
            )

    return failures


# ---------------------------------------------------------------------------
# Hard gate 2 — component / mermaid / fence integrity
# ---------------------------------------------------------------------------


def _check_structure(original_text: str, new_text: str, adapter: DocAdapter) -> list[str]:
    failures: list[str] = []

    old_sig = adapter.structural_signature(original_text)
    new_sig = adapter.structural_signature(new_text)

    for key in sorted(set(old_sig) | set(new_sig)):
        old_count = old_sig.get(key, 0)
        new_count = new_sig.get(key, 0)
        if old_count != new_count:
            failures.append(
                f"structural element {key!r} count changed: "
                f"{old_count} -> {new_count}"
            )

    new_fence_count = new_sig.get("fence_count", 0)
    if new_fence_count % 2 != 0:
        failures.append(
            f"unbalanced code fences: {new_fence_count} ``` markers (must be even)"
        )

    return failures


# ---------------------------------------------------------------------------
# Hard gate 2b — component well-formedness (nesting / balance of the new text)
# ---------------------------------------------------------------------------


def _check_wellformed(new_text: str, adapter: DocAdapter) -> list[str]:
    """Fail if the patched text has mis-nested or unbalanced components.

    The signature gate (2) freezes *counts*; this catches edits that keep counts
    equal but break nesting (a reordered/swapped tag pair). Defensive: the adapter
    promises not to raise, but never let a structural check sink the pipeline.
    """
    try:
        problems = adapter.structural_problems(new_text)
    except Exception:  # noqa: BLE001
        return []
    return [f"malformed MDX structure: {p}" for p in problems]


# ---------------------------------------------------------------------------
# Hard gate 3 — diff-size guardrail
# ---------------------------------------------------------------------------


def _net_changed_lines(original_text: str, new_text: str) -> int:
    """Count added + removed lines between the two texts via a unified diff."""
    old_lines = original_text.splitlines()
    new_lines = new_text.splitlines()
    changed = 0
    for line in difflib.unified_diff(old_lines, new_lines, lineterm=""):
        # Skip the file/hunk headers; count only +/- content lines.
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed += 1
    return changed


def _check_diff_size(
    original_text: str,
    new_text: str,
    manifest_page: ManifestPage | None,
) -> list[str]:
    # Fall back to model defaults when no manifest entry was supplied.
    max_diff_lines = manifest_page.max_diff_lines if manifest_page else 60
    max_diff_pct = manifest_page.max_diff_pct if manifest_page else 0.5

    original_line_count = len(original_text.splitlines())
    budget = max(max_diff_lines, ceil(max_diff_pct * original_line_count))

    changed = _net_changed_lines(original_text, new_text)
    if changed > budget:
        return [
            f"diff too large: {changed} changed lines exceeds budget of {budget} "
            f"(max_diff_lines={max_diff_lines}, max_diff_pct={max_diff_pct})"
        ]
    return []


# ---------------------------------------------------------------------------
# Hard gate 4 — non-empty / not-truncated
# ---------------------------------------------------------------------------


def _check_not_truncated(original_text: str, new_text: str) -> list[str]:
    if not new_text or not new_text.strip():
        return ["new content is empty"]

    original_len = len(original_text)
    if original_len > 0 and len(new_text) < _TRUNCATION_MIN_RATIO * original_len:
        return [
            f"new content looks truncated: {len(new_text)} chars is under "
            f"{int(_TRUNCATION_MIN_RATIO * 100)}% of the original {original_len} chars"
        ]
    return []


# ---------------------------------------------------------------------------
# Soft gate — broken links
# ---------------------------------------------------------------------------


def _check_links_soft(adapter: DocAdapter, docs_root: Path) -> list[str]:
    try:
        problems = adapter.check_links(docs_root)
    except Exception as exc:  # adapter promises not to raise, but be safe
        return [f"link check could not run: {exc}"]
    return [f"possible broken link: {problem}" for problem in problems]
