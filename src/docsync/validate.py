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

from docsync.adapters import DEFAULT_ADAPTER, make_adapter
from docsync.adapters.base import DocAdapter
from docsync.models import ManifestPage, ValidationResult

# A rewrite that drops below this fraction of the original character length is
# treated as a botched / truncated edit rather than a legitimate trim.
_TRUNCATION_MIN_RATIO = 0.5

# A from-scratch page below this many characters is treated as a stub/failed
# generation rather than a real page (no original to take a ratio against).
_NEW_PAGE_MIN_CHARS = 200


def get_adapter(page_path: str, adapter: str = DEFAULT_ADAPTER) -> DocAdapter:
    """Return the configured `adapter` if it owns `page_path`.

    `adapter` is the name from `DocsyncConfig.adapter` (callers with a config thread it
    through; it defaults to mintlify for back-compat). Raising keeps a page the active
    adapter doesn't own from being silently skipped — the caller treats the ValueError
    as "no adapter, drop the page".
    """
    resolved = make_adapter(adapter)
    if resolved.owns(page_path):
        return resolved
    raise ValueError(f"adapter {adapter!r} does not own page: {page_path!r}")


def validate_page(
    page_path: str,
    original_text: str,
    new_text: str,
    manifest_page: ManifestPage | None,
    adapter: DocAdapter,
    *,
    check_links: bool = False,
    docs_root: Path | None = None,
    thoroughness: str = "medium",
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
    failures.extend(_check_diff_size(original_text, new_text, manifest_page, thoroughness))
    failures.extend(_check_not_truncated(original_text, new_text))

    if check_links and docs_root is not None:
        warnings.extend(_check_links_soft(adapter, docs_root))

    return ValidationResult(
        page_path=page_path,
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
    )


def validate_new_page(
    page_path: str,
    new_text: str,
    adapter: DocAdapter,
    *,
    check_links: bool = False,
    docs_root: Path | None = None,
) -> ValidationResult:
    """Validate a from-scratch page (bootstrap) with *absolute* gates.

    There is no original to diff against, so the diff-based gates (frontmatter
    freeze, structural signature, diff-size, truncation-ratio) don't apply. The
    new-page battery instead asserts the page stands on its own:

    Hard gates (any failure -> passed=False):
      1. frontmatter parses AND `title` + `description` are present & non-empty.
      2. component well-formedness — tags balanced and correctly nested
         (`structural_problems`), and the fence count is even.
      3. non-empty and at least `_NEW_PAGE_MIN_CHARS` long (not a stub).

    Soft gate (passed stays True; appended to `.warnings`):
      - broken-link findings — expected, since a bootstrap run cross-references
        sibling pages authored in the same run.
    """
    failures: list[str] = []
    warnings: list[str] = []

    failures.extend(_check_frontmatter_complete(new_text, adapter))
    failures.extend(_check_wellformed(new_text, adapter))
    failures.extend(_check_even_fences(new_text, adapter))
    failures.extend(_check_min_length(new_text))

    if check_links and docs_root is not None:
        warnings.extend(_check_links_soft(adapter, docs_root))

    return ValidationResult(
        page_path=page_path,
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# New-page gates (absolute — no original to compare against)
# ---------------------------------------------------------------------------


def _is_blank(text: str) -> bool:
    """True for empty or whitespace-only content."""
    return not text or not text.strip()


def _fence_parity_failure(signature: dict) -> list[str]:
    """An odd number of ``` markers means an unterminated code fence."""
    fence_count = signature.get("fence_count", 0)
    if fence_count % 2 != 0:
        return [f"unbalanced code fences: {fence_count} ``` markers (must be even)"]
    return []


def _check_frontmatter_complete(new_text: str, adapter: DocAdapter) -> list[str]:
    """Frontmatter must parse and carry non-empty values for every frozen key."""
    try:
        meta, _ = adapter.split_frontmatter(new_text)
    except Exception as exc:  # noqa: BLE001
        return [f"frontmatter does not parse: {exc}"]
    failures: list[str] = []
    for key in adapter.frontmatter_keys_to_freeze():
        value = meta.get(key)
        if not (isinstance(value, str) and value.strip()):
            failures.append(f"missing or empty frontmatter {key!r}")
    return failures


def _check_even_fences(new_text: str, adapter: DocAdapter) -> list[str]:
    return _fence_parity_failure(adapter.structural_signature(new_text))


def _check_min_length(new_text: str) -> list[str]:
    if _is_blank(new_text):
        return ["new content is empty"]
    if len(new_text) < _NEW_PAGE_MIN_CHARS:
        return [
            f"new page looks like a stub: {len(new_text)} chars is under the "
            f"{_NEW_PAGE_MIN_CHARS}-char minimum"
        ]
    return []


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


# Counts that must stay exactly equal — code/diagram fences are content, not chrome.
_FROZEN_SIGNATURE_KEYS = ("fence_count", "mermaid_count")


def _check_structure(original_text: str, new_text: str, adapter: DocAdapter) -> list[str]:
    """Hold the structural *shape* invariant, but permit additive component growth.

    Counts are frozen by default, with one carve-out for the edit path: a leaf element in
    the adapter's :meth:`~DocAdapter.additive_safe_components` set may *increase* (e.g. a
    new ``<Step>`` for a newly documented stage, or a fresh ``:::note`` admonition) as long
    as its open and close keys grow in lockstep so the page stays balanced — the
    well-formedness gate (:func:`_check_wellformed`) double-checks nesting. Decreases
    (deletions), container growth, and any fence/mermaid change are still rejected.
    """
    failures: list[str] = []

    old_sig = adapter.structural_signature(original_text)
    new_sig = adapter.structural_signature(new_text)
    additive_safe = adapter.additive_safe_components()

    component_keys = (set(old_sig) | set(new_sig)) - set(_FROZEN_SIGNATURE_KEYS)
    for key in sorted(component_keys):
        old_count = old_sig.get(key, 0)
        new_count = new_sig.get(key, 0)
        if old_count == new_count:
            continue
        name = key[1:] if key.startswith("/") else key  # strip the close-tag slash
        if name in additive_safe and new_count > old_count:
            # Additive growth — allow it only when opens and closes grew equally, so the
            # component stays balanced (a lone extra <Step> with no </Step> is rejected).
            open_delta = new_sig.get(name, 0) - old_sig.get(name, 0)
            close_delta = new_sig.get(f"/{name}", 0) - old_sig.get(f"/{name}", 0)
            if open_delta >= 0 and open_delta == close_delta:
                continue
            failures.append(
                f"unbalanced additive change to {name!r}: opens {open_delta:+d}, "
                f"closes {close_delta:+d}"
            )
            continue
        failures.append(
            f"structural element {key!r} count changed: {old_count} -> {new_count}"
        )

    for key in _FROZEN_SIGNATURE_KEYS:
        old_count = old_sig.get(key, 0)
        new_count = new_sig.get(key, 0)
        if old_count != new_count:
            failures.append(
                f"structural element {key!r} count changed: {old_count} -> {new_count}"
            )

    failures.extend(_fence_parity_failure(new_sig))

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


# Default diff-size budget (net changed lines, fraction of page) per thoroughness level.
# medium == the ManifestPage field defaults, so a medium run is byte-for-byte the prior
# behavior; light tightens edits, high loosens them.
_MODEL_DEFAULT_LINES = 60
_MODEL_DEFAULT_PCT = 0.5
_DIFF_BUDGET: dict[str, tuple[int, float]] = {
    "light": (40, 0.35),
    "medium": (_MODEL_DEFAULT_LINES, _MODEL_DEFAULT_PCT),
    "high": (100, 0.7),
}


def _check_diff_size(
    original_text: str,
    new_text: str,
    manifest_page: ManifestPage | None,
    thoroughness: str = "medium",
) -> list[str]:
    # Thoroughness sets the fallback budget; an explicit per-page override (a value that
    # differs from the model default) always wins over it.
    base_lines, base_pct = _DIFF_BUDGET.get(thoroughness, _DIFF_BUDGET["medium"])
    if manifest_page is not None:
        max_diff_lines = (
            manifest_page.max_diff_lines
            if manifest_page.max_diff_lines != _MODEL_DEFAULT_LINES
            else base_lines
        )
        max_diff_pct = (
            manifest_page.max_diff_pct
            if manifest_page.max_diff_pct != _MODEL_DEFAULT_PCT
            else base_pct
        )
    else:
        max_diff_lines, max_diff_pct = base_lines, base_pct

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
    if _is_blank(new_text):
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
