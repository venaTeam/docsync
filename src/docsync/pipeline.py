"""Pipeline orchestration — Stages 3-5 over a single CodeDiff.

Stage 1 (event capture) and Stage 2 (diff extraction) happen in the CLI before
this; Stage 6 (PR creation) happens after, in pr.py. This module is the pure
core: given a diff + manifest + config, produce a PipelineResult describing,
per impacted page, the surgical edit and whether it passed validation.

No git side effects here — the CLI decides whether to write files / open a PR.
"""

from __future__ import annotations

from pathlib import Path

from . import edits as edits_mod
from .config import DOCSYNC_DIR
from .impact import map_impact
from .models import (
    CodeDiff,
    DocsyncConfig,
    Manifest,
    PageOutcome,
    PipelineResult,
)
from .validate import get_adapter, validate_page


def _read_page(docs_root: Path, page_path: str) -> str | None:
    fp = docs_root / page_path
    if not fp.exists():
        return None
    return fp.read_text(encoding="utf-8")


def run(
    diff: CodeDiff,
    docs_repo: Path,
    config: DocsyncConfig,
    manifest: Manifest,
    *,
    use_embeddings: bool = False,
    check_links: bool = False,
    client=None,
) -> PipelineResult:
    """Map the diff to impacted pages, generate + validate edits for each.

    Returns a PipelineResult; `result.changed()` lists pages with applied,
    validated edits ready to be written to disk and opened as a PR.
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root
    result = PipelineResult(diff=diff)

    impacted = map_impact(
        diff, manifest, docs_root, config, use_embeddings=use_embeddings, client=client
    )

    for page in impacted:
        outcome = PageOutcome(page_path=page.page_path, impacted=page)
        manifest_page = manifest.page(page.page_path)

        original = _read_page(docs_root, page.page_path)
        if original is None:
            outcome.note = f"page not found on disk: {page.page_path}"
            result.outcomes.append(outcome)
            continue

        # Stage 4 — generate surgical edits.
        try:
            edit = edits_mod.generate_page_edit(
                page.page_path, original, diff, page, manifest_page, config, client=client
            )
        except Exception as exc:  # API error etc. — record and move on
            outcome.note = f"edit generation failed: {exc}"
            result.outcomes.append(outcome)
            continue
        outcome.edit = edit

        if not edit.edits:
            outcome.note = edit.no_change_reason or "model returned no edits"
            result.outcomes.append(outcome)
            continue

        # Apply edits (strict str-replace).
        try:
            new_text = edits_mod.apply_edits(original, edit)
        except edits_mod.EditApplicationError as exc:
            outcome.note = f"edit not applicable (dropped): {exc}"
            result.outcomes.append(outcome)
            continue

        # Stage 5 — validate.
        adapter = get_adapter(page.page_path)
        validation = validate_page(
            page.page_path,
            original,
            new_text,
            manifest_page,
            adapter,
            check_links=check_links,
            docs_root=docs_root,
        )
        outcome.validation = validation
        if not validation.passed:
            outcome.note = "dropped by validation: " + "; ".join(validation.failures)
            result.outcomes.append(outcome)
            continue

        outcome.new_content = new_text
        outcome.applied = True
        outcome.note = "ready"
        result.outcomes.append(outcome)

    return result


def write_changes(result: PipelineResult, docs_repo: Path, config: DocsyncConfig) -> list[str]:
    """Write validated page edits to disk. Returns the list of written page paths.

    Used in non-dry-run mode before committing / opening a PR. Never touches
    the .docsync directory.
    """
    docs_root = Path(docs_repo) / config.docs_root
    written: list[str] = []
    for outcome in result.changed():
        if outcome.page_path.startswith(DOCSYNC_DIR):
            continue
        (docs_root / outcome.page_path).write_text(outcome.new_content, encoding="utf-8")
        written.append(outcome.page_path)
    return written
