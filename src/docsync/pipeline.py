"""Pipeline orchestration — Stages 3-5 over a single CodeDiff.

Stage 1 (event capture) and Stage 2 (diff extraction) happen in the CLI before
this; Stage 6 (PR creation) happens after, in pr.py. This module is the pure
core: given a diff + manifest + config, produce a PipelineResult describing,
per impacted page, the surgical edit and whether it passed validation.

No git side effects here — the CLI decides whether to write files / open a PR.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import critique as critique_mod
from . import edits as edits_mod
from .config import DOCSYNC_DIR
from .cost import MeteredClient, UsageMeter
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
    self_critique: bool = False,
    min_confidence: float | None = None,
    max_pages: int | None = None,
    client=None,
    meter: UsageMeter | None = None,
) -> PipelineResult:
    """Map the diff to impacted pages, generate + validate edits for each.

    Returns a PipelineResult; `result.changed()` lists pages with applied,
    validated edits ready to be written to disk and opened as a PR.

    All LLM calls (judge, edit, critique) go through one injected `client`; it is
    wrapped in a `MeteredClient` so the run's token usage + estimated cost land on
    `result.usage`. Pass a shared `meter` to accumulate across multiple runs (eval).
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root
    result = PipelineResult(diff=diff)

    # Meter every LLM call by wrapping the client once, here. When no client is
    # supplied the stages lazily build their own (unmetered) client; the CLI and
    # eval always pass one, so real runs are always accounted for.
    meter = meter or UsageMeter()
    if client is not None:
        client = MeteredClient(client, meter)

    # Ship-safety: skip the edit stage for low-confidence pages (CLI flag overrides
    # the configured floor). Anchor autopass is 1.0, so this only gates judge/
    # embedding-sourced pages — useful for a conservative first rollout.
    confidence_floor = (
        min_confidence if min_confidence is not None else config.min_edit_confidence
    )

    # Persist the embeddings index here so repeated runs reuse it (CI caches this dir).
    cache_dir = docs_repo / DOCSYNC_DIR / "state" / "embeddings"
    impacted = map_impact(
        diff, manifest, docs_root, config,
        use_embeddings=use_embeddings, cache_dir=cache_dir, client=client,
    )
    # Edit highest-confidence pages first; secondary key makes drops deterministic.
    impacted = sorted(impacted, key=lambda p: (-p.confidence, p.page_path))

    # Partition by the confidence floor, then apply the spend cap to what survives.
    # The cap counts only pages that would reach the (expensive) edit stage, so
    # below-floor pages never starve high-confidence ones out of the budget.
    eligible = [p for p in impacted if p.confidence >= confidence_floor]
    below_floor = [p for p in impacted if p.confidence < confidence_floor]
    cap = max_pages if max_pages is not None else config.max_pages_per_run
    if cap and cap > 0:
        to_edit, capped = eligible[:cap], eligible[cap:]
    else:
        to_edit, capped = eligible, []

    # Cache the (run-invariant) diff as a shared prompt block when it pays off
    # (multi-page + large diff). Primed on page 1 below, then read by the rest.
    cache_diff = edits_mod.should_cache_diff(diff, len(to_edit))

    def _process_page(page) -> PageOutcome:
        """Run stages 4-5 for one page and return a self-contained outcome.

        Pure w.r.t. shared state and never raises — safe under a thread pool (an
        escaping exception would abort the whole map). The shared client is the
        thread-safe MeteredClient; the meter is lock-guarded.
        """
        outcome = PageOutcome(page_path=page.page_path, impacted=page)
        try:
            manifest_page = manifest.page(page.page_path)
            original = _read_page(docs_root, page.page_path)
            if original is None:
                outcome.note = f"page not found on disk: {page.page_path}"
                return outcome

            # Stage 4 — generate surgical edits.
            try:
                edit = edits_mod.generate_page_edit(
                    page.page_path, original, diff, page, manifest_page, config,
                    cache_diff=cache_diff, client=client,
                )
            except Exception as exc:  # API error etc. — record and move on
                outcome.note = f"edit generation failed: {exc}"
                return outcome
            outcome.edit = edit
            if not edit.edits:
                outcome.note = edit.no_change_reason or "model returned no edits"
                return outcome

            # Stage 4b — adversarial self-critique (opt-in). Best-effort: on failure
            # keep the original edit rather than blocking the page.
            if self_critique:
                try:
                    verdict = critique_mod.critique_page_edit(
                        client, diff=diff, page_path=page.page_path,
                        page_edit=edit, model=config.models.judge_model,
                    )
                    edit = critique_mod.apply_critique(edit, verdict)
                    outcome.edit = edit
                    if not edit.edits:
                        outcome.note = "dropped by self-critique: " + (
                            verdict.reason or "no edits survived"
                        )
                        return outcome
                except Exception as exc:  # noqa: BLE001 - must never block the page
                    outcome.note = f"self-critique skipped: {exc}"

            # Apply edits (strict str-replace).
            try:
                new_text = edits_mod.apply_edits(original, edit)
            except edits_mod.EditApplicationError as exc:
                outcome.note = f"edit not applicable (dropped): {exc}"
                return outcome

            # Stage 5 — validate.
            adapter = get_adapter(page.page_path)
            validation = validate_page(
                page.page_path, original, new_text, manifest_page, adapter,
                check_links=check_links, docs_root=docs_root,
            )
            outcome.validation = validation
            if not validation.passed:
                outcome.note = "dropped by validation: " + "; ".join(validation.failures)
                return outcome

            outcome.new_content = new_text
            outcome.applied = True
            outcome.note = "ready"
            return outcome
        except Exception as exc:  # noqa: BLE001 - a worker must never raise
            outcome.note = f"unexpected error: {type(exc).__name__}: {exc}"
            return outcome

    # Run the edit stage concurrently — pages are independent; map() preserves order.
    workers = max(1, min(config.max_parallel_requests, len(to_edit)))
    if workers <= 1:
        edited = [_process_page(p) for p in to_edit]
    elif cache_diff:
        # Prime the shared-diff cache on page 1 (serial), then fan out the rest so
        # they read the cache instead of each re-writing the same diff block.
        primed = _process_page(to_edit[0])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            edited = [primed, *executor.map(_process_page, to_edit[1:])]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            edited = list(executor.map(_process_page, to_edit))

    capped_outcomes = [
        PageOutcome(
            page_path=p.page_path, impacted=p,
            note=f"skipped: max-pages-per-run cap ({cap}) reached",
        )
        for p in capped
    ]
    below_outcomes = [
        PageOutcome(
            page_path=p.page_path, impacted=p,
            note=(
                f"skipped: confidence {p.confidence:.2f} below floor "
                f"{confidence_floor:.2f}"
            ),
        )
        for p in below_floor
    ]

    result.outcomes = edited + capped_outcomes + below_outcomes
    result.usage = meter.finalize()
    return result


def write_changes(result: PipelineResult, docs_repo: Path, config: DocsyncConfig) -> list[str]:
    """Write validated page edits to disk. Returns the written paths, repo-root-relative.

    Page paths are stored relative to ``docs_root``; the returned paths are prefixed
    with ``config.docs_root`` so they're relative to the docs *repo* root — what
    ``pr.open_pr`` needs for ``git add`` (it runs from the repo root, and docs may
    live in a subdirectory like ``docs/``). Never touches the .docsync directory.
    """
    docs_root = Path(docs_repo) / config.docs_root
    prefix = Path(config.docs_root)
    written: list[str] = []
    for outcome in result.changed():
        if outcome.page_path.startswith(DOCSYNC_DIR):
            continue
        (docs_root / outcome.page_path).write_text(outcome.new_content, encoding="utf-8")
        written.append((prefix / outcome.page_path).as_posix())
    return written
