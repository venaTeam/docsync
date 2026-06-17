"""Regression-eval harness — measure docsync's page-level accuracy.

Given a *golden set* of labeled real PRs (each: repo, base, head, and the doc
pages docsync is expected to impact / edit), this module runs docsync's impact
mapping (and optionally the full LLM edit pipeline) over every case and scores
page-level precision / recall / F1 against the labels. That turns docsync's
accuracy into a single reproducible number and guards against regressions.

Two evaluation modes:

* ``mode="map"`` — FREE, no LLM. Actual pages are the unique ``page_path`` values
  from :func:`docsync.impact.find_anchor_candidates`. This measures the
  *mapping* (recall-oriented) stage.
* ``mode="full"`` — runs the LLM edit pipeline (:func:`docsync.pipeline.run`) and
  takes the pages it actually *edits* (``PipelineResult.changed()``). This
  measures end-to-end *edit* precision and needs an Anthropic ``client``.

IMPORTANT — what ``expected_pages`` in the golden set means
-----------------------------------------------------------
The golden fixture's ``expected_pages`` are **EDIT-stage** expectations: the
pages docsync should actually *edit*. Some cases legitimately *map* (anchor) a
page without *editing* it — e.g. a soft-delete route that adds new, undocumented
behavior, or a watcher task added in a lifespan: the mapping stage anchors the
page (a source file it watches changed) but the edit stage correctly produces no
change because nothing in the existing text is invalidated.

Consequently, for those true-negative-at-edit cases, ``mode="map"`` is *expected*
to over-select (it maps pages it won't edit), which lowers map-mode precision.
That gap between mapping recall and edit precision is the documented, intended
behavior of the two-stage design — not a bug. Read map-mode and full-mode scores
accordingly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from .impact import find_anchor_candidates
from .models import CodeDiff, DocsyncConfig, Manifest

# A diff builder: (repo, base, head) -> CodeDiff. Injectable so tests avoid the
# network; defaults to the real GitHub path in :func:`run_eval`.
DiffFn = Callable[[str, str, str], CodeDiff]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class GoldenCase(BaseModel):
    """One labeled real PR in the golden set."""

    repo: str  # canonical owner/name (the matcher normalizes forks)
    base: str  # base ref/sha of the compare
    head: str  # head ref/sha of the compare
    label: str  # human-readable description of what the PR does
    expected_pages: list[str] = Field(default_factory=list)  # EDIT-stage expectations


class CaseResult(BaseModel):
    """Per-case scoring outcome (page-level confusion counts)."""

    label: str
    repo: str
    expected: list[str] = Field(default_factory=list)
    actual: list[str] = Field(default_factory=list)
    tp: int = 0  # pages in both expected and actual
    fp: int = 0  # pages in actual but not expected
    fn: int = 0  # pages in expected but not actual


class EvalReport(BaseModel):
    """Aggregate report over a golden set: per-case results + micro-averaged scores."""

    cases: list[CaseResult] = Field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    n_cases: int = 0


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------


def score_case(expected: set[str], actual: set[str]) -> tuple[int, int, int]:
    """Score one case at the page level. Returns ``(tp, fp, fn)``.

    Pure: depends only on the two page sets.
    """
    tp = len(expected & actual)
    fp = len(actual - expected)
    fn = len(expected - actual)
    return tp, fp, fn


def aggregate(results: list[CaseResult]) -> tuple[float, float, float]:
    """Micro-averaged ``(precision, recall, f1)`` over summed tp/fp/fn.

    Micro-averaging sums the confusion counts across all cases first, then
    computes the rates once — so larger cases contribute proportionally more.
    Each rate guards division by zero by returning ``0.0``. Pure.
    """
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Golden-set loading
# ---------------------------------------------------------------------------


def load_golden(path: Path) -> list[GoldenCase]:
    """Load a golden set from a JSON file into :class:`GoldenCase` models.

    Accepts either a top-level list ``[ {...}, ... ]`` or an object with a
    ``"cases"`` key ``{"cases": [ {...}, ... ]}``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases", [])
    return [GoldenCase.model_validate(item) for item in data]


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def _map_actual_pages(diff: CodeDiff, manifest: Manifest) -> list[str]:
    """mode="map": sorted unique anchored page paths (FREE — no LLM)."""
    candidates = find_anchor_candidates(diff, manifest)
    return sorted({c.page_path for c in candidates})


def _full_actual_pages(
    diff: CodeDiff,
    docs_repo: Path,
    config: DocsyncConfig,
    manifest: Manifest,
    client,
) -> list[str]:
    """mode="full": sorted page paths the LLM pipeline actually edits."""
    from . import pipeline  # local import: keep the LLM path off the map-mode hot path

    result = pipeline.run(diff, docs_repo, config, manifest, client=client)
    return sorted(o.page_path for o in result.changed())


def run_eval(
    cases: list[GoldenCase],
    docs_repo: Path,
    config: DocsyncConfig,
    manifest: Manifest,
    *,
    mode: str = "map",
    client=None,
    diff_fn: Optional[DiffFn] = None,
) -> EvalReport:
    """Run every golden case and score it; return the aggregate :class:`EvalReport`.

    For each case the diff is built via ``diff_fn(repo, base, head)`` (defaults to
    :func:`docsync.diff.diff_github`, injectable so tests stay offline). Then:

    * ``mode="map"`` — actual pages = unique anchored ``page_path`` from
      :func:`find_anchor_candidates` (FREE, no LLM).
    * ``mode="full"`` — actual pages = ``page_path`` for each outcome in
      ``pipeline.run(...).changed()`` (LLM path; requires ``client``).

    A failure while processing one case is caught and recorded as a case with
    ``actual=[]`` and a note appended to its label — the whole run never aborts.
    """
    if diff_fn is None:
        from .diff import diff_github  # lazy: avoids importing gh path in offline tests

        def diff_fn(repo: str, base: str, head: str) -> CodeDiff:  # type: ignore[misc]
            return diff_github(repo, base, head)

    if mode not in ("map", "full"):
        raise ValueError(f"unknown eval mode {mode!r} (expected 'map' or 'full')")

    results: list[CaseResult] = []
    for case in cases:
        expected = list(dict.fromkeys(case.expected_pages))  # de-dupe, keep order
        try:
            diff = diff_fn(case.repo, case.base, case.head)
            if mode == "map":
                actual = _map_actual_pages(diff, manifest)
            else:
                actual = _full_actual_pages(diff, docs_repo, config, manifest, client)
            label = case.label
        except Exception as exc:  # noqa: BLE001 — one bad case must not kill the run
            actual = []
            label = f"{case.label} [ERROR: {type(exc).__name__}: {exc}]"

        tp, fp, fn = score_case(set(expected), set(actual))
        results.append(
            CaseResult(
                label=label,
                repo=case.repo,
                expected=expected,
                actual=actual,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )

    precision, recall, f1 = aggregate(results)
    return EvalReport(
        cases=results,
        precision=precision,
        recall=recall,
        f1=f1,
        n_cases=len(results),
    )
