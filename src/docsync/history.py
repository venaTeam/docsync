"""Run-history persistence — one distilled `RunRecord` per run, appended to JSONL.

docsync's `run`/`bootstrap` produce rich in-memory results then discard them. This
module distills each result into a small, history-safe `RunRecord` and appends it to
`<docs-repo>/.docsync/state/runs.jsonl`, which `docsync dashboard` reads back.

Design notes:
- **Append-only JSONL** (one line per run) so concurrent CI writers don't clobber a
  shared file and the per-run git diff (when uploaded as an artifact) is one line.
- **Secret hygiene:** distillation keeps only counts + the model's per-edit `rationale`.
  It never serializes `EditOp.find`/`replace` or `PageOutcome.new_content` (page/source
  body text), so the history can't leak doc content.
- **Tolerant load:** malformed or future-schema lines are skipped, so a schema bump
  never bricks the dashboard.

The file is local, derived state (gitignored, never committed into a docs PR) — mirrors
the `.docsync/state/` home of the cursor file in `config.py`.
"""

from __future__ import annotations

import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from . import config as cfg
from .models import (
    BootstrapResult,
    ModelStageRecord,
    PageOutcome,
    PageRecord,
    PipelineResult,
    RunRecord,
    RunStatus,
    RunUsage,
    UsageRecord,
)

RUNS_FILE = "state/runs.jsonl"

# Keep the persisted history bounded. Each line stays small (capped per-page diffs, no
# whole-page bodies), so this caps the file at a few hundred KB while preserving a long,
# useful timeline.
_MAX_RECORDS = 1000

# Per-page diff cap: enough to show the actual change, bounded so one big edit can't
# bloat the record (and so we never store a whole untouched page body).
_DIFF_MAX_LINES = 120


def runs_path(docs_repo: Path) -> Path:
    return cfg.docsync_dir(docs_repo) / RUNS_FILE


def record_run(
    docs_repo: Path,
    result: Union[PipelineResult, BootstrapResult],
    *,
    command: str,
    status: RunStatus = "opened",
    pr_url: Optional[str] = None,
    originals: Optional[dict[str, str]] = None,
    when: Optional[datetime] = None,
) -> RunRecord:
    """Distill `result` into a `RunRecord` and append it as one JSONL line.

    `originals` maps page_path -> pre-edit text; when present, each edited page gets a
    bounded unified diff so the dashboard can show the actual change. `when` is injectable
    for deterministic tests; defaults to now (UTC). The record is appended atomically (one
    `write`), then trimmed to the most recent `_MAX_RECORDS` lines if it grew past the cap.
    """
    ts = (when or datetime.now(timezone.utc)).isoformat()
    if isinstance(result, BootstrapResult):
        record = _distill_bootstrap(result, command=command, status=status, ts=ts, pr_url=pr_url)
    else:
        record = _distill_pipeline(
            result, command=command, status=status, ts=ts, pr_url=pr_url, originals=originals
        )

    path = runs_path(docs_repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
    _trim(path)
    return record


def load_runs(docs_repo: Path, *, limit: Optional[int] = None) -> list[RunRecord]:
    """Load persisted runs oldest-first; `limit` keeps only the most recent N.

    Tolerant: blank lines and lines that don't parse as a current `RunRecord` are
    skipped rather than raising, so a partial write or a schema change degrades to
    "fewer rows" instead of a crash.
    """
    path = runs_path(docs_repo)
    if not path.exists():
        return []
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(RunRecord.model_validate_json(line))
        except ValueError:
            continue
    if limit is not None and limit >= 0:
        records = records[-limit:]
    return records


# --- distillation ----------------------------------------------------------


def _distill_pipeline(
    result: PipelineResult,
    *,
    command: str,
    status: RunStatus,
    ts: str,
    pr_url: Optional[str],
    originals: Optional[dict[str, str]] = None,
) -> RunRecord:
    pages = _page_summaries(result.outcomes, originals)
    updated = sum(1 for p in pages if p.applied)
    counts = {
        "impacted": len(pages),
        "updated": updated,
        "dropped": len(pages) - updated,
    }
    diff = result.diff
    return RunRecord(
        timestamp=ts,
        command="run",
        status=status,
        repo=diff.repo,
        base_sha=diff.base_sha,
        head_sha=diff.head_sha,
        pr_number=diff.pr_number,
        pr_title=diff.pr_title,
        pr_url=pr_url,
        counts=counts,
        pages=pages,
        usage=_usage_summary(result.usage),
    )


def _distill_bootstrap(
    result: BootstrapResult, *, command: str, status: RunStatus, ts: str, pr_url: Optional[str]
) -> RunRecord:
    pages = _page_summaries(result.outcomes)
    authored = sum(1 for p in pages if p.applied)
    counts = {
        "planned": len(result.plan.pages),
        "authored": authored,
        "skipped": len(result.skipped),
    }
    return RunRecord(
        timestamp=ts,
        command="bootstrap",
        status=status,
        repo=result.repo,
        counts=counts,
        pages=pages,
        usage=_usage_summary(result.usage),
    )


def _page_summaries(
    outcomes: list[PageOutcome], originals: Optional[dict[str, str]] = None
) -> list[PageRecord]:
    """One `PageRecord` per outcome — counts, rationales, and a bounded change diff."""
    out: list[PageRecord] = []
    for o in outcomes:
        edits = o.edit.edits if o.edit else []
        before = (originals or {}).get(o.page_path)
        diff = None
        if before is not None and o.new_content is not None:
            diff = _page_diff(o.page_path, before, o.new_content)
        out.append(
            PageRecord(
                page_path=o.page_path,
                applied=o.applied,
                note=o.note,
                edit_count=len(edits),
                rationales=[e.rationale for e in edits],
                diff=diff,
                validation_passed=o.validation.passed if o.validation else None,
                warnings=list(o.validation.warnings) if o.validation else [],
            )
        )
    return out


def _page_diff(path: str, before: str, after: str) -> Optional[str]:
    """A bounded unified diff (original -> new) for one page, or None if unchanged."""
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
            n=3,
        )
    )
    if not lines:
        return None
    if len(lines) > _DIFF_MAX_LINES:
        dropped = len(lines) - _DIFF_MAX_LINES
        lines = lines[:_DIFF_MAX_LINES] + [f"… (+{dropped} more diff line(s) truncated)"]
    return "\n".join(lines)


def _usage_summary(usage: Optional[RunUsage]) -> Optional[UsageRecord]:
    if usage is None:
        return None
    by_model = [
        ModelStageRecord(
            model=m.model,
            stage=m.stage,
            calls=m.calls,
            tokens=m.prompt_tokens + m.output_tokens,
            cost_usd=m.cost_usd,
        )
        for m in usage.by_model
    ]
    return UsageRecord(
        cost_usd=usage.cost_usd,
        calls=usage.calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_hit_rate=usage.cache_hit_rate,
        estimated=usage.estimated,
        by_model=by_model,
    )


def _trim(path: Path) -> None:
    """Bound the file to the most recent `_MAX_RECORDS` lines (rare rewrite path)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= _MAX_RECORDS:
        return
    kept = lines[-_MAX_RECORDS:]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
