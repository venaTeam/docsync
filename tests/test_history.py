"""Tests for run-history persistence (history.py) — append, load, distill, hygiene.

No network, no LLM: results are built by hand and written to a tmp docs repo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from docsync import history
from docsync.models import (
    BootstrapResult,
    CodeDiff,
    DocPlan,
    EditOp,
    ModelUsage,
    PageEdit,
    PageOutcome,
    PipelineResult,
    PlannedPage,
    RunRecord,
    RunUsage,
    ValidationResult,
)

_WHEN = datetime(2026, 6, 20, 17, 39, tzinfo=timezone.utc)


def _pipeline_result() -> PipelineResult:
    diff = CodeDiff(
        repo="venaTeam/docsync",
        base_sha="aaaaaaaa1111",
        head_sha="bbbbbbbb2222",
        pr_number=42,
        pr_title="Optimize content generation",
    )
    usage = RunUsage(
        by_model=[
            ModelUsage(
                model="claude-opus-4-8", stage="edit", calls=2,
                input_tokens=1000, output_tokens=500, cost_usd=0.12,
            )
        ],
        calls=2, input_tokens=1000, output_tokens=500, cost_usd=0.12, cache_hit_rate=0.4,
    )
    return PipelineResult(
        diff=diff,
        usage=usage,
        outcomes=[
            PageOutcome(
                page_path="reference/cli.mdx",
                applied=True,
                note="ready",
                edit=PageEdit(
                    edits=[
                        EditOp(
                            find="SECRET-FIND-TOKEN",
                            replace="SECRET-REPLACE-TOKEN",
                            rationale="document the --polish flag",
                        )
                    ]
                ),
                validation=ValidationResult(page_path="reference/cli.mdx", passed=True),
                new_content="SECRET-BODY-CONTENT",
            ),
            PageOutcome(
                page_path="concepts/how.mdx",
                applied=False,
                note="dropped: diff too large",
                validation=ValidationResult(
                    page_path="concepts/how.mdx", passed=False, failures=["too big"]
                ),
            ),
        ],
    )


def _bootstrap_result() -> BootstrapResult:
    return BootstrapResult(
        repo="venaTeam/docsync",
        plan=DocPlan(
            pages=[
                PlannedPage(page_path="a.mdx", title="A", kind="reference", section="Reference", order=1, summary="s"),
                PlannedPage(page_path="b.mdx", title="B", kind="concept", section="Concepts", order=1, summary="s"),
            ]
        ),
        skipped=["c.mdx"],
        outcomes=[PageOutcome(page_path="a.mdx", applied=True, note="authored", new_content="...")],
    )


def test_run_record_round_trips_jsonl(tmp_path: Path):
    rec = history.record_run(
        tmp_path, _pipeline_result(), command="run", status="opened",
        pr_url="https://github.com/venaTeam/docsync/pull/42", when=_WHEN,
    )
    loaded = history.load_runs(tmp_path)
    assert len(loaded) == 1
    assert loaded[0] == rec
    assert loaded[0].pr_url.endswith("/pull/42")
    # exactly one line in the file
    text = history.runs_path(tmp_path).read_text()
    assert text.count("\n") == 1


def test_appends_accumulate(tmp_path: Path):
    history.record_run(tmp_path, _pipeline_result(), command="run", when=_WHEN)
    history.record_run(tmp_path, _bootstrap_result(), command="bootstrap", status="patched", when=_WHEN)
    loaded = history.load_runs(tmp_path)
    assert [r.command for r in loaded] == ["run", "bootstrap"]


def test_distills_pipeline_counts_and_rationales(tmp_path: Path):
    rec = history.record_run(tmp_path, _pipeline_result(), command="run", when=_WHEN)
    assert rec.counts == {"impacted": 2, "updated": 1, "dropped": 1}
    applied = next(p for p in rec.pages if p.applied)
    assert applied.rationales == ["document the --polish flag"]
    assert applied.edit_count == 1
    dropped = next(p for p in rec.pages if not p.applied)
    assert dropped.validation_passed is False
    assert rec.usage.cost_usd == 0.12
    assert rec.usage.by_model[0].tokens == 1500  # prompt(1000) + output(500)


def test_secret_hygiene_no_body_text_persisted(tmp_path: Path):
    history.record_run(tmp_path, _pipeline_result(), command="run", when=_WHEN)
    raw = history.runs_path(tmp_path).read_text()
    # The persisted line must carry rationale text but never find/replace/new_content bodies.
    assert "document the --polish flag" in raw
    assert "SECRET-FIND-TOKEN" not in raw
    assert "SECRET-REPLACE-TOKEN" not in raw
    assert "SECRET-BODY-CONTENT" not in raw
    assert '"find"' not in raw and '"new_content"' not in raw


def test_distills_bootstrap_counts(tmp_path: Path):
    rec = history.record_run(tmp_path, _bootstrap_result(), command="bootstrap", when=_WHEN)
    assert rec.command == "bootstrap"
    assert rec.counts == {"planned": 2, "authored": 1, "skipped": 1}
    assert rec.base_sha is None and rec.head_sha is None


def test_load_runs_skips_malformed_lines(tmp_path: Path):
    history.record_run(tmp_path, _pipeline_result(), command="run", when=_WHEN)
    path = history.runs_path(tmp_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write('{"valid_json_but": "not a RunRecord"}\n')
        fh.write("\n")  # blank line
    history.record_run(tmp_path, _bootstrap_result(), command="bootstrap", when=_WHEN)
    loaded = history.load_runs(tmp_path)
    assert len(loaded) == 2  # two good records; the garbage lines are skipped
    assert all(isinstance(r, RunRecord) for r in loaded)


def test_load_runs_limit(tmp_path: Path):
    for _ in range(5):
        history.record_run(tmp_path, _pipeline_result(), command="run", when=_WHEN)
    assert len(history.load_runs(tmp_path, limit=2)) == 2
    assert len(history.load_runs(tmp_path)) == 5


def test_load_runs_missing_file_is_empty(tmp_path: Path):
    assert history.load_runs(tmp_path) == []
