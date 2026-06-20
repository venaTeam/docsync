"""Tests for the dashboard (dashboard.py) — aggregation, budget, render, command, serve.

No network, no LLM: RunRecords are built directly and the dashboard renders to a string.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from docsync import dashboard as dash
from docsync.cli import app
from docsync.models import (
    DocsyncConfig,
    Manifest,
    ManifestPage,
    ManifestSource,
    ModelStageRecord,
    PageRecord,
    RunRecord,
    UsageRecord,
)

runner = CliRunner()


def _usage(cost: float) -> UsageRecord:
    return UsageRecord(
        cost_usd=cost, calls=2, input_tokens=1000, output_tokens=500, cache_hit_rate=0.5,
        by_model=[ModelStageRecord(model="claude-opus-4-8", stage="edit", calls=2, tokens=1500, cost_usd=cost)],
    )


def _run_record(ts: str, *, command="run", status="opened", cost=0.10, **kw) -> RunRecord:
    counts = {"impacted": 2, "updated": 1, "dropped": 1} if command == "run" else {
        "planned": 3, "authored": 2, "skipped": 1
    }
    return RunRecord(
        timestamp=ts, command=command, status=status, repo="venaTeam/docsync",
        base_sha="aaaaaaaa", head_sha="bbbbbbbb", pr_number=42,
        counts=counts, usage=_usage(cost),
        pages=[
            PageRecord(page_path="reference/cli.mdx", applied=True, edit_count=1,
                       rationales=["document the --polish flag"]),
        ],
        **kw,
    )


def _config(budget=None) -> DocsyncConfig:
    return DocsyncConfig(monthly_budget_usd=budget)


def _manifest() -> Manifest:
    return Manifest(pages=[
        ManifestPage(path="reference/cli.mdx",
                     sources=[ManifestSource(repo="docsync", globs=["src/docsync/cli.py"], symbols=["run"])]),
    ])


# --- aggregation -----------------------------------------------------------


def test_aggregate_stats():
    runs = [
        _run_record("2026-06-10T10:00:00+00:00", cost=0.10),
        _run_record("2026-06-12T10:00:00+00:00", command="bootstrap", status="patched", cost=0.20),
        _run_record("2026-06-14T10:00:00+00:00", status="error", cost=0.05),
    ]
    s = dash.aggregate(runs)
    assert s.total_runs == 3
    assert s.run_runs == 2 and s.bootstrap_runs == 1
    assert s.pages_updated == 2  # two "run" records, each updated=1
    assert s.pages_authored == 2  # one bootstrap, authored=2
    assert abs(s.total_cost_usd - 0.35) < 1e-9
    assert abs(s.success_rate - 2 / 3) < 1e-9  # the error run is not a success
    assert s.first_ts.startswith("2026-06-10")
    assert s.last_ts.startswith("2026-06-14")


def test_aggregate_empty():
    s = dash.aggregate([])
    assert s.total_runs == 0 and s.total_cost_usd == 0.0


def test_model_stage_breakdown_sums_across_runs():
    runs = [_run_record("2026-06-10T10:00:00+00:00", cost=0.10),
            _run_record("2026-06-11T10:00:00+00:00", cost=0.20)]
    bd = dash.model_stage_breakdown(runs)
    assert len(bd) == 1
    assert bd[0].calls == 4 and bd[0].tokens == 3000
    assert abs(bd[0].cost_usd - 0.30) < 1e-9


# --- budget ----------------------------------------------------------------


def test_budget_status_projection_and_over_flag():
    # One $5 run on the 1st; viewed on the 10th of a 30-day month → ~30% elapsed →
    # projection ≈ $5 / 0.3 ≈ $16.6, which is over a $10 budget.
    runs = [_run_record("2026-06-01T00:00:00+00:00", cost=5.0)]
    now = datetime(2026, 6, 10, 0, 0, tzinfo=timezone.utc)
    b = dash.budget_status(runs, _config(budget=10.0), now=now)
    assert abs(b.month_spend - 5.0) < 1e-9
    assert b.projected_usd > 10.0
    assert b.over_budget is True


def test_budget_excludes_other_months():
    runs = [
        _run_record("2026-05-20T00:00:00+00:00", cost=9.0),  # previous month
        _run_record("2026-06-02T00:00:00+00:00", cost=1.0),  # current month
    ]
    now = datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc)
    b = dash.budget_status(runs, _config(budget=100.0), now=now)
    assert abs(b.month_spend - 1.0) < 1e-9
    assert b.over_budget is False


def test_budget_none_when_unset():
    runs = [_run_record("2026-06-02T00:00:00+00:00", cost=1.0)]
    now = datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc)
    b = dash.budget_status(runs, _config(budget=None), now=now)
    assert b.budget_usd is None and b.over_budget is False


# --- render ----------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


def test_render_html_contains_expected_stats():
    runs = [_run_record("2026-06-10T10:00:00+00:00", cost=0.50)]
    cfg = _config(budget=10.0)
    health = dash.build_health(Path("/nonexistent"), cfg, _manifest(), {})
    html = dash.render_dashboard(runs, cfg, _manifest(), health, now=_now())
    assert "docsync dashboard" in html
    assert "venaTeam/docsync" in html
    assert "/pull/42" in html  # PR link synthesized from repo + pr_number
    assert "document the --polish flag" in html  # latest-changes drill-down
    assert "this month" in html  # budget banner


def test_render_escapes_injected_html():
    runs = [_run_record("2026-06-10T10:00:00+00:00", cost=0.10,
                        pr_title="<script>alert(1)</script>")]
    # also smuggle an injection via the page rationale
    runs[0].pages[0].rationales = ["<img src=x onerror=alert(2)>"]
    cfg = _config()
    health = dash.build_health(Path("/nonexistent"), cfg, _manifest(), {})
    html = dash.render_dashboard(runs, cfg, _manifest(), health, now=_now())
    assert "<script>alert(1)</script>" not in html
    assert "<img src=x onerror=alert(2)>" not in html
    assert "&lt;img src=x onerror=alert(2)&gt;" in html


def test_render_empty_history():
    cfg = _config()
    health = dash.build_health(Path("/nonexistent"), cfg, None, {})
    html = dash.render_dashboard([], cfg, None, health, now=_now())
    assert "No runs recorded yet" in html
    assert "docsync dashboard" in html


# --- CLI command -----------------------------------------------------------


def _seed_docs(tmp_path: Path) -> Path:
    from docsync import history

    docs = tmp_path / "docs"
    (docs / ".docsync").mkdir(parents=True)
    (docs / ".docsync" / "manifest.yml").write_text(
        "pages:\n  - path: reference/cli.mdx\n", encoding="utf-8"
    )
    # write a history line directly via the persistence layer's path
    path = history.runs_path(docs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_run_record("2026-06-10T10:00:00+00:00", cost=0.33).model_dump_json() + "\n",
                    encoding="utf-8")
    return docs


def test_dashboard_command_writes_file(tmp_path: Path):
    docs = _seed_docs(tmp_path)
    out = tmp_path / "dash.html"
    result = runner.invoke(app, ["dashboard", "--docs-repo", str(docs), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    html = out.read_text()
    assert "venaTeam/docsync" in html
    assert "dashboard written to" in result.output


def test_dashboard_command_empty_history(tmp_path: Path):
    docs = tmp_path / "docs"
    (docs / ".docsync").mkdir(parents=True)
    (docs / ".docsync" / "manifest.yml").write_text("pages: []\n", encoding="utf-8")
    out = tmp_path / "dash.html"
    result = runner.invoke(app, ["dashboard", "--docs-repo", str(docs), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "no runs recorded yet" in result.output.lower()
    assert "No runs recorded yet" in out.read_text()


# --- serve handler ---------------------------------------------------------


def test_serve_handler_writes_html_without_socket():
    handler_cls = dash._handler_for("<html>hi</html>")

    # Build an instance without running BaseHTTPRequestHandler.__init__ (no socket),
    # then drive do_GET with a fake wfile and no-op response writers.
    h = handler_cls.__new__(handler_cls)
    h.wfile = io.BytesIO()
    h.send_response = lambda *a, **k: None
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda *a, **k: None
    h.do_GET()
    assert h.wfile.getvalue() == b"<html>hi</html>"
