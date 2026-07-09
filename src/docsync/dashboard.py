"""Dashboard — aggregate persisted run history into a self-contained HTML view.

`docsync dashboard` reads `.docsync/state/runs.jsonl` (see `history.py`) and renders a
single, dependency-free HTML file: overview stats, a cost/budget panel, a runs timeline,
a per-model/stage cost breakdown, the latest doc changes (drill-down), and a health
panel (config + manifest + cursors + doctor). With `--serve` the same HTML is served
over stdlib `http.server` for local browsing ("spawn a dashboard").

No third-party deps: `string`/`html`/`http.server`/`webbrowser` only — honoring docsync's
zero-web-dep, CI-first philosophy. The default surface is a static file (a natural CI
artifact); `--serve` is a local convenience and must never be used in CI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from . import config as cfg
from .models import DocsyncConfig, Manifest, ModelStageRecord, RunRecord

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class DashboardStats(BaseModel):
    """Aggregate run/bootstrap statistics rendered on the dashboard.

    Counts totals for both commands (runs, pages, cost, calls) alongside
    cache-hit and success rates. The edit-drop health fields track the `run`
    command only: a page counts as attempted once the editor was actually
    invoked on it (deliberate skips from the max-pages cap or the confidence
    floor are excluded), and a drop is an attempted page whose editor output a
    gate rejected (validation, self-critique, non-applicable find, or error);
    a correct no-change dilutes the rate as a processed page should.

    Attributes:
        total_runs: Combined number of run and bootstrap invocations recorded.
        run_runs: Number of `run` command invocations.
        bootstrap_runs: Number of `bootstrap` command invocations.
        pages_updated: Pages edited by `run`.
        pages_authored: Pages authored by `bootstrap`.
        total_cost_usd: Total LLM cost in USD across all recorded activity.
        total_calls: Total number of LLM calls.
        avg_cache_hit_rate: Average prompt-cache hit rate.
        success_rate: Fraction of invocations that succeeded.
        pages_attempted: Pages the editor was actually invoked on (`run` only).
        pages_dropped: Attempted pages whose editor output a gate rejected.
        drop_rate: Fraction of attempted pages that were dropped.
        drops_by_reason: Count of drops keyed by rejection reason.
        first_ts: Timestamp of the earliest recorded activity, if any.
        last_ts: Timestamp of the most recent recorded activity, if any.
    """
    total_runs: int = 0
    run_runs: int = 0
    bootstrap_runs: int = 0
    pages_updated: int = 0
    pages_authored: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    avg_cache_hit_rate: float = 0.0
    success_rate: float = 0.0
    # Edit-drop health (run command only): a page is "attempted" once the editor was
    # actually invoked on it — deliberate skips (max-pages cap / below confidence floor)
    # don't count. A "drop" is an attempted page the editor produced but a gate rejected
    # (validation / self-critique / non-applicable find / error). A correct no-change is
    # neither a drop nor excluded — it dilutes the rate, as a processed page should.
    pages_attempted: int = 0
    pages_dropped: int = 0
    drop_rate: float = 0.0
    drops_by_reason: dict[str, int] = Field(default_factory=dict)
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None


def _drop_reason(note: str) -> Optional[str]:
    """Classify a non-applied page's outcome note into a drop reason, or None.

    Matches the note prefixes the pipeline writes (`pipeline.py`). Returns None for
    notes that are *not* drops — a deliberate "skipped:" cap/floor or a legitimate
    "no edits" no-op — so the caller can keep them out of the dropped count.
    """
    if note.startswith("dropped by validation"):
        return "validation"
    if note.startswith("dropped by self-critique"):
        return "critique"
    if note.startswith("edit not applicable"):
        return "not_applicable"
    if note.startswith(("edit generation failed", "page not found", "unexpected error")):
        return "error"
    return None


def aggregate(runs: list[RunRecord]) -> DashboardStats:
    """Roll a list of runs up into the overview stat-card numbers."""
    if not runs:
        return DashboardStats()
    stats = DashboardStats(total_runs=len(runs))
    cache_rates: list[float] = []
    successes = 0
    for r in runs:
        if r.command == "bootstrap":
            stats.bootstrap_runs += 1
            stats.pages_authored += r.counts.get("authored", 0)
        else:
            stats.run_runs += 1
            stats.pages_updated += r.counts.get("updated", 0)
            for p in r.pages:
                if p.note.startswith("skipped:"):
                    continue  # cap/floor — the editor was never invoked on this page
                stats.pages_attempted += 1
                if p.applied:
                    continue
                reason = _drop_reason(p.note)
                if reason is None:
                    continue  # attempted but a legit no-change no-op, not a drop
                stats.pages_dropped += 1
                stats.drops_by_reason[reason] = stats.drops_by_reason.get(reason, 0) + 1
        if r.status != "error":
            successes += 1
        if r.usage:
            stats.total_cost_usd += r.usage.cost_usd
            stats.total_calls += r.usage.calls
            if r.usage.calls:
                cache_rates.append(r.usage.cache_hit_rate)
    stats.success_rate = successes / len(runs)
    stats.avg_cache_hit_rate = sum(cache_rates) / len(cache_rates) if cache_rates else 0.0
    stats.drop_rate = stats.pages_dropped / stats.pages_attempted if stats.pages_attempted else 0.0
    ordered = sorted(runs, key=lambda r: r.timestamp)
    stats.first_ts = ordered[0].timestamp
    stats.last_ts = ordered[-1].timestamp
    return stats


def cost_over_time(runs: list[RunRecord]) -> list[tuple[str, float]]:
    """Cumulative estimated cost after each run, oldest-first (for the sparkline)."""
    out: list[tuple[str, float]] = []
    total = 0.0
    for r in sorted(runs, key=lambda r: r.timestamp):
        total += r.usage.cost_usd if r.usage else 0.0
        out.append((r.timestamp, total))
    return out


def model_stage_breakdown(runs: list[RunRecord]) -> list[ModelStageRecord]:
    """Sum per-(model, stage) calls/tokens/cost across every run, cost-sorted."""
    acc: dict[tuple[str, Optional[str]], ModelStageRecord] = {}
    for r in runs:
        if not r.usage:
            continue
        for m in r.usage.by_model:
            key = (m.model, m.stage)
            cur = acc.get(key)
            if cur is None:
                acc[key] = ModelStageRecord(model=m.model, stage=m.stage)
                cur = acc[key]
            cur.calls += m.calls
            cur.tokens += m.tokens
            cur.cost_usd += m.cost_usd
    return sorted(acc.values(), key=lambda m: m.cost_usd, reverse=True)


# ---------------------------------------------------------------------------
# Budget & projection
# ---------------------------------------------------------------------------


class BudgetStatus(BaseModel):
    """Monthly spend tracked against an optional budget.

    Attributes:
        budget_usd: Configured monthly budget in USD, or None if unset.
        month_spend: Spend so far this month in USD.
        projected_usd: Projected end-of-month spend in USD, or None.
        over_budget: Whether spending has exceeded the budget.
        fraction_elapsed: Fraction of the month that has elapsed.
    """
    budget_usd: Optional[float] = None
    month_spend: float = 0.0
    projected_usd: Optional[float] = None
    over_budget: bool = False
    fraction_elapsed: float = 0.0


def _month_bounds(now: datetime) -> tuple[datetime, datetime]:
    """First instant of `now`'s month and of the next month."""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end


def budget_status(runs: list[RunRecord], config: DocsyncConfig, *, now: datetime) -> BudgetStatus:
    """Current calendar-month spend, projected month-end, and an over-budget flag.

    Projection assumes spend continues at the month-to-date rate:
    `projected = month_spend / fraction_of_month_elapsed`. `now` is injectable so the
    projection is deterministic in tests.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    start, end = _month_bounds(now)
    span = (end - start).total_seconds()
    elapsed = max((now - start).total_seconds(), 0.0)
    fraction = min(elapsed / span, 1.0) if span else 1.0

    month_spend = 0.0
    for r in runs:
        ts = _parse_ts(r.timestamp)
        if ts is not None and start <= ts < end and r.usage:
            month_spend += r.usage.cost_usd

    status = BudgetStatus(
        budget_usd=config.monthly_budget_usd,
        month_spend=month_spend,
        fraction_elapsed=fraction,
    )
    if fraction > 0:
        status.projected_usd = month_spend / fraction
    if config.monthly_budget_usd:
        reference = status.projected_usd if status.projected_usd is not None else month_spend
        status.over_budget = reference > config.monthly_budget_usd
    return status


# ---------------------------------------------------------------------------
# Health panel
# ---------------------------------------------------------------------------


class HealthPanel(BaseModel):
    """Configuration and manifest health summary for the dashboard.

    Attributes:
        edit_model: Model used for the edit/author stage.
        judge_model: Model used for the judge/critique stage.
        docs_root: Configured docs root path.
        thresholds: Configured numeric thresholds keyed by name.
        manifest_pages: Number of pages declared in the manifest.
        manifest_anchors: Number of anchors declared across the manifest.
        cursors: Last processed head per repo, keyed by repo.
        doctor_ok: Whether the doctor check passed, or None if not run.
        doctor_issues: Issues reported by the doctor check.
    """
    edit_model: str = ""
    judge_model: str = ""
    docs_root: str = "."
    thresholds: dict[str, float] = Field(default_factory=dict)
    manifest_pages: int = 0
    manifest_anchors: int = 0
    cursors: dict[str, str] = Field(default_factory=dict)
    doctor_ok: Optional[bool] = None
    doctor_issues: list[str] = Field(default_factory=list)


def build_health(
    docs_repo: Path,
    config: DocsyncConfig,
    manifest: Optional[Manifest],
    checkout: Optional[dict[str, Path]] = None,
) -> HealthPanel:
    """Assemble the health panel: config summary, manifest size, cursors, doctor status."""
    panel = HealthPanel(
        edit_model=config.models.edit_model,
        judge_model=config.models.judge_model,
        docs_root=config.docs_root,
        thresholds={
            "judge_confidence_threshold": config.judge_confidence_threshold,
            "min_edit_confidence": config.min_edit_confidence,
            "embedding_floor": config.embedding_floor,
        },
    )
    if manifest is not None:
        panel.manifest_pages = len(manifest.pages)
        panel.manifest_anchors = sum(len(p.sources) for p in manifest.pages)
    try:
        panel.cursors = cfg.load_cursors(docs_repo)
    except (OSError, ValueError):
        panel.cursors = {}
    # doctor is best-effort: it needs a manifest and is most useful with checkouts, but
    # a missing checkout only yields "unmapped repo" warnings, never a crash.
    try:
        from .scaffold import doctor as run_doctor

        report = run_doctor(docs_repo, checkout or {})
        panel.doctor_ok = report.ok
        panel.doctor_issues = (
            [f"missing page: {p}" for p in report.missing_pages]
            + [f"dead glob: {d.page} :: {d.glob}" for d in report.dead_globs]
            + [f"no checkout for {r}" for r in report.unmapped_repos]
        )
    except (FileNotFoundError, OSError):
        panel.doctor_ok = None
    return panel


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #0d1117; color: #c9d1d9; }
a { color: #58a6ff; text-decoration: none; } a:hover { text-decoration: underline; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 32px 0 12px;
     text-transform: uppercase; letter-spacing: .06em; color: #8b949e; }
.sub { color: #8b949e; margin: 0 0 24px; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 16px; }
.card .n { font-size: 24px; font-weight: 600; color: #e6edf3; }
.card .l { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: .04em; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: .05em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 11px;
         background: #1f6feb33; color: #79c0ff; }
.badge.bootstrap { background: #a371f733; color: #d2a8ff; }
.st-opened { color: #3fb950; } .st-no_change { color: #8b949e; }
.st-patched { color: #79c0ff; } .st-error { color: #f85149; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 18px; }
.banner { border-radius: 8px; padding: 14px 18px; margin: 0 0 8px; border: 1px solid; }
.banner.ok { background: #1f6feb1a; border-color: #1f6feb55; }
.banner.over { background: #f851491a; border-color: #f8514955; color: #ff7b72; }
.banner .big { font-size: 18px; font-weight: 600; }
details { border-bottom: 1px solid #21262d; padding: 8px 0; }
summary { cursor: pointer; } summary::-webkit-details-marker { color: #8b949e; }
ul.r { margin: 8px 0 4px 18px; padding: 0; color: #c9d1d9; } ul.r li { margin: 2px 0; }
.muted { color: #8b949e; } code { background: #161b22; padding: 1px 5px; border-radius: 4px; }
.change { margin: 8px 0 4px; }
pre.diff { background: #010409; border: 1px solid #21262d; border-radius: 6px; padding: 8px 0;
           margin: 6px 0 2px; overflow-x: auto;
           font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
pre.diff span { display: block; white-space: pre; padding: 0 12px; }
.diff .add { color: #3fb950; background: #2ea04326; }
.diff .del { color: #f85149; background: #f8514926; }
.diff .hunk { color: #58a6ff; background: #1f6feb1a; }
.diff .fhdr { color: #8b949e; }
svg { display: block; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 13px; }
.kv dt { color: #8b949e; } .kv dd { margin: 0; }
.warn { color: #d29922; }
"""


def render_dashboard(
    runs: list[RunRecord],
    config: DocsyncConfig,
    manifest: Optional[Manifest],
    health: HealthPanel,
    *,
    now: Optional[datetime] = None,
) -> str:
    """Render the whole dashboard as one self-contained HTML string.

    Every dynamic value is `html.escape`d. `now` is injectable so the budget
    projection and "generated at" stamp are deterministic in tests.
    """
    now = now or datetime.now(timezone.utc)
    stats = aggregate(runs)
    budget = budget_status(runs, config, now=now)

    parts: list[str] = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width, initial-scale=1'>",
        "<title>docsync dashboard</title><style>",
        _CSS,
        "</style></head><body><div class=wrap>",
        "<h1>docsync dashboard</h1>",
        f"<p class=sub>Generated {escape(_fmt_ts(now.isoformat()))} · "
        f"{stats.total_runs} run(s) recorded</p>",
    ]

    if not runs:
        parts.append(
            "<div class=panel>No runs recorded yet. Run <code>docsync run</code> or "
            "<code>docsync bootstrap</code> to populate <code>.docsync/state/runs.jsonl</code>, "
            "then re-open this dashboard.</div>"
        )
        parts.append(_health_html(health))
        parts.append("</div></body></html>")
        return "".join(parts)

    parts.append(_overview_html(stats))
    parts.append("<h2>Cost &amp; budget</h2>")
    parts.append(_budget_html(budget))
    parts.append("<h2>Activity</h2>")
    parts.append(_timeline_html(runs))
    parts.append("<h2>Cost breakdown</h2>")
    parts.append(_breakdown_html(runs))
    parts.append("<h2>Latest doc changes</h2>")
    parts.append(_changes_html(runs))
    parts.append("<h2>Health</h2>")
    parts.append(_health_html(health))
    parts.append("</div></body></html>")
    return "".join(parts)


def _overview_html(s: DashboardStats) -> str:
    def card(n: str, label: str) -> str:
        return f"<div class=card><div class=n>{escape(n)}</div><div class=l>{escape(label)}</div></div>"

    drop_card = (
        card(f"{s.drop_rate * 100:.0f}%", "edit-drop rate")
        if s.pages_attempted
        else card("—", "edit-drop rate")
    )
    cards = [
        card(str(s.total_runs), "runs"),
        card(str(s.pages_updated), "pages updated"),
        card(str(s.pages_authored), "pages authored"),
        card(f"${s.total_cost_usd:.2f}", "est. total cost"),
        card(f"{s.avg_cache_hit_rate * 100:.0f}%", "avg cache hit"),
        card(f"{s.success_rate * 100:.0f}%", "success rate"),
        drop_card,
    ]
    html = "<div class=cards>" + "".join(cards) + "</div>"
    if s.pages_dropped:
        reasons = ", ".join(
            f"{escape(reason)} {n}" for reason, n in sorted(s.drops_by_reason.items())
        )
        html += (
            f"<p class=sub style='margin:8px 0 0'>{s.pages_dropped} of {s.pages_attempted} "
            f"attempted page(s) dropped — {escape(reasons)}</p>"
        )
    return html


def _budget_html(b: BudgetStatus) -> str:
    if b.budget_usd:
        pct = (b.month_spend / b.budget_usd) * 100 if b.budget_usd else 0
        cls = "over" if b.over_budget else "ok"
        proj = f"${b.projected_usd:.2f}" if b.projected_usd is not None else "—"
        warn = " ⚠ trending over budget" if b.over_budget else ""
        banner = (
            f"<div class='banner {cls}'>"
            f"<div class=big>${b.month_spend:.2f} / ${b.budget_usd:.2f} this month "
            f"({pct:.0f}%){escape(warn)}</div>"
            f"<div class=muted>projected month-end: {escape(proj)} "
            f"· {b.fraction_elapsed * 100:.0f}% of month elapsed</div></div>"
        )
    else:
        banner = (
            f"<div class='banner ok'><div class=big>${b.month_spend:.2f} this month</div>"
            f"<div class=muted>Set <code>monthly_budget_usd</code> in "
            f"<code>.docsync/config.yml</code> to track a budget and projection.</div></div>"
        )
    return banner


def _timeline_html(runs: list[RunRecord]) -> str:
    rows = []
    for r in sorted(runs, key=lambda r: r.timestamp, reverse=True):
        badge = "badge bootstrap" if r.command == "bootstrap" else "badge"
        if r.command == "bootstrap":
            produced = f"{r.counts.get('authored', 0)}/{r.counts.get('planned', 0)} authored"
        else:
            produced = f"{r.counts.get('updated', 0)}/{r.counts.get('impacted', 0)} updated"
        sha = ""
        if r.base_sha and r.head_sha:
            sha = f"<code>{escape(r.base_sha[:7])}..{escape(r.head_sha[:7])}</code>"
        cost = f"${r.usage.cost_usd:.3f}" if r.usage else "—"
        rows.append(
            "<tr>"
            f"<td>{escape(_fmt_ts(r.timestamp))}</td>"
            f"<td><span class='{badge}'>{escape(r.command)}</span></td>"
            f"<td>{escape(r.repo)} {sha}</td>"
            f"<td>{_pr_link(r)}</td>"
            f"<td>{escape(produced)}</td>"
            f"<td class=num>{escape(cost)}</td>"
            f"<td class='st-{escape(r.status)}'>{escape(r.status)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>When</th><th>Cmd</th><th>Repo</th><th>PR</th>"
        "<th>Produced</th><th class=num>Cost</th><th>Status</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _breakdown_html(runs: list[RunRecord]) -> str:
    curve = _cost_curve_svg(cost_over_time(runs))
    rows = []
    for m in model_stage_breakdown(runs):
        label = f"{m.model}/{m.stage}" if m.stage else m.model
        rows.append(
            "<tr>"
            f"<td><code>{escape(label)}</code></td>"
            f"<td class=num>{m.calls}</td>"
            f"<td class=num>{m.tokens:,}</td>"
            f"<td class=num>${m.cost_usd:.4f}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Model / stage</th><th class=num>Calls</th>"
        "<th class=num>Tokens</th><th class=num>Cost</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )
    return curve + table


def _cost_curve_svg(series: list[tuple[str, float]]) -> str:
    """Hand-rolled cumulative-cost line (inline SVG, no chart lib)."""
    if not series:
        return ""
    w, h, pad = 1040, 80, 4
    ys = [v for _, v in series]
    hi = max(ys) or 1.0
    n = len(series)
    if n == 1:
        pts = f"{pad},{h - pad} {w - pad},{h - pad - (ys[0] / hi) * (h - 2 * pad)}"
    else:
        step = (w - 2 * pad) / (n - 1)
        pts = " ".join(
            f"{pad + i * step:.1f},{h - pad - (v / hi) * (h - 2 * pad):.1f}"
            for i, (_, v) in enumerate(series)
        )
    return (
        f"<svg viewBox='0 0 {w} {h}' width=100% height={h} preserveAspectRatio=none "
        f"style='margin-bottom:12px'>"
        f"<polyline fill=none stroke=#58a6ff stroke-width=1.5 points='{pts}'/>"
        f"<text x={pad} y=14 fill=#8b949e font-size=11>cumulative cost · "
        f"max ${hi:.2f}</text></svg>"
    )


def _changes_html(runs: list[RunRecord], *, recent: int = 8) -> str:
    blocks = []
    recent_runs = sorted(runs, key=lambda r: r.timestamp, reverse=True)[:recent]
    for r in recent_runs:
        touched = [p for p in r.pages if p.applied]
        if not touched:
            continue
        head = (
            f"<summary><b>{escape(_fmt_ts(r.timestamp))}</b> · "
            f"{escape(r.repo)} · {len(touched)} page(s) {_pr_link(r)}</summary>"
        )
        items = []
        for p in touched:
            rats = "".join(f"<li>{escape(rat)}</li>" for rat in p.rationales)
            warns = "".join(f"<li class=warn>⚠ {escape(w)}</li>" for w in p.warnings)
            why = f"<ul class=r>{rats}{warns}</ul>" if (rats or warns) else ""
            change = _diff_html(p.diff) if p.diff else ""
            items.append(
                f"<div class=change><code>{escape(p.page_path)}</code>{why}{change}</div>"
            )
        blocks.append(f"<details>{head}{''.join(items)}</details>")
    if not blocks:
        return "<div class=panel>No applied page changes recorded yet.</div>"
    return "".join(blocks)


def _diff_html(diff: str) -> str:
    """Render a unified-diff string as a colored, escaped <pre> block."""
    rows = []
    for ln in diff.split("\n"):
        if ln.startswith(("+++", "---")):
            cls = "fhdr"
        elif ln.startswith("@@"):
            cls = "hunk"
        elif ln.startswith("+"):
            cls = "add"
        elif ln.startswith("-"):
            cls = "del"
        else:
            cls = ""
        rows.append(f"<span class='{cls}'>{escape(ln) or '&nbsp;'}</span>")
    return "<pre class=diff>" + "".join(rows) + "</pre>"


def _health_html(h: HealthPanel) -> str:
    if h.doctor_ok is True:
        doctor = "<span class=st-opened>OK</span>"
    elif h.doctor_ok is False:
        issues = "; ".join(escape(i) for i in h.doctor_issues[:6]) or "issues found"
        doctor = f"<span class=warn>{issues}</span>"
    else:
        doctor = "<span class=muted>not run</span>"
    cursors = (
        "".join(f"<div><code>{escape(k)}</code> → {escape(v[:10])}</div>" for k, v in h.cursors.items())
        or "<span class=muted>none</span>"
    )
    thr = ", ".join(f"{escape(k)}={v:g}" for k, v in h.thresholds.items())
    return (
        "<div class=panel><dl class=kv>"
        f"<dt>edit model</dt><dd><code>{escape(h.edit_model)}</code></dd>"
        f"<dt>judge model</dt><dd><code>{escape(h.judge_model)}</code></dd>"
        f"<dt>docs root</dt><dd><code>{escape(h.docs_root)}</code></dd>"
        f"<dt>thresholds</dt><dd>{escape(thr)}</dd>"
        f"<dt>manifest</dt><dd>{h.manifest_pages} page(s), {h.manifest_anchors} anchor(s)</dd>"
        f"<dt>cursors</dt><dd>{cursors}</dd>"
        f"<dt>doctor</dt><dd>{doctor}</dd>"
        "</dl></div>"
    )


def _pr_link(r: RunRecord) -> str:
    if r.pr_url:
        label = f"#{r.pr_number}" if r.pr_number else "PR"
        return f"<a href='{escape(r.pr_url, quote=True)}'>{escape(label)}</a>"
    if r.pr_number and "/" in r.repo and not Path(r.repo).exists():
        url = f"https://github.com/{r.repo}/pull/{r.pr_number}"
        return f"<a href='{escape(url, quote=True)}'>#{r.pr_number}</a>"
    if r.pr_number:
        return f"#{r.pr_number}"
    return "<span class=muted>—</span>"


def _fmt_ts(iso: str) -> str:
    ts = _parse_ts(iso)
    return ts.strftime("%Y-%m-%d %H:%M") if ts else iso


def _parse_ts(iso: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Local serving (--serve)
# ---------------------------------------------------------------------------


def _handler_for(html: str):
    """Build a one-route handler that serves `html` for any GET (and quiets logging)."""
    body = html.encode("utf-8")

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server's required name
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence per-request stderr noise
            pass

    return _Handler


def serve(html: str, *, port: int = 8765, open_browser: bool = False) -> None:
    """Serve the dashboard on 127.0.0.1:`port` until interrupted. Local use only."""
    import webbrowser

    httpd = HTTPServer(("127.0.0.1", port), _handler_for(html))
    url = f"http://127.0.0.1:{port}/"
    print(f"docsync: dashboard at {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
