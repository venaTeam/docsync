"""docsync CLI — the engine, dogfooded locally and wrapped by action.yml.

    docsync run   --src-repo ... --base ... --head ... --docs-repo ...
    docsync map   ...     # impact mapping only (no LLM edits) — cheap dry inspection
    docsync index ...     # build/refresh the embeddings index (optional recall-net)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from . import bootstrap as bootstrap_mod
from . import config as cfg
from . import diff as diff_mod
from . import pipeline as pipeline_mod
from . import pr as pr_mod
from . import report as report_mod
from .impact import find_anchor_candidates, find_embedding_candidates

app = typer.Typer(add_completion=False, help="Keep documentation in sync with code changes.")


_THOROUGHNESS_LEVELS = ("light", "medium", "high")


def _apply_thoroughness(config, value: Optional[str]) -> None:
    """Override `config.thoroughness` from a CLI flag, validating the level."""
    if value is None:
        return
    if value not in _THOROUGHNESS_LEVELS:
        raise typer.BadParameter(
            f"--thoroughness must be one of {', '.join(_THOROUGHNESS_LEVELS)}"
        )
    config.thoroughness = value


def _load_config(docs_repo: Path):
    """Load config, exiting with a friendly framed message on an invalid config.yml."""
    try:
        return cfg.load_config(docs_repo)
    except cfg.ConfigError as exc:
        typer.echo(f"docsync: {exc}")
        raise typer.Exit(2) from exc


def _load_manifest_or_hint(docs_repo: Path):
    """Load the manifest, exiting with a 'run `docsync init`' hint when it's missing."""
    try:
        return cfg.load_manifest(docs_repo)
    except FileNotFoundError as exc:
        typer.echo(f"docsync: {exc}\nRun `docsync init` first.")
        raise typer.Exit(2) from exc


def _build_diff(
    src_repo: str, base: str, head: str, pr_number: Optional[int], pr_title: Optional[str]
):
    """Local checkout if src_repo is an existing path, else a GitHub owner/name."""
    p = Path(src_repo)
    if p.exists() and (p / ".git").exists():
        return diff_mod.diff_local(p, base, head, pr_number=pr_number, pr_title=pr_title)
    return diff_mod.diff_github(src_repo, base, head, pr_number=pr_number, pr_title=pr_title)


def _resolve_diff(src_repo, base, head, pr_number, pr_title, from_event):
    """Pick the diff source: explicit --from-event > explicit flags > CI auto-detect.

    In CI repo/base/head/PR are derived automatically — no flags needed (the core
    "triggered by commits" goal). The platform is auto-detected: GitLab exposes
    ``CI_*`` env vars (signalled by ``GITLAB_CI``), GitHub writes an event JSON at
    ``$GITHUB_EVENT_PATH``. Locally you pass --src-repo/--base/--head.
    """
    explicit = bool(src_repo and base and head)
    if from_event:
        from .events import diff_from_event

        return diff_from_event(from_event)
    if not explicit:
        from .events import diff_from_ci

        try:
            return diff_from_ci()
        except ValueError as exc:
            raise typer.BadParameter(
                "provide --from-event (or run in CI with $GITLAB_CI / "
                "$GITHUB_EVENT_PATH set), or all of --src-repo / --base / --head."
            ) from exc
    return _build_diff(src_repo, base, head, pr_number, pr_title)


@app.command()
def run(
    docs_repo: Path = typer.Option(..., help="Path to the docs repo checkout."),
    src_repo: Optional[str] = typer.Option(
        None, help="Service repo: local path or GitHub owner/name. Omit when using --from-event."
    ),
    base: Optional[str] = typer.Option(None, help="Base ref/sha (before). Omit with --from-event."),
    head: Optional[str] = typer.Option(None, help="Head ref/sha (after). Omit with --from-event."),
    from_event: Optional[Path] = typer.Option(
        None,
        help="GitHub event JSON (the CI $GITHUB_EVENT_PATH): derive repo/base/head/PR "
        "automatically. Auto-detected from $GITHUB_EVENT_PATH when no flags are given.",
    ),
    pr_number: Optional[int] = typer.Option(None),
    pr_title: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(True, help="Compute + report only; do not write or open a PR."),
    open_pr: bool = typer.Option(False, help="Branch, commit, push, and open a docs PR."),
    use_embeddings: bool = typer.Option(
        True,
        help="Embeddings recall-net: also surface drift on pages the manifest doesn't "
        "anchor. Degrades to anchors-only if the embeddings extra isn't installed.",
    ),
    check_links: bool = typer.Option(
        False, help="Run the active adapter's broken-link soft gate (no-op for adapters "
        "without a link checker, e.g. plain markdown)."
    ),
    self_critique: Optional[bool] = typer.Option(
        None,
        "--self-critique/--no-self-critique",
        help="Adversarially re-check each generated edit against the diff (adds a "
        "judge-model call per page) and drop edits not justified by the change. On by "
        "default; --no-self-critique disables it. Overrides config.self_critique.",
    ),
    polish: Optional[bool] = typer.Option(
        None,
        "--polish/--no-polish",
        help="Readability pass: after each edit, run a fact-frozen pass that revises the "
        "page for a leading summary + scannable structure (adds an edit-model call and a "
        "larger diff). Overrides config.readability_pass.",
    ),
    min_confidence: Optional[float] = typer.Option(
        None,
        help="Skip the edit stage for pages below this impact confidence (0-1). "
        "Overrides config.min_edit_confidence; use for a conservative first rollout.",
    ),
    max_pages: Optional[int] = typer.Option(
        None,
        help="Cap pages sent to the edit stage (highest-confidence first; the rest "
        "are reported, not edited). Overrides config.max_pages_per_run.",
    ),
    max_parallel: Optional[int] = typer.Option(
        None,
        help="Max concurrent LLM requests for the judge + edit stages. "
        "Overrides config.max_parallel_requests.",
    ),
    preflight: bool = typer.Option(
        True,
        help="Pre-flight the manifest (doctor) and abort before any LLM spend if it "
        "references doc pages that don't exist. --no-preflight to bypass.",
    ),
    thoroughness: Optional[str] = typer.Option(
        None,
        help="Generation thoroughness: light | medium | high. Controls edit depth and "
        "the diff-size budget. Overrides config.thoroughness.",
    ),
    report_path: Optional[Path] = typer.Option(None, help="Write the PR-body markdown here."),
    backend: str = typer.Option(
        "api",
        help="LLM backend: 'api' (ANTHROPIC_API_KEY) or 'claude-code' "
        "(dev: reuse the local Claude Code CLI auth, no API key).",
    ),
):
    """Full pipeline: diff -> impact -> edits -> validate -> (PR | patch + report)."""
    from .llm_backends import get_client

    config = _load_config(docs_repo)
    manifest = _load_manifest_or_hint(docs_repo)
    if max_parallel is not None:
        config.max_parallel_requests = max_parallel
    if polish is not None:
        config.readability_pass = polish
    _apply_thoroughness(config, thoroughness)

    # Mono-repo convenience: when docs and code share one checkout, default the source
    # to the docs repo so `docsync run --docs-repo . --base X --head Y` needs no
    # --src-repo. Only for an explicit local run (not CI / --from-event, which carry
    # their own repo), and only when the mode isn't pinned to a separate-repo topology.
    if src_repo is None and base and head and not from_event and config.repo_mode in (
        "auto",
        "mono",
    ):
        src_repo = str(docs_repo)

    # Cursor-aware base: a manual run with a repo + head but no base diffs from the
    # last sync point (the stored cursor) instead of erroring on a missing --base.
    if src_repo and head and not base and not from_event:
        base = _base_or_cursor(docs_repo, src_repo)

    # Pre-flight gate: a manifest page that no longer exists silently disables its
    # anchor (a missed doc update reads as "no drift"). Catch it before any LLM spend.
    if preflight:
        from .scaffold import doctor as run_doctor

        report = run_doctor(docs_repo, {})
        if report.missing_pages:
            typer.echo("docsync: preflight failed — manifest references missing page(s):")
            for p in report.missing_pages:
                typer.echo(f"  ✗ {p}")
            typer.echo(
                "Fix .docsync/manifest.yml (or restore the page); "
                "rerun with --no-preflight to bypass."
            )
            raise typer.Exit(2)

    client = get_client(backend)

    diff = _resolve_diff(src_repo, base, head, pr_number, pr_title, from_event)

    # In a mono repo the diff also carries the docs subtree; filter it so a merged doc
    # change doesn't map onto itself and drive further doc edits.
    repo_mode = cfg.resolve_repo_mode(config, docs_repo, diff.repo, manifest)
    if repo_mode == "mono":
        from .impact import filter_docs_paths

        diff = filter_docs_paths(diff, config.docs_root)

    if cfg.already_processed(docs_repo, diff.repo, diff.head_sha):
        typer.echo(f"docsync: {diff.repo}@{diff.head_sha[:8]} already processed — skipping.")
        raise typer.Exit(0)

    # Snapshot originals for the diff preview before we overwrite anything.
    docs_root = docs_repo / config.docs_root
    result = pipeline_mod.run(
        diff, docs_repo, config, manifest,
        use_embeddings=use_embeddings, check_links=check_links,
        self_critique=self_critique, min_confidence=min_confidence,
        max_pages=max_pages, client=client,
    )
    originals = {
        o.page_path: (docs_root / o.page_path).read_text(encoding="utf-8")
        for o in result.changed()
    }

    typer.echo(report_mod.console_summary(result))
    body = report_mod.pr_body(result, original_texts=originals)
    if report_path:
        report_path.write_text(body, encoding="utf-8")
        typer.echo(f"docsync: report written to {report_path}")

    from . import history

    changed = result.changed()
    if not changed:
        history.record_run(docs_repo, result, command="run", status="no_change")
        typer.echo("docsync: no validated changes; nothing to open.")
        raise typer.Exit(0)

    if dry_run and not open_pr:
        history.record_run(docs_repo, result, command="run", status="dry_run", originals=originals)
        typer.echo("docsync: dry run — not writing changes. Use --no-dry-run / --open-pr to apply.")
        raise typer.Exit(0)

    written = pipeline_mod.write_changes(result, docs_repo, config)
    cfg.advance_cursor(docs_repo, diff.repo, diff.head_sha)
    typer.echo(f"docsync: wrote {len(written)} page(s).")

    if open_pr:
        title = f"docs: sync with {diff.repo}" + (f" #{diff.pr_number}" if diff.pr_number else "")
        url = pr_mod.open_pr(
            docs_repo,
            branch=pr_mod.branch_name(diff.repo, diff.head_sha),
            title=title,
            body=body,
            paths=written,
            reviewers=config.reviewers,
            labels=config.pr_labels,
        )
        history.record_run(
            docs_repo, result, command="run", status="opened", pr_url=url, originals=originals
        )
        typer.echo(f"docsync: PR -> {url}")
    else:
        history.record_run(docs_repo, result, command="run", status="patched", originals=originals)
        patch = pr_mod.write_patch(docs_repo, docs_repo / "docsync.patch")
        if patch:
            typer.echo(f"docsync: patch written to {patch}")
        else:
            typer.echo("docsync: docs repo is not a git repo — skipped patch (changes written).")


def _format_symbol_list(symbols: list[str], *, limit: int = 12) -> str:
    """A compact, readable symbol line: first `limit` names + a (+N more) tail.

    A real diff can touch hundreds of symbols; dumping them all makes `map` output
    unreadable (and pollutes any captured log). Show the head and summarize the rest.
    """
    if not symbols:
        return "—"
    shown = symbols[:limit]
    extra = len(symbols) - len(shown)
    tail = f", (+{extra} more)" if extra > 0 else ""
    return ", ".join(shown) + tail


def _base_or_cursor(docs_repo: Path, src_repo: str) -> str:
    """Resolve a base ref from the stored cursor for `src_repo` when --base is omitted.

    The cursor records the last head_sha docsync processed for a repo, so it is the
    natural "since last sync" base for a manual run. Raises BadParameter when there's
    no cursor for the repo yet (the first run must pass --base explicitly).
    """
    from .impact import _repo_key

    cursors = cfg.load_cursors(docs_repo)
    key = _repo_key(src_repo)
    for repo, sha in cursors.items():
        if _repo_key(repo) == key:
            return sha
    raise typer.BadParameter(
        f"no --base given and no stored cursor for {src_repo!r}; pass --base explicitly."
    )


def _parse_repo_spec(item: str) -> tuple[str, Path]:
    """Parse a `--src-repo` value: `name=path` or bare `path` (name = dir basename)."""
    if "=" in item:
        name, _, path = item.partition("=")
        name, path = name.strip(), path.strip()
    else:
        path = item.strip()
        name = Path(path).resolve().name
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"--src-repo path does not exist: {path}")
    return name, p


@app.command()
def bootstrap(
    docs_repo: Path = typer.Option(..., help="Path to the docs repo checkout (writes go here)."),
    src_repo: List[str] = typer.Option(
        ...,
        help="Service repo to document (read-only), as 'name=path' or a bare path. "
        "Repeatable — pass several to generate a cross-repo platform site.",
    ),
    plan_only: bool = typer.Option(
        False, help="Plan the site and print the outline only — no authoring (no Opus spend)."
    ),
    dry_run: bool = typer.Option(True, help="Compute + report only; do not write or open a PR."),
    open_pr: bool = typer.Option(False, help="Branch, commit, push, and open a docs PR."),
    max_pages: Optional[int] = typer.Option(
        None, help="Cap authored pages (default: unbounded — author every planned page)."
    ),
    max_parallel: Optional[int] = typer.Option(
        None, help="Max concurrent author requests. Overrides config.max_parallel_requests."
    ),
    force: bool = typer.Option(False, help="Overwrite existing page files (default: skip)."),
    check_links: bool = typer.Option(
        False, help="Run the active adapter's broken-link soft gate (no-op for adapters "
        "without a link checker, e.g. plain markdown)."
    ),
    polish: Optional[bool] = typer.Option(
        None,
        "--polish/--no-polish",
        help="Readability pass: after authoring each page, run a fact-frozen pass that "
        "revises it for a leading summary + scannable structure (adds an edit-model call "
        "per page). Overrides config.readability_pass.",
    ),
    thoroughness: Optional[str] = typer.Option(
        None,
        help="Generation thoroughness: light | medium | high. Controls how many pages "
        "the planner targets and how deep each page goes. Overrides config.thoroughness.",
    ),
    report_path: Optional[Path] = typer.Option(None, help="Write the PR-body markdown here."),
    backend: str = typer.Option("api", help="LLM backend: 'api' or 'claude-code'."),
):
    """Generate a docs SITE from scratch: ingest repos -> plan an IA -> author -> validate.

    Reads each `--src-repo` read-only; plans a sequenced, sectioned site (Getting Started
    -> Concepts -> Architecture -> Reference -> Operations) with narrative + reference
    pages, then writes pages, ordered nav, and manifest anchors into `docs_repo` (point
    this at a shadow/empty scaffold to keep the real docs repo untouched).
    """
    from .llm_backends import get_client

    repos = [_parse_repo_spec(item) for item in src_repo]

    config = _load_config(docs_repo)
    if max_parallel is not None:
        config.max_parallel_requests = max_parallel
    if polish is not None:
        config.readability_pass = polish
    _apply_thoroughness(config, thoroughness)

    client = get_client(backend)
    result = bootstrap_mod.run_bootstrap(
        repos, docs_repo, config,
        max_pages=max_pages, check_links=check_links, plan_only=plan_only, client=client,
    )

    typer.echo(report_mod.bootstrap_console_summary(result))
    body = report_mod.bootstrap_pr_body(result)
    if report_path:
        report_path.write_text(body, encoding="utf-8")
        typer.echo(f"docsync: report written to {report_path}")

    if plan_only:
        typer.echo("docsync: plan-only — no pages authored.")
        raise typer.Exit(0)

    from . import history

    authored = result.authored()
    if not authored:
        history.record_run(docs_repo, result, command="bootstrap", status="no_change")
        typer.echo("docsync: no validated pages; nothing to write.")
        raise typer.Exit(0)

    if dry_run and not open_pr:
        history.record_run(docs_repo, result, command="bootstrap", status="dry_run")
        typer.echo("docsync: dry run — not writing. Use --no-dry-run / --open-pr to apply.")
        raise typer.Exit(0)

    written = bootstrap_mod.write_bootstrap(result, docs_repo, config, force=force)
    typer.echo(f"docsync: wrote {len(written)} path(s) (pages + nav + manifest).")

    if open_pr:
        slug = repos[0][0].split("/")[-1] if repos else "platform"
        url = pr_mod.open_pr(
            docs_repo,
            branch=f"docsync/bootstrap-{slug}",
            title=f"docs: bootstrap documentation for {result.repo}",
            body=body,
            paths=written,
            reviewers=config.reviewers,
            labels=config.pr_labels,
        )
        history.record_run(docs_repo, result, command="bootstrap", status="opened", pr_url=url)
        typer.echo(f"docsync: PR -> {url}")
    else:
        history.record_run(docs_repo, result, command="bootstrap", status="patched")
        patch = pr_mod.write_patch(docs_repo, docs_repo / "docsync-bootstrap.patch")
        if patch:
            typer.echo(f"docsync: patch written to {patch}")
        else:
            typer.echo("docsync: docs repo is not a git repo — skipped patch (pages written).")


@app.command()
def infer(
    docs_repo: Path = typer.Option(..., help="Docs repo with existing pages (writes anchors here)."),
    src_repo: List[str] = typer.Option(
        ...,
        help="Source repo to anchor against (read-only), as 'name=path' or a bare path. "
        "Repeatable — pass several to anchor a multi-repo site.",
    ),
    dry_run: bool = typer.Option(
        True, help="Propose anchors and report only; do not touch the manifest."
    ),
    write: bool = typer.Option(
        False, help="Write inferred anchors into .docsync/manifest.yml (same as --no-dry-run)."
    ),
    max_pages: Optional[int] = typer.Option(
        None, help="Cap pages examined per run (cost control)."
    ),
    max_parallel: Optional[int] = typer.Option(
        None, help="Max concurrent judge requests. Overrides config.max_parallel_requests."
    ),
    backend: str = typer.Option("api", help="LLM backend for the judge: 'api' or 'claude-code'."),
    report_path: Optional[Path] = typer.Option(None, help="Write the console summary here."),
):
    """Infer manifest anchors for an EXISTING docs site (the brownfield onboarding path).

    For each page not yet in the manifest: shortlist the source code it likely documents
    (embeddings), have the judge confirm/sharpen the anchors, validate them against the
    real tree, and merge the survivors into `.docsync/manifest.yml`. Reads each `--src-repo`
    read-only. Needs the `embeddings` extra (`poetry install -E embeddings`).
    """
    from . import infer as infer_mod
    from .llm_backends import get_client

    repos = [_parse_repo_spec(item) for item in src_repo]
    config = _load_config(docs_repo)
    if max_parallel is not None:
        config.max_parallel_requests = max_parallel

    client = get_client(backend)
    try:
        result = infer_mod.run_infer(repos, docs_repo, config, max_pages=max_pages, client=client)
    except ImportError:
        typer.echo(
            "docsync: manifest inference needs the embeddings extra "
            "(install with `poetry install -E embeddings`)."
        )
        raise typer.Exit(0) from None

    summary = report_mod.infer_console_summary(result)
    typer.echo(summary)
    if report_path:
        report_path.write_text(summary + "\n", encoding="utf-8")
        typer.echo(f"docsync: report written to {report_path}")

    if not result.anchored():
        typer.echo("docsync: no anchors inferred; manifest unchanged.")
        raise typer.Exit(0)

    if dry_run and not write:
        typer.echo("docsync: dry run — manifest unchanged. Use --write to apply.")
        raise typer.Exit(0)

    added = infer_mod.write_infer(result, docs_repo, config)
    typer.echo(f"docsync: wrote {len(added)} anchor(s) to .docsync/manifest.yml.")


@app.command()
def dashboard(
    docs_repo: Path = typer.Option(..., help="Docs repo with .docsync/ run history."),
    out: Path = typer.Option(
        Path("docsync-dashboard.html"), help="Write the self-contained HTML dashboard here."
    ),
    serve: bool = typer.Option(
        False, help="Serve the dashboard locally over http.server instead of writing a file."
    ),
    port: int = typer.Option(8765, help="Port for --serve."),
    open_browser: bool = typer.Option(
        False, "--open/--no-open", help="Open the dashboard in a browser (with --serve)."
    ),
    limit: Optional[int] = typer.Option(None, help="Show only the most recent N runs."),
    checkout: Optional[List[str]] = typer.Option(
        None, help="Source checkout 'owner/name=path' (repeatable) for the health panel's doctor."
    ),
):
    """Render a dashboard of docsync activity: stats, runs, cost/budget, changes, health.

    Reads `.docsync/state/runs.jsonl` (populated on each `run`/`bootstrap`). Writes a single
    self-contained HTML file by default; `--serve` hosts it locally (never use in CI).
    """
    from . import dashboard as dash_mod
    from . import history

    config = _load_config(docs_repo)
    try:
        manifest = _load_manifest_or_hint(docs_repo)
    except FileNotFoundError:
        manifest = None

    checkouts: dict[str, Path] = {}
    for item in checkout or []:
        if "=" not in item:
            raise typer.BadParameter(f"--checkout must be 'owner/name=path', got {item!r}")
        repo, _, path = item.partition("=")
        checkouts[repo] = Path(path)

    runs = history.load_runs(docs_repo, limit=limit)
    health = dash_mod.build_health(docs_repo, config, manifest, checkouts)
    html = dash_mod.render_dashboard(runs, config, manifest, health)

    if serve:
        if not runs:
            typer.echo("docsync: no runs recorded yet — serving an empty dashboard.")
        dash_mod.serve(html, port=port, open_browser=open_browser)
    else:
        out.write_text(html, encoding="utf-8")
        typer.echo(f"docsync: dashboard written to {out} ({len(runs)} run(s)).")
        if not runs:
            typer.echo("docsync: no runs recorded yet — run `docsync run`/`bootstrap` first.")


@app.command()
def map(  # noqa: A001 - intentional command name
    src_repo: str = typer.Option(...),
    head: str = typer.Option(...),
    docs_repo: Path = typer.Option(...),
    base: Optional[str] = typer.Option(
        None, help="Base ref (before). Omit to use the stored cursor for this repo."
    ),
    use_embeddings: bool = typer.Option(False),
):
    """Impact mapping only: which pages would be touched, via anchors (+embeddings).

    Cheap inspection — no LLM judge or edits. Prints candidate pages and why.
    """
    config = _load_config(docs_repo)
    manifest = _load_manifest_or_hint(docs_repo)
    base = base or _base_or_cursor(docs_repo, src_repo)
    diff = _build_diff(src_repo, base, head, None, None)
    docs_root = docs_repo / config.docs_root

    anchors = find_anchor_candidates(diff, manifest)
    typer.echo(
        f"Changed: {len(diff.files)} file(s); "
        f"symbols: {_format_symbol_list(diff.all_symbols())}"
    )
    typer.echo(f"Anchor candidates ({len(anchors)}):")
    for c in anchors:
        typer.echo(f"  · {c.page_path}  [score {c.score:.0f}]  {c.reason}")
    if use_embeddings:
        anchored = {c.page_path for c in anchors}
        cache_dir = docs_repo / cfg.DOCSYNC_DIR / "state" / "embeddings"
        # Full-tree recall-net (scan all pages), excluding already-anchored ones.
        emb = [
            c for c in find_embedding_candidates(diff, docs_root, None, config, cache_dir=cache_dir)
            if c.page_path not in anchored
        ]
        typer.echo(f"Embedding candidates ({len(emb)}):")
        for c in emb:
            typer.echo(f"  · {c.page_path}  [sim {c.score:.2f}]  {c.reason}")


@app.command()
def index(
    docs_repo: Path = typer.Option(...),
):
    """Build/refresh + persist the embeddings index (optional recall-net).

    Encodes every page once and caches it under .docsync/state/embeddings, so later
    runs reuse it unless the docs change. If the `embeddings` extra isn't installed,
    reports that and exits 0.
    """
    from . import embeddings as embeddings_mod

    config = _load_config(docs_repo)
    docs_root = docs_repo / config.docs_root
    cache_dir = docs_repo / cfg.DOCSYNC_DIR / "state" / "embeddings"
    try:
        encoder = embeddings_mod.default_encoder(config.embedding_model)
    except ImportError:
        typer.echo("docsync: embeddings extra not installed "
                   "(install with `poetry install -E embeddings`).")
        raise typer.Exit(0) from None

    idx = embeddings_mod.load_or_build(
        docs_root, cache_dir, model_name=config.embedding_model, encoder=encoder
    )
    pages = len(set(idx.page_paths))
    typer.echo(
        f"docsync: embeddings index ready — {len(idx.page_paths)} chunk(s) across "
        f"{pages} page(s), cached at {cache_dir}."
    )


@app.command()
def init(
    docs_repo: Path = typer.Option(Path("."), help="Docs repo to scaffold .docsync/ into."),
    force: bool = typer.Option(False, help="Overwrite existing .docsync files."),
    minimal: bool = typer.Option(
        False,
        help="Zero-config: auto-detect docs_root + adapter and write a minimal config "
        "(no placeholder manifest).",
    ),
    detect: bool = typer.Option(
        True, help="With --minimal, auto-detect docs_root + adapter from the docs config "
        "(docs.json / docusaurus.config.js) or the .mdx/.md tree."
    ),
    infer: bool = typer.Option(
        False,
        help="After scaffolding, infer manifest anchors from --src-repo (implies --minimal).",
    ),
    src_repo: Optional[List[str]] = typer.Option(
        None, help="Source repo(s) for --infer, as 'name=path' or a bare path. Repeatable."
    ),
    backend: str = typer.Option("api", help="LLM backend for --infer: 'api' or 'claude-code'."),
):
    """Scaffold `.docsync/` in a docs repo (first step to adopting docsync).

    Default: a starter config + a commented manifest template you edit to map pages to
    source code. `--minimal` instead auto-detects `docs_root`/adapter and writes a tiny
    config with no placeholder manifest; add `--infer --src-repo name=path` to go straight
    from empty to a populated, doctor-clean manifest in one command.
    """
    from .scaffold import detect_adapter, detect_docs_root, detect_repo_mode, init_docs_repo

    use_minimal = minimal or infer
    created = init_docs_repo(docs_repo, force=force, minimal=use_minimal, detect=detect)
    if not created and not infer:
        typer.echo("docsync: .docsync/ already present — nothing to do (use --force to overwrite).")
        return
    for p in created:
        typer.echo(f"  created {p}")

    if use_minimal:
        root = detect_docs_root(docs_repo) if detect else "."
        adapter = detect_adapter(docs_repo, root)
        repo_mode = detect_repo_mode(docs_repo, root) if detect else "auto"
        typer.echo(
            f"docsync: detected docs_root='{root}'"
            + (f", adapter={adapter}" if adapter else " (adapter unknown)")
            + f", repo_mode={repo_mode}"
        )

    if not infer:
        populate = (
            "docsync infer --src-repo name=path   # auto-propose manifest anchors"
            if use_minimal
            else "edit .docsync/manifest.yml          # map pages to their source code"
        )
        typer.echo(f"\ndocsync: scaffolded {len(created)} file(s). Next steps:")
        typer.echo(f"  1. {populate}")
        typer.echo("  2. docsync doctor                    # check the manifest resolves")
        typer.echo("  3. docsync run --docs-repo . --src-repo … --base … --head …   # dry-run preview")
        typer.echo("  4. add --open-pr to step 3 to open the docs PR")
        typer.echo("\nRun `docsync explain` to see every config option.")
        return

    # --infer: chain straight into inference against the given source checkout(s).
    if not src_repo:
        raise typer.BadParameter("--infer requires at least one --src-repo name=path")
    from . import infer as infer_mod
    from .llm_backends import get_client

    repos = [_parse_repo_spec(item) for item in src_repo]
    config = _load_config(docs_repo)
    client = get_client(backend)
    try:
        result = infer_mod.run_infer(repos, docs_repo, config, client=client)
    except ImportError:
        typer.echo(
            "docsync: manifest inference needs the embeddings extra "
            "(install with `poetry install -E embeddings`)."
        )
        raise typer.Exit(0) from None
    typer.echo(report_mod.infer_console_summary(result))
    added = infer_mod.write_infer(result, docs_repo, config)
    typer.echo(f"docsync: wrote {len(added)} anchor(s) to .docsync/manifest.yml.")


@app.command()
def doctor(
    docs_repo: Path = typer.Option(..., help="Docs repo with a .docsync/manifest.yml."),
    checkout: Optional[List[str]] = typer.Option(
        None,
        help="Source checkout as 'owner/name=path' (repeatable) — needed to validate "
        "globs/symbols against real code.",
    ),
):
    """Validate the manifest: pages exist, globs resolve, symbols present in the checkouts.

    Catches *manifest* drift (a rename/move that silently breaks anchor mapping) before
    it costs you a missed doc update. Exits non-zero if there are hard issues.
    """
    from .scaffold import doctor as run_doctor

    checkouts: dict[str, Path] = {}
    for item in checkout or []:
        if "=" not in item:
            raise typer.BadParameter(f"--checkout must be 'owner/name=path', got {item!r}")
        repo, _, path = item.partition("=")
        checkouts[repo] = Path(path)

    try:
        report = run_doctor(docs_repo, checkouts)
    except FileNotFoundError as exc:
        typer.echo(f"docsync: {exc}\nRun `docsync init` first.")
        raise typer.Exit(2) from exc

    for pg in report.missing_pages:
        typer.echo(f"  ✗ missing page: {pg}")
    for d in report.dead_globs:
        typer.echo(f"  ✗ dead glob: {d.page} -> {d.repo} :: {d.glob}")
    for r in report.unmapped_repos:
        typer.echo(f"  ? no checkout for repo: {r} (pass --checkout {r}=/path to validate it)")
    for m in report.missing_symbols:
        typer.echo(f"  ! symbol not found: {m.page} -> {m.repo} :: {m.symbol}")

    if report.ok:
        typer.echo("docsync: manifest OK.")
    else:
        typer.echo("docsync: manifest has issues (see above).")
        raise typer.Exit(1)


@app.command()
def eval(  # noqa: A001 - intentional command name
    docs_repo: Path = typer.Option(..., help="Docs repo with .docsync/manifest.yml."),
    golden: Path = typer.Option(..., help="Golden-set JSON of labeled PRs (repo/base/head/expected_pages)."),
    mode: str = typer.Option("map", help="'map' (free anchor mapping) or 'full' (LLM edit pipeline)."),
    use_embeddings: bool = typer.Option(False, help="Also use the embeddings recall-net when mapping."),
    backend: str = typer.Option("api", help="LLM backend for --mode full: 'api' or 'claude-code'."),
    json_out: Optional[Path] = typer.Option(None, help="Write the full EvalReport JSON here."),
    min_recall: Optional[float] = typer.Option(
        None, help="Fail (exit 1) if recall falls below this floor — the regression gate for CI."
    ),
    min_precision: Optional[float] = typer.Option(
        None, help="Fail (exit 1) if precision falls below this floor. In --mode map, precision is "
        "intentionally low on true-negative-at-edit cases; gate recall there, not precision.",
    ),
):
    """Score docsync against a labeled golden set: page-level precision/recall/F1.

    `--mode map` is free (anchors only) and measures mapping recall; `--mode full`
    runs the editor and measures edit-stage precision (costs LLM calls).

    Pass `--min-recall` (and/or `--min-precision`) to turn the score into a CI gate:
    the command exits non-zero when the measured score drops below the floor, so a
    judge/anchor/prompt regression fails the build instead of shipping silently.
    """
    from .eval import load_golden, run_eval, threshold_breaches

    config = _load_config(docs_repo)
    manifest = _load_manifest_or_hint(docs_repo)
    cases = load_golden(golden)

    client = None
    if mode == "full":
        from .llm_backends import get_client

        client = get_client(backend)

    report = run_eval(
        cases, docs_repo, config, manifest,
        mode=mode, use_embeddings=use_embeddings, client=client,
    )
    for c in report.cases:
        mark = "✓" if (c.fp == 0 and c.fn == 0) else "✗"
        typer.echo(f"  {mark} {c.label}")
        typer.echo(f"      expected={c.expected}  actual={c.actual}  tp={c.tp} fp={c.fp} fn={c.fn}")
    typer.echo(
        f"docsync eval [{mode}]: P={report.precision:.2f} R={report.recall:.2f} "
        f"F1={report.f1:.2f} over {report.n_cases} case(s)"
    )
    if report.usage and report.usage.calls:
        from .cost import render_usage_console

        typer.echo(render_usage_console(report.usage))
    if json_out:
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"docsync: eval report written to {json_out}")

    # Regression gate: fail the build when a floor is set and the score is below it.
    breaches = threshold_breaches(report, min_recall=min_recall, min_precision=min_precision)
    if breaches:
        typer.echo("docsync eval: BELOW THRESHOLD — " + "; ".join(breaches))
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# explain — the canonical config/manifest reference (replaces "read models.py")
# ---------------------------------------------------------------------------


def _type_name(annotation) -> str:
    """A readable type name for a pydantic field annotation."""
    import re

    s = str(annotation).replace("typing.", "")
    m = re.fullmatch(r"<class '([\w.]+)'>", s)
    if m:
        return m.group(1).rsplit(".", 1)[-1]
    # Strip dotted module qualifiers inside generics, e.g. list[a.b.C] -> list[C].
    return re.sub(r"[a-zA-Z_][\w]*\.", "", s)


def _field_default(field) -> str:
    """A display string for a field's default (or '(required)')."""
    if field.is_required():
        return "(required)"
    if field.default_factory is not None:
        return repr(field.default_factory())
    return repr(field.default)


def _render_fields(model) -> list[str]:
    """One formatted block per field of a pydantic *model*."""
    lines: list[str] = []
    for name, field in model.model_fields.items():
        lines.append(
            f"  {name}  ({_type_name(field.annotation)}, default: {_field_default(field)})"
        )
        if field.description:
            lines.append(f"      {field.description}")
    return lines


@app.command()
def explain(
    field: Optional[str] = typer.Argument(
        None,
        help="A config field to explain, or 'manifest' for the manifest schema. "
        "Omit to list every config field.",
    ),
):
    """Explain docsync configuration: every `.docsync/config.yml` field and its default.

    `docsync explain` lists all config fields; `docsync explain <field>` shows one;
    `docsync explain manifest` shows the manifest page/source schema.
    """
    from .models import DocsyncConfig, ManifestPage, ManifestSource

    if field == "manifest":
        typer.echo("docsync manifest schema (.docsync/manifest.yml) — pages: [ManifestPage]\n")
        typer.echo("ManifestPage:")
        typer.echo("\n".join(_render_fields(ManifestPage)))
        typer.echo("\nManifestSource (each entry under a page's `sources`):")
        typer.echo("\n".join(_render_fields(ManifestSource)))
        return

    if field:
        if field not in DocsyncConfig.model_fields:
            valid = ", ".join(DocsyncConfig.model_fields)
            raise typer.BadParameter(f"unknown config field {field!r}. Valid fields: {valid}")
        info = DocsyncConfig.model_fields[field]
        typer.echo(f"{field}  ({_type_name(info.annotation)}, default: {_field_default(info)})")
        if info.description:
            typer.echo(f"  {info.description}")
        return

    typer.echo("docsync config (.docsync/config.yml) — all fields optional:\n")
    typer.echo("\n".join(_render_fields(DocsyncConfig)))
    typer.echo("\nRun `docsync explain manifest` for the manifest schema.")


if __name__ == "__main__":
    app()
