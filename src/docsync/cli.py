"""docsync CLI — the engine, dogfooded locally and wrapped by action.yml.

    docsync run   --src-repo ... --base ... --head ... --docs-repo ...
    docsync map   ...     # impact mapping only (no LLM edits) — cheap dry inspection
    docsync index ...     # build/refresh the embeddings index (optional recall-net)
"""

from __future__ import annotations

import os
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


def _build_diff(
    src_repo: str, base: str, head: str, pr_number: Optional[int], pr_title: Optional[str]
):
    """Local checkout if src_repo is an existing path, else a GitHub owner/name."""
    p = Path(src_repo)
    if p.exists() and (p / ".git").exists():
        return diff_mod.diff_local(p, base, head, pr_number=pr_number, pr_title=pr_title)
    return diff_mod.diff_github(src_repo, base, head, pr_number=pr_number, pr_title=pr_title)


def _resolve_diff(src_repo, base, head, pr_number, pr_title, from_event):
    """Pick the diff source: explicit --from-event > explicit flags > $GITHUB_EVENT_PATH.

    In CI the only thing wired up is the GitHub event JSON, so repo/base/head/PR are
    derived from it automatically — no flags needed (the core "triggered by commits"
    goal). Locally you pass --src-repo/--base/--head.
    """
    explicit = bool(src_repo and base and head)
    event_path = from_event or (None if explicit else os.environ.get("GITHUB_EVENT_PATH"))
    if event_path:
        from .events import diff_from_event

        return diff_from_event(event_path)
    if not explicit:
        raise typer.BadParameter(
            "provide --from-event (or run in CI with $GITHUB_EVENT_PATH set), "
            "or all of --src-repo / --base / --head."
        )
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
    check_links: bool = typer.Option(False, help="Run the mintlify broken-link soft gate."),
    self_critique: bool = typer.Option(
        False,
        help="Adversarially re-check each generated edit against the diff (adds a "
        "judge-model call per page) and drop edits not justified by the change.",
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
    report_path: Optional[Path] = typer.Option(None, help="Write the PR-body markdown here."),
    backend: str = typer.Option(
        "api",
        help="LLM backend: 'api' (ANTHROPIC_API_KEY) or 'claude-code' "
        "(dev: reuse the local Claude Code CLI auth, no API key).",
    ),
):
    """Full pipeline: diff -> impact -> edits -> validate -> (PR | patch + report)."""
    from .llm_backends import get_client

    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)
    if max_parallel is not None:
        config.max_parallel_requests = max_parallel

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

    changed = result.changed()
    if not changed:
        typer.echo("docsync: no validated changes; nothing to open.")
        raise typer.Exit(0)

    if dry_run and not open_pr:
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
        typer.echo(f"docsync: PR -> {url}")
    else:
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
    check_links: bool = typer.Option(False, help="Run the mintlify broken-link soft gate."),
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

    config = cfg.load_config(docs_repo)
    if max_parallel is not None:
        config.max_parallel_requests = max_parallel

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

    authored = result.authored()
    if not authored:
        typer.echo("docsync: no validated pages; nothing to write.")
        raise typer.Exit(0)

    if dry_run and not open_pr:
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
        typer.echo(f"docsync: PR -> {url}")
    else:
        patch = pr_mod.write_patch(docs_repo, docs_repo / "docsync-bootstrap.patch")
        if patch:
            typer.echo(f"docsync: patch written to {patch}")
        else:
            typer.echo("docsync: docs repo is not a git repo — skipped patch (pages written).")


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
    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)
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

    config = cfg.load_config(docs_repo)
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
):
    """Scaffold .docsync/{config.yml,manifest.yml,state/cursors.json} in a docs repo.

    First step to adopting docsync: creates a starter config + a commented manifest
    template you then edit to map pages to their source code.
    """
    from .scaffold import init_docs_repo

    created = init_docs_repo(docs_repo, force=force)
    if not created:
        typer.echo("docsync: .docsync/ already present — nothing to do (use --force to overwrite).")
        return
    for p in created:
        typer.echo(f"  created {p}")
    typer.echo(
        f"docsync: scaffolded {len(created)} file(s). "
        "Edit .docsync/manifest.yml to map pages to source code, then run `docsync doctor`."
    )


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
):
    """Score docsync against a labeled golden set: page-level precision/recall/F1.

    `--mode map` is free (anchors only) and measures mapping recall; `--mode full`
    runs the editor and measures edit-stage precision (costs LLM calls).
    """
    from .eval import load_golden, run_eval

    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)
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


if __name__ == "__main__":
    app()
