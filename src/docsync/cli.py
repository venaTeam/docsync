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
    use_embeddings: bool = typer.Option(False, help="Enable the embeddings recall-net."),
    check_links: bool = typer.Option(False, help="Run the mintlify broken-link soft gate."),
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
    client = get_client(backend)

    diff = _resolve_diff(src_repo, base, head, pr_number, pr_title, from_event)

    if cfg.already_processed(docs_repo, diff.repo, diff.head_sha):
        typer.echo(f"docsync: {diff.repo}@{diff.head_sha[:8]} already processed — skipping.")
        raise typer.Exit(0)

    # Snapshot originals for the diff preview before we overwrite anything.
    docs_root = docs_repo / config.docs_root
    result = pipeline_mod.run(
        diff, docs_repo, config, manifest,
        use_embeddings=use_embeddings, check_links=check_links, client=client,
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
        )
        typer.echo(f"docsync: PR -> {url}")
    else:
        patch = pr_mod.write_patch(docs_repo, docs_repo / "docsync.patch")
        typer.echo(f"docsync: patch written to {patch}")


@app.command()
def map(  # noqa: A001 - intentional command name
    src_repo: str = typer.Option(...),
    base: str = typer.Option(...),
    head: str = typer.Option(...),
    docs_repo: Path = typer.Option(...),
    use_embeddings: bool = typer.Option(False),
):
    """Impact mapping only: which pages would be touched, via anchors (+embeddings).

    Cheap inspection — no LLM judge or edits. Prints candidate pages and why.
    """
    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)
    diff = _build_diff(src_repo, base, head, None, None)
    docs_root = docs_repo / config.docs_root

    anchors = find_anchor_candidates(diff, manifest)
    typer.echo(f"Changed: {len(diff.files)} file(s); symbols: {', '.join(diff.all_symbols()) or '—'}")
    typer.echo(f"Anchor candidates ({len(anchors)}):")
    for c in anchors:
        typer.echo(f"  · {c.page_path}  [score {c.score:.0f}]  {c.reason}")
    if use_embeddings:
        emb = find_embedding_candidates(diff, docs_root, manifest.pages, config)
        typer.echo(f"Embedding candidates ({len(emb)}):")
        for c in emb:
            typer.echo(f"  · {c.page_path}  [sim {c.score:.2f}]  {c.reason}")


@app.command()
def index(
    docs_repo: Path = typer.Option(...),
):
    """Build/refresh the embeddings index (optional recall-net).

    No-op-friendly: if sentence-transformers isn't installed, reports that and exits 0.
    """
    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)
    docs_root = docs_repo / config.docs_root
    # find_embedding_candidates builds the index on demand; calling it with an empty
    # diff just exercises/install-checks the path.
    from .models import CodeDiff

    empty = CodeDiff(repo="_", base_sha="0", head_sha="0")
    out = find_embedding_candidates(empty, docs_root, manifest.pages, config)
    if not out:
        typer.echo("docsync: embeddings unavailable or no candidates "
                   "(install with `poetry install -E embeddings`).")
    else:
        typer.echo("docsync: embeddings index ready.")


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


if __name__ == "__main__":
    app()
