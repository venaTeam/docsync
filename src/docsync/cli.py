"""docsync CLI — the engine, dogfooded locally and wrapped by action.yml.

    docsync run   --src-repo ... --base ... --head ... --docs-repo ...
    docsync map   ...     # impact mapping only (no LLM edits) — cheap dry inspection
    docsync index ...     # build/refresh the embeddings index (optional recall-net)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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


@app.command()
def run(
    src_repo: str = typer.Option(..., help="Service repo: local path or GitHub owner/name."),
    base: str = typer.Option(..., help="Base ref/sha (before)."),
    head: str = typer.Option(..., help="Head ref/sha (after)."),
    docs_repo: Path = typer.Option(..., help="Path to the docs repo checkout."),
    pr_number: Optional[int] = typer.Option(None),
    pr_title: Optional[str] = typer.Option(None),
    dry_run: bool = typer.Option(True, help="Compute + report only; do not write or open a PR."),
    open_pr: bool = typer.Option(False, help="Branch, commit, push, and open a docs PR."),
    use_embeddings: bool = typer.Option(False, help="Enable the embeddings recall-net."),
    check_links: bool = typer.Option(False, help="Run the mintlify broken-link soft gate."),
    report_path: Optional[Path] = typer.Option(None, help="Write the PR-body markdown here."),
):
    """Full pipeline: diff -> impact -> edits -> validate -> (PR | patch + report)."""
    config = cfg.load_config(docs_repo)
    manifest = cfg.load_manifest(docs_repo)

    diff = _build_diff(src_repo, base, head, pr_number, pr_title)

    if cfg.already_processed(docs_repo, diff.repo, diff.head_sha):
        typer.echo(f"docsync: {diff.repo}@{diff.head_sha[:8]} already processed — skipping.")
        raise typer.Exit(0)

    # Snapshot originals for the diff preview before we overwrite anything.
    docs_root = docs_repo / config.docs_root
    result = pipeline_mod.run(
        diff, docs_repo, config, manifest,
        use_embeddings=use_embeddings, check_links=check_links,
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


if __name__ == "__main__":
    app()
