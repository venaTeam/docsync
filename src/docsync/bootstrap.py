"""From-scratch doc generation — `docsync bootstrap`.

Where the update pipeline (pipeline.py) edits *existing* pages from a diff, this
authors *new* pages from a whole-repo snapshot:

    B1 ingest  → RepoDigest (ingest.walk_repo, read-only)
    B2 plan    → DocPlan    (one Haiku call, then deterministic collision dedupe)
    B3 author  → full MDX   (Opus per page, in parallel, metered)
    B4 validate→ validate_new_page (absolute gates — no original to diff)
    B5 emit    → write files + register nav + merge manifest anchors  (write_bootstrap)
    B6 PR      → pr.open_pr (the CLI wires this)

Every LLM call goes through the injectable `client` (wrapped once in a
`MeteredClient`), so cost lands on `BootstrapResult.usage`, attributed per stage.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import cost
from . import ingest as ingest_mod
from .config import MANIFEST_FILE, merge_manifest_pages
from .cost import MeteredClient, UsageMeter
from .models import (
    AuthoredPage,
    BootstrapResult,
    DocPlan,
    DocsyncConfig,
    ManifestPage,
    ManifestSource,
    PageOutcome,
    PlannedPage,
    RepoDigest,
)
from .validate import get_adapter, validate_new_page

DEFAULT_NAV_GROUP = "Reference (docsync)"

_PLAN_MAX_TOKENS = 4_000
_AUTHOR_MAX_TOKENS = 8_000
# Cap the digest handed to the planner so a huge repo can't overflow its context.
_DIGEST_MAX_CHARS = 30_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_page_path(page_path: str) -> str:
    """A clean docs-root-relative `.mdx` path (no leading slash, has an extension)."""
    p = page_path.strip().lstrip("/")
    if not (p.lower().endswith(".mdx") or p.lower().endswith(".md")):
        p = f"{p}.mdx"
    return p


def _route_of(page_path: str) -> str:
    """The extensionless nav route for a page path (mirrors the adapter's ref form)."""
    if page_path.lower().endswith(".mdx"):
        return page_path[:-4]
    if page_path.lower().endswith(".md"):
        return page_path[:-3]
    return page_path


def _existing_page_paths(docs_root: Path) -> set[str]:
    """Every .mdx/.md already on disk under docs_root (relative, posix)."""
    out: set[str] = set()
    for ext in ("*.mdx", "*.md"):
        for fp in docs_root.rglob(ext):
            if ".docsync" in fp.parts:
                continue
            out.add(fp.relative_to(docs_root).as_posix())
    return out


def render_digest(digest: RepoDigest, *, max_chars: int = _DIGEST_MAX_CHARS) -> str:
    """A compact, deterministic listing of files + their top-level symbols."""
    lines: list[str] = []
    total = 0
    for unit in digest.units:
        syms = ", ".join(unit.symbols[:15])
        line = f"- {unit.path}" + (f"  ::  {syms}" if syms else "")
        if total + len(line) + 1 > max_chars:
            lines.append(f"… ({len(digest.units)} files total; list truncated)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# B2 — plan
# ---------------------------------------------------------------------------


def build_plan_prompt(
    digest: RepoDigest,
    *,
    existing_routes: list[str],
    existing_pages: set[str],
    group: str,
) -> tuple[str, str]:
    """Return (system, user) for the planner call."""
    system = "\n".join(
        [
            "You are a senior technical writer planning developer documentation for a "
            "code repository that currently has little or no docs.",
            "Propose a focused set of NEW reference pages, each covering a coherent "
            "slice of the codebase (a subsystem, a route group, a service module).",
            "Prefer a handful of substantial pages over many thin ones. Every page must "
            "map to real source files from the digest.",
            "For each page set: page_path (a NEW .mdx path under the docs root, kebab-case, "
            "grouped in a sensible folder), a clear title, a one-sentence summary of what "
            "it should cover, the source_paths it documents, and the key symbols it anchors.",
            f"Put every page in the nav group {group!r}.",
            "Do NOT propose a page whose path or route already exists (listed below). "
            "Do NOT duplicate page paths within your plan.",
        ]
    )
    existing_block = (
        "\n".join(f"- {r}" for r in sorted(existing_routes) + sorted(existing_pages))
        or "(none)"
    )
    user = (
        f"# Repository: {digest.repo}\n\n"
        f"# Source digest (path :: top-level symbols)\n\n{render_digest(digest)}\n\n"
        f"# Pages/routes that ALREADY EXIST (do not propose these)\n\n{existing_block}\n\n"
        "Produce a DocPlan: the set of new pages to author."
    )
    return system, user


def plan_docs(
    digest: RepoDigest,
    docs_root: Path,
    config: DocsyncConfig,
    *,
    client,
    group: str = DEFAULT_NAV_GROUP,
    max_pages: int | None = None,
) -> tuple[DocPlan, list[str]]:
    """Ask the judge model for a DocPlan, then deterministically dedupe collisions.

    Returns (plan, skipped) where `skipped` lists planned page paths dropped for
    colliding with an existing page/route or an earlier plan entry. The optional
    `max_pages` cap is applied AFTER dedupe (so collisions don't consume the budget).
    """
    adapter = get_adapter("_.mdx")
    existing_pages = _existing_page_paths(docs_root)
    existing_routes = adapter.nav_routes(docs_root)
    system, user = build_plan_prompt(
        digest, existing_routes=existing_routes, existing_pages=existing_pages, group=group
    )

    with cost.stage("plan"):
        resp = client.messages.parse(
            model=config.models.judge_model,
            max_tokens=_PLAN_MAX_TOKENS,
            system=[{"type": "text", "text": system}],
            messages=[{"role": "user", "content": user}],
            output_format=DocPlan,
        )
    raw_plan: DocPlan = resp.parsed_output

    taken_routes = set(existing_routes)
    kept: list[PlannedPage] = []
    skipped: list[str] = []
    for page in raw_plan.pages:
        path = _normalize_page_path(page.page_path)
        route = _route_of(path)
        if path in existing_pages or route in taken_routes:
            skipped.append(path)
            continue
        page.page_path = path
        if not page.group:
            page.group = group
        kept.append(page)
        existing_pages.add(path)
        taken_routes.add(route)

    if max_pages and max_pages > 0:
        kept = kept[:max_pages]
    return DocPlan(pages=kept), skipped


# ---------------------------------------------------------------------------
# B3 — author
# ---------------------------------------------------------------------------


def build_author_prompt(planned: PlannedPage, excerpts: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (system, user) for authoring one full Mintlify MDX page."""
    system = "\n".join(
        [
            "You author a single, complete Mintlify documentation page in MDX.",
            "Return the ENTIRE file: YAML frontmatter with a non-empty `title` and "
            "`description`, followed by the MDX body.",
            "Use Mintlify components where they help (<CardGroup>/<Card>, <Steps>/<Step>, "
            "<Note>, <Warning>, <Tabs>/<Tab>); keep every component tag balanced and "
            "correctly nested, and keep code fences balanced.",
            "Ground every statement in the provided source code — describe the actual "
            "functions, routes, classes, and behavior. Do not invent APIs.",
            "Do NOT include navigation config or links to pages you can't see; write a "
            "self-contained reference page.",
        ]
    )
    code_blocks = "\n\n".join(
        f"## `{path}`\n\n```\n{text}\n```" for path, text in excerpts if text
    ) or "(no source excerpts available — write from the title and summary)"
    user = (
        f"# Page to write: {planned.page_path}\n\n"
        f"Title: {planned.title}\n"
        f"Intended coverage: {planned.summary or '(use your judgment)'}\n"
        f"Key symbols to document: {', '.join(planned.symbols) or '(none specified)'}\n\n"
        f"# Source code this page documents\n\n{code_blocks}\n\n"
        "Write the complete .mdx file now."
    )
    return system, user


def author_page(
    planned: PlannedPage,
    src_root: str | Path,
    config: DocsyncConfig,
    *,
    client,
) -> str:
    """Generate the full MDX text for one planned page (Opus, structured output)."""
    excerpts = [
        (path, ingest_mod.read_excerpt(src_root, path)) for path in planned.source_paths
    ]
    system, user = build_author_prompt(planned, excerpts)
    with cost.stage("author"):
        resp = client.messages.parse(
            model=config.models.edit_model,
            max_tokens=_AUTHOR_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": config.models.edit_effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=AuthoredPage,
        )
    return resp.parsed_output.content


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_bootstrap(
    src_repo: str | Path,
    docs_repo: Path,
    config: DocsyncConfig,
    *,
    repo: str | None = None,
    group: str = DEFAULT_NAV_GROUP,
    max_pages: int | None = None,
    check_links: bool = False,
    plan_only: bool = False,
    client=None,
    meter: UsageMeter | None = None,
) -> BootstrapResult:
    """Ingest → plan → author → validate. Returns a BootstrapResult (no writes).

    Writing pages / nav / manifest is deferred to `write_bootstrap`, so `--dry-run`
    can report everything without touching disk. All LLM spend is metered onto
    `result.usage`. With `plan_only=True` it stops after planning (no author spend).
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root

    meter = meter or UsageMeter()
    if client is not None:
        client = MeteredClient(client, meter)

    digest = ingest_mod.walk_repo(src_repo, repo=repo)
    result = BootstrapResult(repo=digest.repo)

    plan, skipped = plan_docs(
        digest, docs_root, config, client=client, group=group, max_pages=max_pages
    )
    result.plan = plan
    result.skipped = skipped

    if plan_only:
        result.usage = meter.finalize()
        return result

    adapter = get_adapter("_.mdx")

    def _author_one(planned: PlannedPage) -> PageOutcome:
        """Author + validate one page. Pure w.r.t. shared state; never raises."""
        outcome = PageOutcome(page_path=planned.page_path)
        try:
            text = author_page(planned, digest.root, config, client=client)
        except Exception as exc:  # noqa: BLE001 - one bad page must not abort the pool
            outcome.note = f"author failed: {exc}"
            return outcome
        validation = validate_new_page(
            planned.page_path, text, adapter,
            check_links=check_links, docs_root=docs_root,
        )
        outcome.validation = validation
        if not validation.passed:
            outcome.note = "dropped by validation: " + "; ".join(validation.failures)
            return outcome
        outcome.new_content = text
        outcome.applied = True
        outcome.note = "authored"
        return outcome

    pages = plan.pages
    workers = max(1, min(config.max_parallel_requests, len(pages)))
    if workers <= 1:
        outcomes = [_author_one(p) for p in pages]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(_author_one, pages))

    result.outcomes = outcomes
    result.usage = meter.finalize()
    return result


def write_bootstrap(
    result: BootstrapResult,
    docs_repo: Path,
    config: DocsyncConfig,
    *,
    force: bool = False,
) -> list[str]:
    """Write authored pages, register them in nav, and merge manifest anchors.

    Returns the repo-relative paths touched (pages + modified nav files + manifest),
    ready to hand to `pr.open_pr`. Writes files first, then nav, then manifest.
    Existing page files are left untouched unless `force` (idempotent re-runs).
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root
    adapter = get_adapter("_.mdx")
    by_path = {p.page_path: p for p in result.plan.pages}

    written: list[str] = []
    for outcome in result.authored():
        fp = docs_root / outcome.page_path
        if fp.exists() and not force:
            continue
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(outcome.new_content, encoding="utf-8")
        written.append(outcome.page_path)

    # Register pages in nav, grouped by the nav group each plan entry requested.
    nav_touched: set[str] = set()
    groups: dict[str, list[str]] = {}
    for page_path in written:
        planned = by_path.get(page_path)
        grp = planned.group if planned and planned.group else DEFAULT_NAV_GROUP
        groups.setdefault(grp, []).append(page_path)
    for grp, page_paths in groups.items():
        nav_touched.update(adapter.register_pages_in_nav(docs_root, page_paths, group=grp))

    # Merge manifest anchors so `docsync run` can maintain these pages afterward.
    manifest_pages: list[ManifestPage] = []
    for page_path in written:
        planned = by_path.get(page_path)
        if planned is None:
            continue
        manifest_pages.append(
            ManifestPage(
                path=page_path,
                sources=[
                    ManifestSource(
                        repo=result.repo,
                        globs=list(planned.source_paths),
                        symbols=list(planned.symbols),
                    )
                ],
            )
        )
    added = merge_manifest_pages(docs_repo, manifest_pages)

    touched = [*written, *sorted(nav_touched)]
    if added:
        touched.append(f".docsync/{MANIFEST_FILE}")
    return touched
