"""From-scratch doc generation — `docsync bootstrap`.

Where the update pipeline (pipeline.py) edits *existing* pages from a diff, this
authors a *structured, sequenced docs site* from a whole-platform snapshot:

    B1 ingest  → RepoDigest per repo (ingest.walk_repos, read-only)
    B2 plan    → DocPlan: an ordered, sectioned IA (one judge-model call) + dedupe
    B3 author  → full MDX per page, kind-specific prompt (Opus, parallel, metered)
    B4 validate→ validate_new_page (absolute gates — no original to diff)
    B5 emit    → write files + ordered nav sections + manifest anchors (write_bootstrap)
    B6 PR      → pr.open_pr (the CLI wires this)

The site is planned as sections (Getting Started → Concepts → Architecture → Reference
→ Operations) with page *kinds* (concept | guide | reference). Narrative pages anchor
to broad subsystem globs and carry `judge_required`, so `docsync run` keeps them live
without firing an edit on every unrelated change. Every LLM call goes through the
injectable `client` (wrapped in `MeteredClient`), so cost lands on `result.usage`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fnmatch import fnmatch
from pathlib import Path

from . import cost
from . import ingest as ingest_mod
from .config import MANIFEST_FILE, merge_manifest_pages
from .cost import MeteredClient, UsageMeter
from .models import (
    SECTION_ORDER,
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

_PLAN_MAX_TOKENS = 6_000
_AUTHOR_MAX_TOKENS = 8_000
# Cap the digest handed to the planner so a big platform can't overflow its context.
_DIGEST_MAX_CHARS_PER_REPO = 12_000
# How many source files to excerpt into one page's author prompt.
_MAX_EXCERPT_FILES = 6


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


def render_digests(digests: list[RepoDigest], *, max_chars: int = _DIGEST_MAX_CHARS_PER_REPO) -> str:
    """A compact, per-repo listing of files + their top-level symbols for the planner."""
    blocks: list[str] = []
    for digest in digests:
        lines: list[str] = [f"## repo: {digest.repo}"]
        total = 0
        for unit in digest.units:
            syms = ", ".join(unit.symbols[:12])
            line = f"- {unit.path}" + (f"  ::  {syms}" if syms else "")
            if total + len(line) + 1 > max_chars:
                lines.append(f"… ({len(digest.units)} files in this repo; list truncated)")
                break
            lines.append(line)
            total += len(line) + 1
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _repo_units(digests: list[RepoDigest]) -> dict[str, tuple[str, list[str]]]:
    """Map repo id -> (root, [unit paths]) for resolving a page's source globs."""
    return {d.repo: (d.root, [u.path for u in d.units]) for d in digests}


# ---------------------------------------------------------------------------
# B2 — plan (information architecture)
# ---------------------------------------------------------------------------


def build_plan_prompt(
    digests: list[RepoDigest],
    *,
    existing_routes: list[str],
    existing_pages: set[str],
) -> tuple[str, str]:
    """Return (system, user) for the IA planner call (a sectioned, ordered site)."""
    repos = ", ".join(d.repo for d in digests)
    sections = ", ".join(SECTION_ORDER)
    system = "\n".join(
        [
            "You are a senior technical writer planning a complete developer "
            "documentation SITE for a software platform that currently has no docs.",
            f"The platform spans these source repos: {repos}.",
            "Design an information architecture with a real reading flow, not a flat list "
            "of API pages. Organize pages into ordered SECTIONS, using this vocabulary and "
            f"order where they apply: {sections}.",
            "Use three page KINDS:",
            "  - guide: task-oriented onboarding (Getting Started, how-to, setup/run).",
            "  - concept: narrative explanations of a subsystem or cross-service flow "
            "    (Concepts, Architecture, data flow) — prose, not API tables.",
            "  - reference: code-anchored API / data-model pages (Reference).",
            "Include platform-level narrative pages (an Introduction, an Architecture "
            "overview, a cross-service data-flow page) as concept/guide kind — these may "
            "span MULTIPLE repos.",
            "For EACH page provide: page_path (NEW kebab-case .mdx under a section folder, "
            "e.g. getting-started/introduction.mdx, architecture/data-flow.mdx, "
            "reference/alerts.mdx); title; kind; section (the nav group heading); order "
            "(integer, ascending within the section); a one-sentence summary; and sources "
            "— a list of {repo, globs, symbols} anchoring the page to real code. Reference "
            "pages should anchor to specific files + symbols; concept/guide pages should "
            "anchor to BROADER globs over the subsystem(s) they describe (few/no symbols).",
            "Every page MUST have at least one source with a real repo from the list. Do "
            "NOT propose a page whose path or route already exists. No duplicate paths.",
        ]
    )
    existing_block = (
        "\n".join(f"- {r}" for r in sorted(existing_routes) + sorted(existing_pages))
        or "(none)"
    )
    user = (
        f"# Source digests (path :: top-level symbols), per repo\n\n"
        f"{render_digests(digests)}\n\n"
        f"# Pages/routes that ALREADY EXIST (do not propose these)\n\n{existing_block}\n\n"
        "Produce a DocPlan: the ordered, sectioned set of pages to author."
    )
    return system, user


def plan_docs(
    digests: list[RepoDigest],
    docs_root: Path,
    config: DocsyncConfig,
    *,
    client,
    max_pages: int | None = None,
) -> tuple[DocPlan, list[str]]:
    """Ask the judge model for a sectioned DocPlan, then dedupe collisions + cap.

    Returns (plan, skipped). `skipped` lists planned page paths dropped for colliding
    with an existing page/route or an earlier plan entry. `max_pages` caps AFTER dedupe.
    """
    adapter = get_adapter("_.mdx")
    existing_pages = _existing_page_paths(docs_root)
    existing_routes = adapter.nav_routes(docs_root)
    system, user = build_plan_prompt(
        digests, existing_routes=existing_routes, existing_pages=existing_pages
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
        if not page.section:
            page.section = "Reference"
        kept.append(page)
        existing_pages.add(path)
        taken_routes.add(route)

    if max_pages and max_pages > 0:
        kept = kept[:max_pages]
    return DocPlan(pages=kept), skipped


# ---------------------------------------------------------------------------
# B3 — author (kind-specific)
# ---------------------------------------------------------------------------


def _gather_excerpts(
    planned: PlannedPage, repo_units: dict[str, tuple[str, list[str]]]
) -> list[tuple[str, str]]:
    """Resolve a page's source globs to (label, excerpt) pairs across repos."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src in planned.sources:
        root, paths = repo_units.get(src.repo, (None, []))
        if root is None:
            continue
        globs = src.globs or []
        for path in paths:
            if any(fnmatch(path, g) for g in globs) and (src.repo, path) not in seen:
                seen.add((src.repo, path))
                out.append((f"{src.repo}/{path}", ingest_mod.read_excerpt(root, path)))
                if len(out) >= _MAX_EXCERPT_FILES:
                    return out
    return out


_BASE_RULES = [
    "Return the ENTIRE file: YAML frontmatter with a non-empty `title` and "
    "`description`, then the MDX body.",
    "Use Mintlify components where they help (<CardGroup>/<Card>, <Steps>/<Step>, "
    "<Note>, <Warning>, <Tabs>/<Tab>); keep every component tag balanced and correctly "
    "nested, and keep code fences balanced.",
    "Ground every statement in the provided source code — do not invent APIs or behavior.",
    "Do NOT include navigation config; write a self-contained page.",
]

_KIND_INTRO = {
    "reference": "You author a single, complete Mintlify API/data-model REFERENCE page. "
    "Document the actual functions, routes, classes, fields, and enums precisely "
    "(tables are good here).",
    "concept": "You author a single, complete Mintlify CONCEPT page: a narrative that "
    "explains how this subsystem (or cross-service flow) works to a developer new to it. "
    "Favor clear prose, a mermaid diagram where it clarifies flow, and worked examples "
    "over exhaustive API tables.",
    "guide": "You author a single, complete Mintlify GUIDE page: task-oriented onboarding "
    "(what it is, how to set it up, run it, and make a first call). Use <Steps> for "
    "procedures and keep it practical.",
}


def build_author_prompt(planned: PlannedPage, excerpts: list[tuple[str, str]]) -> tuple[str, str]:
    """Return (system, user) for authoring one page, tailored to its kind."""
    intro = _KIND_INTRO.get(planned.kind, _KIND_INTRO["reference"])
    system = "\n".join([intro, *_BASE_RULES])
    code_blocks = "\n\n".join(
        f"## `{label}`\n\n```\n{text}\n```" for label, text in excerpts if text
    ) or "(no source excerpts resolved — write from the title and summary)"
    symbols = sorted({s for src in planned.sources for s in src.symbols})
    user = (
        f"# Page to write: {planned.page_path}\n\n"
        f"Title: {planned.title}\n"
        f"Section: {planned.section}  ·  Kind: {planned.kind}\n"
        f"Intended coverage: {planned.summary or '(use your judgment)'}\n"
        f"Key symbols: {', '.join(symbols) or '(none specified)'}\n\n"
        f"# Source code this page documents\n\n{code_blocks}\n\n"
        "Write the complete .mdx file now."
    )
    return system, user


def author_page(
    planned: PlannedPage,
    repo_units: dict[str, tuple[str, list[str]]],
    config: DocsyncConfig,
    *,
    client,
) -> str:
    """Generate the full MDX text for one planned page (Opus, structured output)."""
    excerpts = _gather_excerpts(planned, repo_units)
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
    repos: list[tuple[str, str | Path]],
    docs_repo: Path,
    config: DocsyncConfig,
    *,
    max_pages: int | None = None,
    check_links: bool = False,
    plan_only: bool = False,
    client=None,
    meter: UsageMeter | None = None,
) -> BootstrapResult:
    """Ingest repos → plan an IA → author → validate. Returns a BootstrapResult.

    `repos` is a list of ``(repo_id, path)`` — read-only ingest across the platform.
    Writing is deferred to `write_bootstrap`. With `plan_only=True` it stops after
    planning (no author spend). All LLM spend is metered onto `result.usage`.
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root

    meter = meter or UsageMeter()
    if client is not None:
        client = MeteredClient(client, meter)

    digests = ingest_mod.walk_repos(
        repos, exclude_dirs=ingest_mod.resolve_exclude_dirs(config.ingest_exclude_dirs)
    )
    repo_units = _repo_units(digests)
    result = BootstrapResult(repo=", ".join(d.repo for d in digests))

    plan, skipped = plan_docs(digests, docs_root, config, client=client, max_pages=max_pages)
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
            text = author_page(planned, repo_units, config, client=client)
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
    """Write authored pages, register ordered nav sections, and merge manifest anchors.

    Returns the repo-relative paths touched (pages + nav files + manifest). Writes files
    first, then nav (in reading-flow order), then manifest. Existing page files are left
    untouched unless `force`.
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

    # Register nav in reading-flow order: sections sequenced, pages ordered within.
    written_set = set(written)
    ordered_sections: list[tuple[str, list[str]]] = []
    for section_name, pages in result.plan.ordered_sections():
        routes = [_route_of(p.page_path) for p in pages if p.page_path in written_set]
        if routes:
            ordered_sections.append((section_name, routes))
    adapter.ensure_valid_docs_json(docs_root)
    nav_touched = set(adapter.set_nav_sections(docs_root, ordered_sections))

    # Merge manifest anchors (multi-repo sources + per-page judge_required) so
    # `docsync run` keeps every page — narrative included — live afterward.
    manifest_pages: list[ManifestPage] = []
    for page_path in written:
        planned = by_path.get(page_path)
        if planned is None:
            continue
        sources = [
            ManifestSource(repo=s.repo, globs=list(s.globs), symbols=list(s.symbols))
            for s in planned.sources
        ] or [ManifestSource(repo=result.repo)]
        manifest_pages.append(
            ManifestPage(
                path=page_path, sources=sources, judge_required=planned.judge_required
            )
        )
    added = merge_manifest_pages(docs_repo, manifest_pages)

    # Page + nav paths are docs_root-relative; prefix with docs_root so every path
    # is relative to the docs *repo* root — what pr.open_pr's `git add` expects (the
    # manifest path already is). A "." docs_root normalises away to a no-op.
    prefix = Path(config.docs_root)
    page_rel = [(prefix / p).as_posix() for p in written]
    nav_rel = [(prefix / n).as_posix() for n in sorted(nav_touched)]
    touched = [*page_rel, *nav_rel]
    if added:
        touched.append(f".docsync/{MANIFEST_FILE}")
    return touched
