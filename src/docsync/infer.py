"""Stage I — manifest inference for `docsync infer` (the brownfield analogue of bootstrap).

Where `bootstrap` authors NEW pages and emits their anchors, `infer` takes a docs site
whose pages ALREADY EXIST but carry no manifest anchors, and proposes anchors for them so
`docsync run` can keep them live. The hybrid engine:

    I1 discover   → existing pages not yet in the manifest (idempotent re-runs)
    I2 ingest     → RepoDigest per source checkout (read-only, ingest.walk_repos)
    I3 shortlist  → a CODE-side embedding index ranks the units each page likely documents
    I4 judge      → Haiku confirms/sharpens globs+symbols+kind+confidence per page
    I5 validate   → deterministic honesty gate: drop any glob matching no real file and any
                    symbol not present in the matched code, so what we write passes `doctor`
    I6 emit       → merge anchors into .docsync/manifest.yml (idempotent, comment-preserving)

Every LLM call goes through the injectable `client` (wrapped in `MeteredClient`); the
embedding encoder is injectable too, so the whole path is unit-testable without torch.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from . import embeddings as emb
from . import ingest as ingest_mod
from . import llm
from .bootstrap import _existing_page_paths
from .config import DOCSYNC_DIR, load_manifest, merge_manifest_pages
from .cost import MeteredClient, UsageMeter
from .impact import _repo_key
from .models import (
    DocsyncConfig,
    InferredAnchors,
    InferredPage,
    InferResult,
    ManifestPage,
    ManifestSource,
    RepoDigest,
)
from .pool import run_parallel
from .scaffold import _read_corpus
from .validate import get_adapter

# Plenty for one page's structured anchors; the judge output is small.
_INFER_MAX_TOKENS = 1_500
# Chars of page body folded into the query + judge prompt.
_PAGE_BODY_CHARS = 2_000
# Per-candidate excerpt shown to the judge (a touch larger than the index excerpt).
_CANDIDATE_EXCERPT_CHARS = 1_500


# ---------------------------------------------------------------------------
# Page query text
# ---------------------------------------------------------------------------


def _page_query_text(adapter, docs_root: Path, page_path: str) -> tuple[str, str, str, str]:
    """Return (title, summary, body_excerpt, query_text) for one existing page.

    `query_text` is what we embed to shortlist code: title + description + heading texts
    + the first chunk of body, which together carry the page's topical signal.
    """
    text = (Path(docs_root) / page_path).read_text(encoding="utf-8")
    meta, body = adapter.split_frontmatter(text)
    title = str(meta.get("title", "") or "")
    summary = str(meta.get("description", "") or "")
    headings = [ln.strip() for ln in body.splitlines() if ln.lstrip().startswith("#")]
    body_excerpt = body[:_PAGE_BODY_CHARS]
    query = "\n".join([title, summary, *headings, body_excerpt]).strip()
    return title, summary, body_excerpt, query


# ---------------------------------------------------------------------------
# I4 — judge prompt
# ---------------------------------------------------------------------------


def build_infer_prompt(
    page_path: str,
    title: str,
    summary: str,
    body_excerpt: str,
    candidates: list[tuple[str, str, list[str], str]],
) -> tuple[str, str]:
    """Return (system, user) for inferring one page's anchors from shortlisted code."""
    system = "\n".join(
        [
            "You are a technical-documentation analyst. Given ONE documentation page and a "
            "shortlist of candidate source-code units (already pre-filtered by semantic "
            "similarity), decide which units this page actually documents and produce "
            "manifest anchors that keep the page in sync with that code.",
            "Rules:",
            "  1. Choose ONLY from the provided candidates — never invent repos or paths. "
            "If none genuinely matches, return no sources and a low confidence.",
            "  2. Classify the page KIND: 'reference' = documents specific APIs / functions "
            "/ data-models (anchor to precise files + the symbols it describes); 'concept' = "
            "narrative explanation of a subsystem or flow; 'guide' = task-oriented "
            "onboarding. Concept/guide pages should anchor to BROADER globs and few/no "
            "symbols.",
            "  3. For each chosen source give {repo, globs, symbols}. `globs` are fnmatch "
            "patterns over repo-relative file paths (e.g. 'src/routes/*.py'); `symbols` are "
            "names the page describes (a trailing '*' means prefix-match). Use the EXACT "
            "repo id shown before each candidate's path.",
            "  4. Give a calibrated `confidence` in [0,1]: high only when the anchors clearly "
            "cover what the page is about.",
        ]
    )
    cand_blocks: list[str] = []
    for repo, path, symbols, excerpt in candidates:
        syms = ", ".join(symbols) if symbols else "(none)"
        cand_blocks.append(
            f"## {repo}/{path}\nsymbols: {syms}\n\n```\n{excerpt}\n```"
        )
    user = (
        f"# Page: {page_path}\n"
        f"Title: {title or '(none)'}\n"
        f"Description: {summary or '(none)'}\n\n"
        f"## Page body (excerpt)\n\n{body_excerpt or '(empty)'}\n\n"
        f"# Candidate source units (choose from these only)\n\n"
        + "\n\n".join(cand_blocks)
        + "\n\nReturn the anchors for this page."
    )
    return system, user


def judge_page(
    page_path: str,
    title: str,
    summary: str,
    body_excerpt: str,
    candidates: list[tuple[str, str, list[str], str]],
    config: DocsyncConfig,
    *,
    client,
) -> InferredAnchors:
    """Ask the judge model to confirm/sharpen anchors for one page (structured output)."""
    system, user = build_infer_prompt(page_path, title, summary, body_excerpt, candidates)
    anchors = llm.parse(
        client,
        stage="infer",
        model=config.models.judge_model,
        max_tokens=_INFER_MAX_TOKENS,
        system=system,
        user=user,
        output_format=InferredAnchors,
    )
    anchors.page_path = page_path  # trust our path, not the model's echo
    return anchors


# ---------------------------------------------------------------------------
# I5 — deterministic honesty gate
# ---------------------------------------------------------------------------


def _repo_index(digests: list[RepoDigest]) -> dict[str, tuple[str, str, list[str]]]:
    """Map normalized repo key -> (canonical repo_id, root, [unit paths]).

    Keyed by `impact._repo_key` so a judge that echoes a slightly different repo string
    ('owner/name' vs bare 'name') still resolves the same way mapping/doctor would.
    """
    out: dict[str, tuple[str, str, list[str]]] = {}
    for d in digests:
        out[_repo_key(d.repo)] = (d.repo, d.root, [u.path for u in d.units])
    return out


def _collapse_globs(globs: list[str], repo_paths: list[str]) -> list[str]:
    """Conservatively collapse exact-sibling file globs to `dir/*.ext`.

    Only fires when 2+ literal (wildcard-free) file globs share a directory + extension
    AND every real file of that directory+extension is already among the literals — so the
    collapsed glob never silently pulls in a sibling the judge didn't choose. Anything not
    eligible is kept verbatim.
    """
    literals = [g for g in globs if "*" not in g and "?" not in g and "[" not in g]
    others = [g for g in globs if g not in literals]
    by_bucket: dict[tuple[str, str], list[str]] = {}
    for g in literals:
        p = Path(g)
        by_bucket.setdefault((p.parent.as_posix(), p.suffix), []).append(g)

    collapsed: list[str] = []
    used: set[str] = set()
    for (parent, suffix), members in by_bucket.items():
        if len(members) < 2 or not suffix:
            continue
        pattern = f"{parent}/*{suffix}" if parent not in ("", ".") else f"*{suffix}"
        real = [p for p in repo_paths if fnmatch(p, pattern)]
        if real and set(real) == set(members):  # exact cover, no extra files
            collapsed.append(pattern)
            used.update(members)

    kept = [g for g in literals if g not in used]
    return [*others, *kept, *collapsed]


def _validate_sources(
    anchors: InferredAnchors, repo_index: dict[str, tuple[str, str, list[str]]]
) -> list[ManifestSource]:
    """Sanitize the judge's proposed sources against the real source tree.

    Drops sources whose repo isn't ingested, globs that match no real file, and symbols
    absent from the matched code — so every surviving anchor would pass `docsync doctor`.
    """
    out: list[ManifestSource] = []
    for src in anchors.sources:
        resolved = repo_index.get(_repo_key(src.repo))
        if resolved is None:
            continue
        repo_id, root, repo_paths = resolved

        valid_globs = [g for g in src.globs if any(fnmatch(p, g) for p in repo_paths)]
        if not valid_globs:
            continue
        valid_globs = _collapse_globs(valid_globs, repo_paths)

        matched = [
            Path(root) / p for p in repo_paths if any(fnmatch(p, g) for g in valid_globs)
        ]
        corpus = _read_corpus(matched) if src.symbols else ""
        valid_symbols = [
            s for s in src.symbols
            if (s[:-1] if s.endswith("*") else s) and (s[:-1] if s.endswith("*") else s) in corpus
        ]
        out.append(ManifestSource(repo=repo_id, globs=valid_globs, symbols=valid_symbols))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_infer(
    repos: list[tuple[str, str | Path]],
    docs_repo: Path,
    config: DocsyncConfig,
    *,
    max_pages: int | None = None,
    client=None,
    encoder=None,
    meter: UsageMeter | None = None,
) -> InferResult:
    """Infer manifest anchors for the unanchored pages of an existing docs site.

    Read-only on the source `repos`; writes nothing (use `write_infer` to persist). All
    LLM spend is metered onto `result.usage`. `encoder` defaults to the sentence-
    transformers encoder (raises ImportError if the `embeddings` extra is absent — the
    CLI turns that into a clean install hint).
    """
    docs_repo = Path(docs_repo)
    docs_root = docs_repo / config.docs_root

    meter = meter or UsageMeter()
    if client is not None:
        client = MeteredClient(client, meter)

    # Resolve a single concrete encoder up front so per-page queries reuse it (a None
    # encoder would otherwise reload the model on every call).
    if encoder is None:
        encoder = emb.default_encoder(config.embedding_model)

    # I1 — discover unanchored pages.
    existing = sorted(_existing_page_paths(docs_root))
    anchored: set[str] = set()
    try:
        anchored = {p.path for p in load_manifest(docs_repo).pages}
    except FileNotFoundError:
        pass
    work = [p for p in existing if p not in anchored]
    skipped = [p for p in existing if p in anchored]
    if max_pages and max_pages > 0:
        work = work[:max_pages]

    result = InferResult(skipped_already_anchored=skipped)
    if not work:
        result.usage = meter.finalize()
        return result

    # I2 — ingest source checkouts (read-only).
    digests = ingest_mod.walk_repos(
        repos, exclude_dirs=ingest_mod.resolve_exclude_dirs(config.ingest_exclude_dirs)
    )
    repo_index = _repo_index(digests)
    roots = {d.repo: d.root for d in digests}
    symbols_by_unit = {(d.repo, u.path): list(u.symbols) for d in digests for u in d.units}

    # I3 — code-side embedding index (cached separately from the doc recall-net).
    cache_dir = docs_repo / DOCSYNC_DIR / "state" / "infer-embeddings"
    code_index = emb.load_or_build_code_index(
        digests, cache_dir, model_name=config.embedding_model, encoder=encoder
    )

    adapter = get_adapter("_.mdx")

    def _infer_one(page_path: str) -> InferredPage:
        """Shortlist → judge → validate one page. Never raises (pool-safe)."""
        title, summary, body_excerpt, query = _page_query_text(adapter, docs_root, page_path)
        hits = emb.query_code_index(
            code_index, query, encoder=encoder, model_name=config.embedding_model,
            top_k=config.embedding_top_k, floor=config.embedding_floor,
        )
        if not hits:
            return InferredPage(
                page_path=page_path, status="no_match",
                reason="no source unit above the similarity floor",
            )
        candidates = [
            (
                repo, path, symbols_by_unit.get((repo, path), []),
                ingest_mod.read_excerpt(roots[repo], path, max_chars=_CANDIDATE_EXCERPT_CHARS),
            )
            for (repo, path), _score in hits
        ]
        try:
            anchors = judge_page(
                page_path, title, summary, body_excerpt, candidates, config, client=client
            )
        except Exception as exc:  # noqa: BLE001 - one bad page must not abort the pool
            return InferredPage(
                page_path=page_path, status="no_match", reason=f"judge failed: {exc}"
            )

        sources = _validate_sources(anchors, repo_index)
        if not sources:
            return InferredPage(
                page_path=page_path, kind=anchors.kind, confidence=anchors.confidence,
                status="no_match", reason=anchors.reason or "no anchor survived validation",
            )
        status = (
            "anchored" if anchors.confidence >= config.judge_confidence_threshold
            else "low_confidence"
        )
        return InferredPage(
            page_path=page_path, sources=sources, kind=anchors.kind,
            confidence=anchors.confidence, status=status, reason=anchors.reason,
        )

    pages = run_parallel(_infer_one, work, config.max_parallel_requests)

    result.pages = pages
    result.usage = meter.finalize()
    return result


def write_infer(result: InferResult, docs_repo: Path, config: DocsyncConfig) -> list[str]:
    """Merge the validated anchors into .docsync/manifest.yml. Returns paths added.

    Idempotent (via `merge_manifest_pages`): only `anchored` pages are written, and a
    re-run that re-proposes an already-present page is a no-op.
    """
    pages = [
        ManifestPage(
            path=p.page_path, sources=p.sources, judge_required=p.judge_required
        )
        for p in result.anchored()
    ]
    if not pages:
        return []
    return merge_manifest_pages(docs_repo, pages)
