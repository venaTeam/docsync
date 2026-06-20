"""Stage 3 — doc-impact mapping: which pages does a code diff affect?

A hybrid, anchor-first mapper:

1. **Anchors** (deterministic) — match the diff's changed paths/symbols against the
   manifest's declared source globs/symbols. These are high-precision and, when
   `config.anchor_autopass` is set, skip the LLM judge entirely.
2. **Embeddings** (optional recall-net) — a semantic fallback that catches pages
   whose manifest anchors are stale or missing. Needs the optional
   `sentence-transformers` extra; degrades to a no-op when it isn't installed.
3. **Judge** (Haiku LLM) — confirms candidate pages are genuinely invalidated by the
   change before they reach the expensive edit stage.

`map_impact()` is the orchestrator pipeline.py calls.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from . import embeddings as embeddings_mod
from . import llm
from .llm import get_client
from .models import (
    CandidateSource,
    CodeDiff,
    DocsyncConfig,
    ImpactCandidate,
    ImpactedPage,
    JudgeVerdict,
    Manifest,
    ManifestPage,
)
from .pool import run_parallel

# Cap the page text we feed the judge — keeps the prompt cheap and well within Haiku's
# window while preserving enough of the page to judge relevance.
_MAX_PAGE_CHARS = 6000

_JUDGE_SYSTEM = (
    "You decide whether a documentation page is invalidated by a code change. "
    "Answer affected=true only if the diff changes something the page documents "
    "(routes, env vars, schemas, metrics, behavior). Be conservative: prefer "
    "affected=false when the change is internal/refactoring and does not alter "
    "anything a reader of the page would observe. Provide a calibrated confidence "
    "in [0, 1] and a one-sentence reason."
)


# ---------------------------------------------------------------------------
# 1. Anchors — deterministic, manifest-driven
# ---------------------------------------------------------------------------


def _symbol_matches(changed: str, pattern: str) -> bool:
    """Exact match, plus trailing-* prefix match (e.g. 'ENV_*' matches 'ENV_HOST')."""
    if pattern.endswith("*"):
        return changed.startswith(pattern[:-1])
    return changed == pattern


def _repo_key(repo: str) -> str:
    """Normalize a repo reference to its bare name for scoping.

    A manifest source uses the canonical ``owner/name``, but a diff's ``repo``
    can be a local checkout path (CLI dogfood), a fork ``otherowner/name``, or
    ``owner/name`` (gh path). Comparing the final path component reconciles all
    three so the same anchor matches regardless of how the diff was produced.
    """
    key = repo.rstrip("/").rsplit("/", 1)[-1]
    if key.endswith(".git"):
        key = key[:-4]
    return key.lower()


def _repo_matches(source_repo: str, diff_repo: str) -> bool:
    # An empty source repo is a wildcard ("the only/local repo"): it matches any diff,
    # so mono- and single-repo manifests need not repeat a repo on every anchor.
    if not source_repo:
        return True
    return _repo_key(source_repo) == _repo_key(diff_repo)


def filter_docs_paths(diff: CodeDiff, docs_root: str) -> CodeDiff:
    """Drop files under *docs_root* from *diff* (for mono-repo runs).

    In a mono repo the source diff also carries the docs subtree; left in, a merged
    doc change would map onto itself and spuriously drive further doc edits. Comparing
    against ``docs_root`` removes those files (by ``path`` and any ``previous_path``)
    so only real code changes reach impact mapping. A ``docs_root`` of ``"."`` (docs at
    the repo root) is a no-op — there is no code subtree to separate.
    """
    root = docs_root.strip("/")
    if not root or root == ".":
        return diff

    prefix = root + "/"

    def _under_docs(path: str | None) -> bool:
        return bool(path) and (path == root or path.startswith(prefix))

    kept = [
        f for f in diff.files
        if not (_under_docs(f.path) or _under_docs(f.previous_path))
    ]
    if len(kept) == len(diff.files):
        return diff
    return diff.model_copy(update={"files": kept})


def find_anchor_candidates(diff: CodeDiff, manifest: Manifest) -> list[ImpactCandidate]:
    """Match the diff against each manifest page's declared sources.

    Paths are matched via fnmatch against `source.globs` (only for sources whose
    `source.repo == diff.repo`). Symbols are matched exactly against `source.symbols`,
    plus trailing-`*` prefix matching. Score is the number of matches; the reason
    names what matched.
    """
    changed_paths = diff.changed_paths()
    changed_symbols = diff.all_symbols()
    candidates: list[ImpactCandidate] = []

    for page in manifest.pages:
        path_hits: list[str] = []
        symbol_hits: list[str] = []

        for source in page.sources:
            if not _repo_matches(source.repo, diff.repo):
                continue  # repo-scoped: a source for another repo can't match this diff

            for glob in source.globs:
                for path in changed_paths:
                    if fnmatch.fnmatch(path, glob):
                        path_hits.append(f"{path} ~ {glob}")

            for pattern in source.symbols:
                for symbol in changed_symbols:
                    if _symbol_matches(symbol, pattern):
                        symbol_hits.append(
                            f"{symbol} (={pattern})" if symbol == pattern
                            else f"{symbol} (~{pattern})"
                        )

        score = len(path_hits) + len(symbol_hits)
        if score == 0:
            continue

        reason_parts: list[str] = []
        if path_hits:
            reason_parts.append("paths: " + ", ".join(path_hits))
        if symbol_hits:
            reason_parts.append("symbols: " + ", ".join(symbol_hits))

        candidates.append(
            ImpactCandidate(
                page_path=page.path,
                source=CandidateSource.ANCHOR,
                score=float(score),
                reason="; ".join(reason_parts),
            )
        )

    return candidates


# ---------------------------------------------------------------------------
# 2. Embeddings — optional semantic recall-net
# ---------------------------------------------------------------------------


def _query_tokens(diff: CodeDiff, config: DocsyncConfig) -> list[str]:
    """Identifier tokens to embed: changed symbols + changed-path basenames,
    minus configured stopword symbols."""
    stop = {s.lower() for s in config.stopword_symbols}
    tokens: list[str] = []

    for symbol in diff.all_symbols():
        if symbol.lower() not in stop:
            tokens.append(symbol)

    for path in diff.changed_paths():
        base = Path(path).name
        stem = base.rsplit(".", 1)[0] if "." in base else base
        if stem and stem.lower() not in stop:
            tokens.append(stem)

    # Dedupe, preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


def find_embedding_candidates(
    diff: CodeDiff,
    docs_root: Path,
    pages: list[ManifestPage] | None,
    config: DocsyncConfig,
    *,
    cache_dir: Path | None = None,
    encoder=None,
    top_k: int | None = None,
) -> list[ImpactCandidate]:
    """Optional recall-net: rank doc pages by semantic similarity to the diff.

    Builds (or loads a cached) embedding index over the docs and queries it with the
    diff's identifier tokens. `pages=None` scans the whole docs tree — the recall-net
    over pages the manifest doesn't anchor. With `cache_dir` set, the index is
    persisted and reused across runs unless the docs content changed.

    The encoder is injectable for testing; in production it defaults to
    sentence-transformers. If that optional extra isn't installed, returns [] so the
    pipeline degrades gracefully to anchors only. Excluding already-anchored pages is
    the caller's job.
    """
    query_tokens = _query_tokens(diff, config)
    if not query_tokens:
        return []

    enc = encoder
    if enc is None:
        try:
            enc = embeddings_mod.default_encoder(config.embedding_model)
        except ImportError:
            return []  # `embeddings` extra absent — recall-net unavailable

    page_paths = [p.path for p in pages] if pages is not None else None
    index = embeddings_mod.load_or_build(
        docs_root,
        cache_dir,
        model_name=config.embedding_model,
        encoder=enc,
        page_paths=page_paths,
    )
    ranked = embeddings_mod.query_index(
        index,
        " ".join(query_tokens),
        encoder=enc,
        model_name=config.embedding_model,
        top_k=top_k or config.embedding_top_k,
        floor=config.embedding_floor,
    )
    return [
        ImpactCandidate(
            page_path=page_path,
            source=CandidateSource.EMBEDDING,
            score=score,
            reason=f"semantic match (cos={score:.2f}) on: {', '.join(query_tokens[:8])}",
        )
        for page_path, score in ranked
    ]


# ---------------------------------------------------------------------------
# 3. Judge — Haiku LLM confirmation
# ---------------------------------------------------------------------------


def _read_page_text(docs_root: Path, page_path: str) -> str:
    try:
        text = (Path(docs_root) / page_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return text[:_MAX_PAGE_CHARS]


def _judge_user_message(diff: CodeDiff, candidate: ImpactCandidate, page_text: str) -> str:
    changed_files = "\n".join(f"  - {p}" for p in diff.changed_paths()) or "  (none)"
    symbols = ", ".join(diff.all_symbols()) or "(none)"
    return (
        f"# Documentation page: {candidate.page_path}\n\n"
        f"```\n{page_text}\n```\n\n"
        f"# Code change (repo {diff.repo}, {diff.base_sha[:8]}..{diff.head_sha[:8]})\n\n"
        f"Changed files:\n{changed_files}\n\n"
        f"Changed symbols: {symbols}\n\n"
        f"Why this page is a candidate ({candidate.source.value}): {candidate.reason}\n\n"
        "Is this documentation page invalidated by the code change?"
    )


def judge_candidates(
    diff: CodeDiff,
    candidates: list[ImpactCandidate],
    docs_root: Path,
    config: DocsyncConfig,
    client=None,
) -> list[JudgeVerdict]:
    """Ask the Haiku judge whether each candidate page is truly affected.

    One page per API call (Haiku judges a single page at a time). Candidates are
    independent, so they're judged concurrently (bounded by
    `config.max_parallel_requests`); `ThreadPoolExecutor.map` preserves input order
    so the returned verdicts align with `candidates`. Uses the SDK's structured-output
    helper (`messages.parse`). `client` is injectable for tests; defaults to
    `anthropic.Anthropic()`. A failed call yields a non-affected verdict (confidence 0)
    so one failure doesn't abort the run.
    """
    if not candidates:
        return []

    client = get_client(client)

    def _judge_one(candidate: ImpactCandidate) -> JudgeVerdict:
        page_text = _read_page_text(docs_root, candidate.page_path)
        try:
            verdict = llm.parse(
                client,
                stage="judge",
                model=config.models.judge_model,
                max_tokens=1024,
                system=_JUDGE_SYSTEM,
                user=_judge_user_message(diff, candidate, page_text),
                output_format=JudgeVerdict,
            )
            # The model may omit or echo page_path; pin it to the candidate's.
            verdict.page_path = candidate.page_path
            return verdict
        except Exception as exc:  # noqa: BLE001 — one failure must not kill the run
            return JudgeVerdict(
                page_path=candidate.page_path,
                affected=False,
                confidence=0.0,
                reason=f"judge call failed: {type(exc).__name__}: {exc}",
            )

    return run_parallel(_judge_one, candidates, config.max_parallel_requests)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _dedupe_anchor_wins(candidates: list[ImpactCandidate]) -> list[ImpactCandidate]:
    """Collapse to one candidate per page_path. Anchors win over embeddings;
    among same-source duplicates, the higher score wins."""
    by_path: dict[str, ImpactCandidate] = {}
    for cand in candidates:
        existing = by_path.get(cand.page_path)
        if existing is None:
            by_path[cand.page_path] = cand
            continue
        existing_anchor = existing.source == CandidateSource.ANCHOR
        cand_anchor = cand.source == CandidateSource.ANCHOR
        if cand_anchor and not existing_anchor:
            by_path[cand.page_path] = cand
        elif cand_anchor == existing_anchor and cand.score > existing.score:
            by_path[cand.page_path] = cand
    return list(by_path.values())


def map_impact(
    diff: CodeDiff,
    manifest: Manifest,
    docs_root: Path,
    config: DocsyncConfig,
    *,
    use_embeddings: bool = False,
    cache_dir: Path | None = None,
    client=None,
) -> list[ImpactedPage]:
    """Orchestrate the doc-impact pipeline.

    anchors (+ optional embeddings) -> dedupe by page_path (anchor wins) -> judge
    -> keep pages where (anchor_autopass and source == ANCHOR) OR
       (verdict.affected and verdict.confidence >= judge_confidence_threshold).
    """
    candidates = find_anchor_candidates(diff, manifest)

    if use_embeddings:
        anchored = {c.page_path for c in candidates}
        # Full-tree recall-net: scan EVERY page (not just manifest pages) so drift on
        # pages the manifest doesn't anchor is still surfaced; drop already-anchored
        # pages, and let the judge confirm the rest (embeddings never autopass).
        embedding_candidates = find_embedding_candidates(
            diff, docs_root, None, config, cache_dir=cache_dir
        )
        candidates.extend(
            c for c in embedding_candidates if c.page_path not in anchored
        )

    candidates = _dedupe_anchor_wins(candidates)
    if not candidates:
        return []

    # Anchors that autopass don't need the judge; everything else does. A page marked
    # `judge_required` in the manifest (narrative/concept pages with broad subsystem
    # anchors) is NEVER autopassed — the judge confirms a real invalidation first, so a
    # broad anchor doesn't fire a costly edit on every unrelated change in the subsystem.
    autopass_paths: set[str] = set()
    to_judge: list[ImpactCandidate] = []
    for cand in candidates:
        manifest_page = manifest.page(cand.page_path)
        judge_required = bool(manifest_page and manifest_page.judge_required)
        if (
            config.anchor_autopass
            and cand.source == CandidateSource.ANCHOR
            and not judge_required
        ):
            autopass_paths.add(cand.page_path)
        else:
            to_judge.append(cand)

    verdicts = judge_candidates(diff, to_judge, docs_root, config, client=client)
    verdict_by_path = {v.page_path: v for v in verdicts}

    impacted: list[ImpactedPage] = []
    for cand in candidates:
        if cand.page_path in autopass_paths:
            impacted.append(
                ImpactedPage(
                    page_path=cand.page_path,
                    source=cand.source,
                    confidence=1.0,
                    reason=f"anchor autopass — {cand.reason}",
                )
            )
            continue

        verdict = verdict_by_path.get(cand.page_path)
        if verdict is None:
            continue
        if verdict.affected and verdict.confidence >= config.judge_confidence_threshold:
            impacted.append(
                ImpactedPage(
                    page_path=cand.page_path,
                    source=cand.source,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            )

    return impacted
