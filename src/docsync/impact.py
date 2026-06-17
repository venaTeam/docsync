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

# Cap the page text we feed the judge — keeps the prompt cheap and well within Haiku's
# window while preserving enough of the page to judge relevance.
_MAX_PAGE_CHARS = 6000

# Similarity floor for the embedding recall-net. Below this, a page is too weakly
# related to the diff's identifiers to be worth a judge call.
_EMBEDDING_FLOOR = 0.2

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
            if source.repo != diff.repo:
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


def _iter_doc_chunks(docs_root: Path, pages: list[ManifestPage] | None):
    """Yield (page_path, chunk_text) for heading sections of each .mdx page.

    If `pages` is given, only those pages are indexed; otherwise every .mdx under
    docs_root is scanned. A "chunk" is a markdown heading section (text from one
    `#`-prefixed heading up to the next).
    """
    if pages is not None:
        paths = [docs_root / p.path for p in pages]
    else:
        paths = sorted(docs_root.rglob("*.mdx"))

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # Report page_path in the manifest-relative form.
        try:
            rel = str(path.relative_to(docs_root))
        except ValueError:
            rel = path.name

        current: list[str] = []
        for line in text.splitlines():
            if line.lstrip().startswith("#") and current:
                chunk = "\n".join(current).strip()
                if chunk:
                    yield rel, chunk
                current = [line]
            else:
                current.append(line)
        tail = "\n".join(current).strip()
        if tail:
            yield rel, tail


def find_embedding_candidates(
    diff: CodeDiff,
    docs_root: Path,
    pages: list[ManifestPage] | None,
    config: DocsyncConfig,
    top_k: int = 5,
) -> list[ImpactCandidate]:
    """Optional recall-net: rank doc pages by semantic similarity to the diff.

    Imports `sentence_transformers` lazily; if it isn't installed, returns []
    (the extra is optional — anchors cover the high-drift pages without it).
    Excluding pages already covered by anchors is the caller's job.
    """
    try:
        from sentence_transformers import SentenceTransformer  # lazy: optional extra
    except ImportError:
        return []

    import numpy as np

    query_tokens = _query_tokens(diff, config)
    if not query_tokens:
        return []

    chunks = list(_iter_doc_chunks(docs_root, pages))
    if not chunks:
        return []

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query = " ".join(query_tokens)
    query_vec = np.asarray(model.encode([query])[0], dtype=np.float32)
    chunk_vecs = np.asarray(
        model.encode([c for _, c in chunks]), dtype=np.float32
    )

    def _cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b, axis=1))
        denom = np.where(denom == 0.0, 1e-12, denom)
        return (b @ a) / denom

    sims = _cosine(query_vec, chunk_vecs)

    # Reduce per-chunk similarities to a single best score per page.
    best: dict[str, float] = {}
    for (page_path, _chunk), sim in zip(chunks, sims):
        val = float(sim)
        if page_path not in best or val > best[page_path]:
            best[page_path] = val

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    candidates: list[ImpactCandidate] = []
    for page_path, score in ranked:
        if score < _EMBEDDING_FLOOR:
            continue
        candidates.append(
            ImpactCandidate(
                page_path=page_path,
                source=CandidateSource.EMBEDDING,
                score=score,
                reason=f"semantic match (cos={score:.2f}) on: {', '.join(query_tokens[:8])}",
            )
        )
        if len(candidates) >= top_k:
            break

    return candidates


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

    One page per API call (Haiku judges a single page at a time). Uses the SDK's
    structured-output helper (`messages.parse`). `client` is injectable for tests;
    defaults to `anthropic.Anthropic()`. A failed call yields a non-affected verdict
    (confidence 0) so one failure doesn't abort the run.
    """
    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    verdicts: list[JudgeVerdict] = []
    for candidate in candidates:
        page_text = _read_page_text(docs_root, candidate.page_path)
        try:
            resp = client.messages.parse(
                model=config.models.judge_model,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": _JUDGE_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": _judge_user_message(diff, candidate, page_text),
                    }
                ],
                output_format=JudgeVerdict,
            )
            verdict = resp.parsed_output
            # The model may not echo the page_path; pin it to the candidate's.
            if not verdict.page_path:
                verdict.page_path = candidate.page_path
            else:
                verdict.page_path = candidate.page_path
            verdicts.append(verdict)
        except Exception as exc:  # noqa: BLE001 — one failure must not kill the run
            verdicts.append(
                JudgeVerdict(
                    page_path=candidate.page_path,
                    affected=False,
                    confidence=0.0,
                    reason=f"judge call failed: {type(exc).__name__}: {exc}",
                )
            )

    return verdicts


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
        # Bias the embedding scan toward manifest pages not already anchored.
        remaining_pages = [p for p in manifest.pages if p.path not in anchored]
        embedding_candidates = find_embedding_candidates(
            diff, docs_root, remaining_pages or None, config
        )
        candidates.extend(
            c for c in embedding_candidates if c.page_path not in anchored
        )

    candidates = _dedupe_anchor_wins(candidates)
    if not candidates:
        return []

    # Anchors that autopass don't need the judge; everything else does.
    autopass_paths: set[str] = set()
    to_judge: list[ImpactCandidate] = []
    for cand in candidates:
        if config.anchor_autopass and cand.source == CandidateSource.ANCHOR:
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
