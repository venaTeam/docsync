"""Optional embeddings recall-net — a cached, full-tree semantic index over the docs.

This is the recall layer behind the anchor mapper: it catches drift on pages the
manifest does **not** anchor, by ranking doc chunks against the diff's identifier
tokens. It is deliberately decoupled from the heavy `sentence-transformers` extra:

* The **encoder** is injectable (`Encoder = Callable[[list[str]], np.ndarray]`), so
  the index / cache / query logic is fully unit-testable with a tiny deterministic
  fake encoder — no torch required. Production calls `default_encoder()`, which
  lazily loads `all-MiniLM-L6-v2`.
* The index is **cached to disk** keyed on a content hash of the indexed chunks, so
  a run reuses the previous embedding pass unless the docs actually changed. This is
  what the CI `actions/cache` step on `.docsync/state/embeddings` is for.

Vectors are L2-normalized at build time, so a query is a single dot product (cosine).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# An encoder turns a list of texts into a 2-D float array (n_texts, dim).
Encoder = Callable[[list[str]], "np.ndarray"]

_META_FILE = "index.json"
_VECTORS_FILE = "vectors.npy"


# ---------------------------------------------------------------------------
# Chunking + content hashing
# ---------------------------------------------------------------------------


def iter_doc_chunks(
    docs_root: Path, page_paths: list[str] | None = None
) -> Iterator[tuple[str, str]]:
    """Yield (relative_page_path, chunk_text) for each markdown heading section.

    If `page_paths` is given, only those pages are read; otherwise every `*.mdx`
    under `docs_root` is scanned (the full recall-net). A chunk runs from one
    `#`-prefixed heading to the next.
    """
    docs_root = Path(docs_root)
    if page_paths is not None:
        paths = [docs_root / p for p in page_paths]
    else:
        paths = sorted(docs_root.rglob("*.mdx"))

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
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


def chunks_content_hash(chunks: list[tuple[str, str]], model_name: str) -> str:
    """Stable content hash over (page_path, chunk_text) pairs + the model name.

    Content-based (not mtime-based) so it is reproducible across machines and CI
    checkouts — two trees with identical docs produce the same key and share cache.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for page_path, chunk in chunks:
        h.update(b"\x00")
        h.update(page_path.encode("utf-8"))
        h.update(b"\x01")
        h.update(chunk.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def default_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Lazily build the production sentence-transformers encoder.

    Raises ImportError if the optional `embeddings` extra isn't installed — callers
    treat that as "recall-net unavailable" and degrade to anchors only.
    """
    from sentence_transformers import SentenceTransformer  # optional extra

    model = SentenceTransformer(model_name)

    def _encode(texts: list[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts)), dtype=np.float32)

    return _encode


def _normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1e-12, norms)
    return mat / norms


# ---------------------------------------------------------------------------
# Index — build / persist / load / query
# ---------------------------------------------------------------------------


@dataclass
class DocIndex:
    """A normalized embedding matrix over doc chunks, with provenance for caching."""

    model_name: str
    content_hash: str
    page_paths: list[str]  # one entry per row of `vectors`
    vectors: np.ndarray  # (n_chunks, dim), L2-normalized

    def save(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / _VECTORS_FILE, self.vectors)
        (cache_dir / _META_FILE).write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "content_hash": self.content_hash,
                    "page_paths": self.page_paths,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, cache_dir: Path) -> "DocIndex | None":
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / _META_FILE
        vec_path = cache_dir / _VECTORS_FILE
        if not meta_path.exists() or not vec_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vectors = np.load(vec_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return cls(
            model_name=meta["model_name"],
            content_hash=meta["content_hash"],
            page_paths=list(meta["page_paths"]),
            vectors=vectors,
        )


def build_index(
    docs_root: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    encoder: Encoder | None = None,
    page_paths: list[str] | None = None,
) -> DocIndex:
    """Encode every doc chunk into a normalized matrix."""
    chunks = list(iter_doc_chunks(docs_root, page_paths))
    enc = encoder or default_encoder(model_name)
    content_hash = chunks_content_hash(chunks, model_name)
    if not chunks:
        return DocIndex(model_name, content_hash, [], np.zeros((0, 0), dtype=np.float32))
    vectors = _normalize(enc([c for _, c in chunks]))
    return DocIndex(model_name, content_hash, [p for p, _ in chunks], vectors)


def load_or_build(
    docs_root: Path,
    cache_dir: Path | None,
    *,
    model_name: str = DEFAULT_MODEL,
    encoder: Encoder | None = None,
    page_paths: list[str] | None = None,
) -> DocIndex:
    """Return a cached index if its content hash still matches, else (re)build + save.

    With `cache_dir=None` caching is disabled (always builds fresh).
    """
    want = chunks_content_hash(list(iter_doc_chunks(docs_root, page_paths)), model_name)
    if cache_dir is not None:
        cached = DocIndex.load(cache_dir)
        if cached is not None and cached.content_hash == want and cached.model_name == model_name:
            return cached
    index = build_index(
        docs_root, model_name=model_name, encoder=encoder, page_paths=page_paths
    )
    if cache_dir is not None:
        index.save(cache_dir)
    return index


def query_index(
    index: DocIndex,
    query_text: str,
    *,
    encoder: Encoder | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    floor: float = 0.2,
) -> list[tuple[str, float]]:
    """Rank pages by best-chunk cosine similarity to `query_text`.

    Returns up to `top_k` (page_path, score) pairs with score >= `floor`, highest
    first, one row per page (a page's score is its best chunk).
    """
    if index.vectors.shape[0] == 0:
        return []
    enc = encoder or default_encoder(model_name)
    qv = _normalize(enc([query_text]))[0]
    sims = index.vectors @ qv  # cosine: both sides normalized

    best: dict[str, float] = {}
    for page_path, sim in zip(index.page_paths, sims):
        val = float(sim)
        if page_path not in best or val > best[page_path]:
            best[page_path] = val

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    out: list[tuple[str, float]] = []
    for page_path, score in ranked:
        if score < floor:
            continue
        out.append((page_path, score))
        if len(out) >= top_k:
            break
    return out
