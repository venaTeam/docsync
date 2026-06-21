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
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np

if TYPE_CHECKING:
    from .models import RepoDigest

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# A copy of the default model can be vendored into the package (…/docsync/_models/<name>)
# so an air-gapped install loads it locally with no HuggingFace call. Populated by
# `scripts/vendor_model.py` and shipped in the wheel; absent in a lean dev checkout.
_MODELS_DIR = Path(__file__).resolve().parent / "_models"

# An encoder turns a list of texts into a 2-D float array (n_texts, dim).
Encoder = Callable[[list[str]], "np.ndarray"]

_META_FILE = "index.json"
_VECTORS_FILE = "vectors.npy"

# How many chars of a source file to fold into its embedded text (path + symbols carry
# most of the signal; a short excerpt sharpens it without making the encode pass costly).
_CODE_EXCERPT_CHARS = 600


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


def bundled_model_dir(model_id: str = DEFAULT_MODEL) -> Path | None:
    """Path to a copy of `model_id` vendored inside the package, or None if not shipped.

    Looks for `…/docsync/_models/<basename>` containing a `config.json` (the marker of a
    real sentence-transformers checkout). Present only when the wheel was built with the
    model vendored in — see `scripts/vendor_model.py`.
    """
    base = _MODELS_DIR / model_id.split("/")[-1]
    return base if (base / "config.json").exists() else None


def resolve_model_source(model_name: str) -> str:
    """Map a model name to what `SentenceTransformer(...)` should actually load.

    An explicit local path or a non-default id is returned unchanged. The default HF id
    resolves to the **bundled** copy when one ships with the package — so an air-gapped
    wheel loads the model locally with zero config and no HuggingFace call, while an
    online/dev install (no bundled model) still resolves the id over the network.
    """
    if Path(model_name).exists():  # already a local path the user pointed at
        return model_name
    if model_name == DEFAULT_MODEL:
        bundled = bundled_model_dir(DEFAULT_MODEL)
        if bundled is not None:
            return str(bundled)
    return model_name


def default_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Lazily build the production sentence-transformers encoder.

    Loads the bundled offline model when one ships with the package (see
    `resolve_model_source`); otherwise resolves `model_name` as usual. Raises ImportError
    if the optional `embeddings` extra isn't installed — callers treat that as "recall-net
    unavailable" and degrade to anchors only.
    """
    from sentence_transformers import SentenceTransformer  # optional extra

    model = SentenceTransformer(resolve_model_source(model_name))

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


def _save_index(cache_dir: Path, vectors: np.ndarray, meta: dict) -> None:
    """Persist an index's vectors (`.npy`) + JSON metadata to `cache_dir`."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / _VECTORS_FILE, vectors)
    (cache_dir / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_index_files(cache_dir: Path) -> tuple[dict, np.ndarray] | None:
    """Read `(meta, vectors)` from `cache_dir`, or None if absent/corrupt."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / _META_FILE
    vec_path = cache_dir / _VECTORS_FILE
    if not meta_path.exists() or not vec_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")), np.load(vec_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _cached_or_build(cache_dir, index_cls, want: str, model_name: str, build):
    """Return a cached index whose hash/model still match, else `build()` + save.

    `cache_dir=None` disables caching (always builds fresh). Shared by both the doc
    and code index loaders — only `build` and the index class differ.
    """
    if cache_dir is not None:
        cached = index_cls.load(cache_dir)
        if cached is not None and cached.content_hash == want and cached.model_name == model_name:
            return cached
    index = build()
    if cache_dir is not None:
        index.save(cache_dir)
    return index


@dataclass
class DocIndex:
    """A normalized embedding matrix over doc chunks, with provenance for caching."""

    model_name: str
    content_hash: str
    page_paths: list[str]  # one entry per row of `vectors`
    vectors: np.ndarray  # (n_chunks, dim), L2-normalized

    def save(self, cache_dir: Path) -> None:
        _save_index(
            cache_dir,
            self.vectors,
            {
                "model_name": self.model_name,
                "content_hash": self.content_hash,
                "page_paths": self.page_paths,
            },
        )

    @classmethod
    def load(cls, cache_dir: Path) -> "DocIndex | None":
        loaded = _load_index_files(cache_dir)
        if loaded is None:
            return None
        meta, vectors = loaded
        if "page_paths" not in meta:  # a code index saved here by mistake — ignore it
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
    return _cached_or_build(
        cache_dir, DocIndex, want, model_name,
        lambda: build_index(
            docs_root, model_name=model_name, encoder=encoder, page_paths=page_paths
        ),
    )


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


# ---------------------------------------------------------------------------
# Code-side index — the REVERSE direction, for `docsync infer`
# ---------------------------------------------------------------------------
#
# The doc index above embeds doc chunks and is queried by a code diff. Inference needs
# the opposite: embed source *units* (one vector per file) and query by an existing
# page's content, to shortlist the code a page likely documents. Same encoder, same
# normalized-cosine math, same on-disk cache shape — only the rows differ.


def _unit_text(repo: str, path: str, symbols: list[str], excerpt: str) -> str:
    """The text embedded for one source unit (units carry no body of their own)."""
    head = f"{repo}/{path}\n{' '.join(symbols)}".strip()
    return f"{head}\n{excerpt}".strip() if excerpt else head


def code_units_content_hash(
    unit_keys: list[tuple[str, str]], all_symbols: list[list[str]], model_name: str
) -> str:
    """Stable hash over (repo, path, symbols) + model — excludes excerpts on purpose.

    A source file's *body* edit that doesn't change its path/symbols shouldn't bust the
    code index (the anchor signal is path + symbols), so the excerpt is left out of the
    key — mirroring `chunks_content_hash` but for code units.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for (repo, path), symbols in zip(unit_keys, all_symbols):
        h.update(b"\x00")
        h.update(repo.encode("utf-8"))
        h.update(b"\x01")
        h.update(path.encode("utf-8"))
        for sym in symbols:
            h.update(b"\x02")
            h.update(sym.encode("utf-8"))
    return h.hexdigest()


@dataclass
class CodeIndex:
    """A normalized embedding matrix over source units, with provenance for caching."""

    model_name: str
    content_hash: str
    unit_keys: list[tuple[str, str]]  # (repo, path) per row of `vectors`
    vectors: np.ndarray  # (n_units, dim), L2-normalized

    def save(self, cache_dir: Path) -> None:
        _save_index(
            cache_dir,
            self.vectors,
            {
                "model_name": self.model_name,
                "content_hash": self.content_hash,
                # JSON has no tuples — store as [repo, path] pairs, restore on load.
                "unit_keys": [[repo, path] for repo, path in self.unit_keys],
            },
        )

    @classmethod
    def load(cls, cache_dir: Path) -> "CodeIndex | None":
        loaded = _load_index_files(cache_dir)
        if loaded is None:
            return None
        meta, vectors = loaded
        if "unit_keys" not in meta:  # a doc index saved here by mistake — ignore it
            return None
        return cls(
            model_name=meta["model_name"],
            content_hash=meta["content_hash"],
            unit_keys=[(repo, path) for repo, path in meta["unit_keys"]],
            vectors=vectors,
        )


def _iter_code_units(
    digests: list["RepoDigest"],
) -> Iterator[tuple[tuple[str, str], list[str], str]]:
    """Yield ((repo, path), symbols, excerpt) for every unit across all digests."""
    from . import ingest as ingest_mod  # local import keeps this module import-light

    for digest in digests:
        for unit in digest.units:
            excerpt = ingest_mod.read_excerpt(
                digest.root, unit.path, max_chars=_CODE_EXCERPT_CHARS
            )
            yield (digest.repo, unit.path), list(unit.symbols), excerpt


def build_code_index(
    digests: list["RepoDigest"],
    *,
    model_name: str = DEFAULT_MODEL,
    encoder: Encoder | None = None,
) -> CodeIndex:
    """Encode every source unit into a normalized matrix (one row per file)."""
    rows = list(_iter_code_units(digests))
    unit_keys = [key for key, _, _ in rows]
    all_symbols = [syms for _, syms, _ in rows]
    content_hash = code_units_content_hash(unit_keys, all_symbols, model_name)
    if not rows:
        return CodeIndex(model_name, content_hash, [], np.zeros((0, 0), dtype=np.float32))
    enc = encoder or default_encoder(model_name)
    texts = [_unit_text(repo, path, syms, exc) for (repo, path), syms, exc in rows]
    vectors = _normalize(enc(texts))
    return CodeIndex(model_name, content_hash, unit_keys, vectors)


def load_or_build_code_index(
    digests: list["RepoDigest"],
    cache_dir: Path | None,
    *,
    model_name: str = DEFAULT_MODEL,
    encoder: Encoder | None = None,
) -> CodeIndex:
    """Return a cached code index if its content hash still matches, else (re)build."""
    rows = list(_iter_code_units(digests))
    want = code_units_content_hash(
        [k for k, _, _ in rows], [s for _, s, _ in rows], model_name
    )
    return _cached_or_build(
        cache_dir, CodeIndex, want, model_name,
        lambda: build_code_index(digests, model_name=model_name, encoder=encoder),
    )


def query_code_index(
    index: CodeIndex,
    query_text: str,
    *,
    encoder: Encoder | None = None,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 5,
    floor: float = 0.2,
) -> list[tuple[tuple[str, str], float]]:
    """Rank source units by cosine similarity to `query_text` (a page's content).

    Returns up to `top_k` ((repo, path), score) pairs with score >= `floor`, highest
    first — the shortlist of code a page most likely documents.
    """
    if index.vectors.shape[0] == 0:
        return []
    enc = encoder or default_encoder(model_name)
    qv = _normalize(enc([query_text]))[0]
    sims = index.vectors @ qv
    ranked = sorted(
        zip(index.unit_keys, (float(s) for s in sims)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    out: list[tuple[tuple[str, str], float]] = []
    for key, score in ranked:
        if score < floor:
            continue
        out.append((key, score))
        if len(out) >= top_k:
            break
    return out
