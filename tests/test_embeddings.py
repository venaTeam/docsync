"""Embeddings recall-net tests — deterministic fake encoder, no sentence-transformers.

The encoder is injectable, so the whole index / cache / query path is exercised
with a tiny token-hashing encoder (texts sharing words get high cosine) — torch is
never imported here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from docsync import embeddings as emb
from docsync.impact import find_embedding_candidates, map_impact
from docsync.models import (
    CandidateSource,
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    FileStatus,
    Manifest,
    ManifestPage,
    ManifestSource,
)

DIM = 24


def _fake_encode(texts: list[str]) -> np.ndarray:
    """Bag-of-words hashing encoder: shared tokens -> overlapping dimensions."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % DIM
            out[i, h] += 1.0
    return out


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return _fake_encode(texts)


def _make_docs(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "services").mkdir(parents=True)
    (root / "services" / "metrics.mdx").write_text(
        "# Metrics\n\nThe prometheus counters live here.\n\n## Counters\n\nevents_in_counter\n",
        encoding="utf-8",
    )
    (root / "services" / "auth.mdx").write_text(
        "# Authentication\n\nIdentity backends and scopes for login.\n",
        encoding="utf-8",
    )
    return root


# --- chunking + hashing -----------------------------------------------------


def test_iter_doc_chunks_splits_on_headings_and_scans_tree(tmp_path: Path):
    root = _make_docs(tmp_path)
    chunks = list(emb.iter_doc_chunks(root))
    pages = {p for p, _ in chunks}
    assert pages == {"services/metrics.mdx", "services/auth.mdx"}
    # metrics.mdx has two headings -> two chunks.
    metrics_chunks = [c for p, c in chunks if p == "services/metrics.mdx"]
    assert len(metrics_chunks) == 2


def test_iter_doc_chunks_restricts_to_given_pages(tmp_path: Path):
    root = _make_docs(tmp_path)
    chunks = list(emb.iter_doc_chunks(root, ["services/auth.mdx"]))
    assert {p for p, _ in chunks} == {"services/auth.mdx"}


def test_content_hash_is_stable_and_content_sensitive(tmp_path: Path):
    root = _make_docs(tmp_path)
    c1 = list(emb.iter_doc_chunks(root))
    h1 = emb.chunks_content_hash(c1, emb.DEFAULT_MODEL)
    assert h1 == emb.chunks_content_hash(list(emb.iter_doc_chunks(root)), emb.DEFAULT_MODEL)
    # Different model name -> different key.
    assert h1 != emb.chunks_content_hash(c1, "other-model")


# --- build / persist / load -------------------------------------------------


def test_build_index_normalizes_rows(tmp_path: Path):
    root = _make_docs(tmp_path)
    idx = emb.build_index(root, encoder=_fake_encode)
    assert idx.vectors.shape[0] == len(idx.page_paths) > 0
    norms = np.linalg.norm(idx.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_load_roundtrip(tmp_path: Path):
    root = _make_docs(tmp_path)
    idx = emb.build_index(root, encoder=_fake_encode)
    cache = tmp_path / "cache"
    idx.save(cache)
    loaded = emb.DocIndex.load(cache)
    assert loaded is not None
    assert loaded.content_hash == idx.content_hash
    assert loaded.page_paths == idx.page_paths
    assert np.allclose(loaded.vectors, idx.vectors)


def test_load_returns_none_when_absent(tmp_path: Path):
    assert emb.DocIndex.load(tmp_path / "nope") is None


def test_doc_and_code_indices_ignore_each_others_cache(tmp_path: Path):
    # The two index kinds share file names in a cache dir. Loading the wrong kind must
    # return None (discriminating-key guard), never raise on the missing rows field.
    root = _make_docs(tmp_path)
    cache = tmp_path / "cache"
    emb.build_index(root, encoder=_fake_encode).save(cache)
    assert emb.CodeIndex.load(cache) is None  # doc index on disk, asked for code
    assert emb.DocIndex.load(cache) is not None


# --- caching behavior -------------------------------------------------------


def test_load_or_build_uses_cache_on_second_call(tmp_path: Path):
    root = _make_docs(tmp_path)
    cache = tmp_path / "cache"
    enc = CountingEncoder()
    emb.load_or_build(root, cache, encoder=enc)
    assert enc.calls == 1
    # Second call: docs unchanged -> cache hit -> encoder NOT invoked again.
    emb.load_or_build(root, cache, encoder=enc)
    assert enc.calls == 1


def test_load_or_build_rebuilds_when_docs_change(tmp_path: Path):
    root = _make_docs(tmp_path)
    cache = tmp_path / "cache"
    enc = CountingEncoder()
    emb.load_or_build(root, cache, encoder=enc)
    assert enc.calls == 1
    # Mutate a page -> content hash changes -> rebuild.
    (root / "services" / "auth.mdx").write_text("# Authentication\n\nNew text.\n", encoding="utf-8")
    emb.load_or_build(root, cache, encoder=enc)
    assert enc.calls == 2


def test_load_or_build_no_cache_dir_always_builds(tmp_path: Path):
    root = _make_docs(tmp_path)
    enc = CountingEncoder()
    emb.load_or_build(root, None, encoder=enc)
    emb.load_or_build(root, None, encoder=enc)
    assert enc.calls == 2


# --- query ------------------------------------------------------------------


def test_query_ranks_token_overlap_first(tmp_path: Path):
    root = _make_docs(tmp_path)
    idx = emb.build_index(root, encoder=_fake_encode)
    ranked = emb.query_index(idx, "prometheus counters events_in_counter",
                             encoder=_fake_encode, top_k=5, floor=0.0)
    assert ranked, "expected at least one match"
    assert ranked[0][0] == "services/metrics.mdx"


def test_query_floor_and_top_k(tmp_path: Path):
    root = _make_docs(tmp_path)
    idx = emb.build_index(root, encoder=_fake_encode)
    # An impossibly high floor filters everything.
    assert emb.query_index(idx, "login scopes", encoder=_fake_encode, floor=1.01) == []
    # top_k caps the count.
    ranked = emb.query_index(idx, "login scopes prometheus", encoder=_fake_encode,
                             top_k=1, floor=0.0)
    assert len(ranked) == 1


def test_query_empty_index_returns_empty():
    idx = emb.DocIndex("m", "h", [], np.zeros((0, 0), dtype=np.float32))
    assert emb.query_index(idx, "anything", encoder=_fake_encode) == []


# --- impact integration -----------------------------------------------------


def _diff_touching_metrics() -> CodeDiff:
    return CodeDiff(
        repo="keephq/keep-event-handler",
        base_sha="a", head_sha="b",
        files=[ChangedFile(
            path="src/core/metrics.py", status=FileStatus.MODIFIED,
            hunks=["@@ events_in_counter @@"], changed_symbols=["events_in_counter"],
        )],
    )


def test_find_embedding_candidates_with_injected_encoder(tmp_path: Path):
    root = _make_docs(tmp_path)
    cands = find_embedding_candidates(
        _diff_touching_metrics(), root, None, DocsyncConfig(),
        encoder=_fake_encode, cache_dir=tmp_path / "cache",
    )
    assert cands and cands[0].page_path == "services/metrics.mdx"
    assert cands[0].source == CandidateSource.EMBEDDING


def test_find_embedding_candidates_no_query_tokens_returns_empty(tmp_path: Path):
    root = _make_docs(tmp_path)
    empty = CodeDiff(repo="r", base_sha="a", head_sha="b")  # no files/symbols
    assert find_embedding_candidates(empty, root, None, DocsyncConfig(), encoder=_fake_encode) == []


def test_map_impact_embeddings_surfaces_unanchored_page(tmp_path: Path, monkeypatch):
    """The recall-net must reach a page that has NO manifest anchor at all."""
    root = _make_docs(tmp_path)
    # Manifest anchors only auth.mdx on an unrelated file; metrics.mdx is unanchored.
    manifest = Manifest(pages=[
        ManifestPage(
            path="services/auth.mdx",
            sources=[ManifestSource(
                repo="keephq/keep-event-handler",
                globs=["src/identitymanager/**"], symbols=["IdentityManagerTypes"],
            )],
        )
    ])
    # Inject the fake encoder where map_impact -> find_embedding_candidates reaches it.
    monkeypatch.setattr(emb, "default_encoder", lambda *_a, **_k: _fake_encode)

    impacted = map_impact(
        _diff_touching_metrics(), manifest, root, DocsyncConfig(),
        use_embeddings=True, cache_dir=tmp_path / "cache",
        client=_NoJudgeClient(),
    )
    paths = {p.page_path for p in impacted}
    assert "services/metrics.mdx" in paths  # surfaced purely by embeddings


class _NoJudgeClient:
    """Judge client that affirms any embedding candidate (so map_impact keeps it)."""

    class _M:
        def parse(self, *, output_format, **kwargs):
            from docsync.models import JudgeVerdict
            return type("R", (), {"parsed_output": JudgeVerdict(
                page_path="", affected=True, confidence=0.99, reason="fake-judge",
            )})()

    def __init__(self):
        self.messages = self._M()


@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_config_floor_is_respected(tmp_path: Path, floor: float):
    root = _make_docs(tmp_path)
    cfg = DocsyncConfig(embedding_floor=floor)
    cands = find_embedding_candidates(
        _diff_touching_metrics(), root, None, cfg, encoder=_fake_encode,
    )
    assert all(c.score >= floor for c in cands)


# ---------------------------------------------------------------------------
# Code-side index (docsync infer) — embed source units, query by page content
# ---------------------------------------------------------------------------


def _make_code(tmp_path: Path):
    """A tiny source checkout -> a RepoDigest, with real files so read_excerpt works."""
    from docsync.ingest import walk_repo

    root = tmp_path / "svc"
    (root / "src" / "routes").mkdir(parents=True)
    (root / "src" / "routes" / "alerts.py").write_text(
        "def get_alerts():\n    # prometheus alerting routes endpoint handler\n    return []\n",
        encoding="utf-8",
    )
    (root / "src" / "auth.py").write_text(
        "def login():\n    # identity oauth session cookie backends\n    return None\n",
        encoding="utf-8",
    )
    return walk_repo(root, repo="svc")


def test_build_code_index_one_row_per_unit_normalized(tmp_path: Path):
    digest = _make_code(tmp_path)
    idx = emb.build_code_index([digest], encoder=_fake_encode)
    assert idx.unit_keys == [("svc", "src/auth.py"), ("svc", "src/routes/alerts.py")]
    assert idx.vectors.shape[0] == len(idx.unit_keys) == 2
    norms = np.linalg.norm(idx.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_query_code_index_ranks_relevant_unit_first(tmp_path: Path):
    digest = _make_code(tmp_path)
    idx = emb.build_code_index([digest], encoder=_fake_encode)
    hits = emb.query_code_index(
        idx, "prometheus alerting routes endpoint handler", encoder=_fake_encode,
        top_k=1, floor=0.0,
    )
    assert hits and hits[0][0] == ("svc", "src/routes/alerts.py")


def test_query_code_index_respects_floor_and_top_k(tmp_path: Path):
    digest = _make_code(tmp_path)
    idx = emb.build_code_index([digest], encoder=_fake_encode)
    capped = emb.query_code_index(
        idx, "identity oauth session cookie", encoder=_fake_encode, top_k=1, floor=0.0
    )
    assert len(capped) <= 1
    floored = emb.query_code_index(
        idx, "identity oauth session cookie", encoder=_fake_encode, top_k=5, floor=0.99
    )
    assert all(score >= 0.99 for _, score in floored)


def test_code_index_cache_hit_and_rebuild(tmp_path: Path):
    digest = _make_code(tmp_path)
    cache = tmp_path / "cache"
    enc = CountingEncoder()
    emb.load_or_build_code_index([digest], cache, encoder=enc)
    assert enc.calls == 1
    # Same units -> cache hit, no re-encode.
    emb.load_or_build_code_index([digest], cache, encoder=enc)
    assert enc.calls == 1
    # A new symbol changes the content hash -> rebuild.
    (Path(digest.root) / "src" / "new.py").write_text("def fresh():\n    pass\n", encoding="utf-8")
    from docsync.ingest import walk_repo

    digest2 = walk_repo(digest.root, repo="svc")
    emb.load_or_build_code_index([digest2], cache, encoder=enc)
    assert enc.calls == 2


def test_code_units_content_hash_excludes_excerpt(tmp_path: Path):
    # Editing a file body (not its path/symbols) must NOT bust the code-index key.
    digest = _make_code(tmp_path)
    keys = [(digest.repo, u.path) for u in digest.units]
    syms = [list(u.symbols) for u in digest.units]
    h1 = emb.code_units_content_hash(keys, syms, emb.DEFAULT_MODEL)
    body = Path(digest.root) / digest.units[0].path
    body.write_text(body.read_text() + "\n# a new comment, no new symbol\n", encoding="utf-8")
    from docsync.ingest import walk_repo

    digest2 = walk_repo(digest.root, repo="svc")
    keys2 = [(digest2.repo, u.path) for u in digest2.units]
    syms2 = [list(u.symbols) for u in digest2.units]
    assert emb.code_units_content_hash(keys2, syms2, emb.DEFAULT_MODEL) == h1


# --- Bundled offline model resolution (air-gapped, no HuggingFace) -----------------


def _vendor_fake_model(models_dir: Path, model_id: str = emb.DEFAULT_MODEL) -> Path:
    """Create a minimal stand-in for a vendored sentence-transformers checkout."""
    d = models_dir / model_id.split("/")[-1]
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    return d


def test_bundled_model_dir_found_when_vendored(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(emb, "_MODELS_DIR", tmp_path)
    assert emb.bundled_model_dir() is None  # nothing vendored yet
    vendored = _vendor_fake_model(tmp_path)
    assert emb.bundled_model_dir() == vendored


def test_resolve_model_source_prefers_bundled_default(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(emb, "_MODELS_DIR", tmp_path)
    # No bundled copy -> the HF id is returned unchanged (online/dev resolves it).
    assert emb.resolve_model_source(emb.DEFAULT_MODEL) == emb.DEFAULT_MODEL
    # Vendored copy present -> resolve to the local path (offline, no HF call).
    vendored = _vendor_fake_model(tmp_path)
    assert emb.resolve_model_source(emb.DEFAULT_MODEL) == str(vendored)


def test_resolve_model_source_honors_explicit_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(emb, "_MODELS_DIR", tmp_path)
    _vendor_fake_model(tmp_path)  # bundled default exists...
    custom = tmp_path / "my-own-model"
    custom.mkdir()
    # ...but an explicit local path the user configured wins and is returned as-is.
    assert emb.resolve_model_source(str(custom)) == str(custom)


def test_resolve_model_source_passes_through_unknown_id(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(emb, "_MODELS_DIR", tmp_path)
    _vendor_fake_model(tmp_path)  # only the DEFAULT_MODEL is vendored
    assert emb.resolve_model_source("some/other-model") == "some/other-model"
