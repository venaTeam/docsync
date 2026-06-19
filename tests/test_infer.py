"""Tests for `docsync infer` — manifest inference for an existing docs site.

Fully offline: a deterministic fake encoder (no sentence-transformers) and a page-aware
fake judge (no network). The embedding floor is set to 0 so the shortlist always reaches
the judge — encoder accuracy is exercised separately in test_embeddings.py; here we test
the judge → honesty-gate → emit path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from docsync import infer as infer_mod
from docsync.config import load_manifest, merge_manifest_pages
from docsync.models import (
    DocsyncConfig,
    InferredAnchors,
    InferredSource,
    ManifestPage,
    ManifestSource,
)
from docsync.scaffold import doctor

DIM = 24


def _fake_encode(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % DIM
            out[i, h] += 1.0
    return out


class FakeJudge:
    """Returns a scripted InferredAnchors per page, keyed off the `# Page:` header."""

    def __init__(self, by_page: dict[str, InferredAnchors]):
        self.by_page = by_page
        self.messages = self

    def parse(self, *, output_format, messages, **kwargs):
        user = messages[0]["content"]
        page = user.split("# Page: ", 1)[1].splitlines()[0].strip()
        anchors = self.by_page.get(page) or InferredAnchors(page_path=page, confidence=0.0)
        # A usage dict so MeteredClient records the call (mirrors the CLI-backend envelope).
        usage = {"input_tokens": 100, "output_tokens": 20}
        return type("R", (), {"parsed_output": anchors, "usage": usage})()


def _cfg(**kw) -> DocsyncConfig:
    base = dict(embedding_floor=0.0, embedding_top_k=10, max_parallel_requests=1)
    base.update(kw)
    return DocsyncConfig(**base)


@pytest.fixture()
def repo(tmp_path: Path):
    """A docs repo (pages, no manifest) + a source checkout. Returns (docs, src)."""
    docs = tmp_path / "docs"
    (docs / "reference").mkdir(parents=True)
    (docs / "concepts").mkdir(parents=True)
    (docs / ".docsync").mkdir()
    (docs / "reference" / "alerts.mdx").write_text(
        "---\ntitle: Alerts API\ndescription: alert routes\n---\n\n# Alerts\n\nThe alert "
        "endpoints and the get_alerts handler.\n",
        encoding="utf-8",
    )
    (docs / "concepts" / "overview.mdx").write_text(
        "---\ntitle: Overview\ndescription: how the service fits together\n---\n\n# Overview\n\n"
        "A narrative tour of the routing subsystem.\n",
        encoding="utf-8",
    )

    src = tmp_path / "svc"
    (src / "src" / "routes").mkdir(parents=True)
    (src / "src" / "routes" / "alerts.py").write_text(
        "def get_alerts():\n    # prometheus alerting routes\n    return []\n", encoding="utf-8"
    )
    (src / "src" / "routes" / "incidents.py").write_text(
        "def get_incidents():\n    # incident routes\n    return []\n", encoding="utf-8"
    )
    return docs, src


def _run(docs, src, by_page, cfg=None):
    return infer_mod.run_infer(
        [("svc", src)], docs, cfg or _cfg(),
        client=FakeJudge(by_page), encoder=_fake_encode,
    )


# ---------------------------------------------------------------------------
# Happy path + idempotency
# ---------------------------------------------------------------------------


def test_anchors_pages_and_writes_manifest(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.9,
            sources=[InferredSource(
                repo="svc", globs=["src/routes/alerts.py"], symbols=["get_alerts"]
            )],
        ),
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", kind="concept", confidence=0.8,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    result = _run(docs, src, by_page)
    assert {p.page_path for p in result.anchored()} == {
        "reference/alerts.mdx", "concepts/overview.mdx"
    }

    added = infer_mod.write_infer(result, docs, _cfg())
    assert set(added) == {"reference/alerts.mdx", "concepts/overview.mdx"}
    manifest = load_manifest(docs)
    jr = {p.path: p.judge_required for p in manifest.pages}
    assert jr["concepts/overview.mdx"] is True       # concept -> judged
    assert jr["reference/alerts.mdx"] is False        # reference -> autopass


def test_already_anchored_pages_are_skipped(repo):
    docs, src = repo
    # Pre-anchor one page; infer must skip it and only examine the other.
    merge_manifest_pages(docs, [ManifestPage(
        path="reference/alerts.mdx",
        sources=[ManifestSource(repo="svc", globs=["src/routes/alerts.py"])],
    )])
    by_page = {
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", kind="concept", confidence=0.8,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    result = _run(docs, src, by_page)
    assert "reference/alerts.mdx" in result.skipped_already_anchored
    assert [p.page_path for p in result.pages] == ["concepts/overview.mdx"]


def test_rerun_is_a_noop(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.9,
            sources=[InferredSource(repo="svc", globs=["src/routes/alerts.py"])],
        ),
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", kind="concept", confidence=0.8,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    infer_mod.write_infer(_run(docs, src, by_page), docs, _cfg())
    # Second pass: both pages now anchored -> nothing to examine, nothing to add.
    second = _run(docs, src, by_page)
    assert second.pages == []
    assert infer_mod.write_infer(second, docs, _cfg()) == []


# ---------------------------------------------------------------------------
# Honesty gate
# ---------------------------------------------------------------------------


def test_dead_glob_becomes_no_match(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.95,
            sources=[InferredSource(repo="svc", globs=["src/ghost/*.py"])],  # matches nothing
        ),
    }
    result = _run(docs, src, by_page)
    page = next(p for p in result.pages if p.page_path == "reference/alerts.mdx")
    assert page.status == "no_match"
    assert page.sources == []


def test_absent_symbol_is_dropped(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.9,
            sources=[InferredSource(
                repo="svc", globs=["src/routes/alerts.py"],
                symbols=["get_alerts", "does_not_exist"],
            )],
        ),
    }
    result = _run(docs, src, by_page)
    page = next(p for p in result.anchored() if p.page_path == "reference/alerts.mdx")
    assert page.sources[0].symbols == ["get_alerts"]  # the bogus symbol is gone


def test_unknown_repo_is_dropped(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.9,
            sources=[InferredSource(repo="not-ingested", globs=["src/routes/alerts.py"])],
        ),
    }
    result = _run(docs, src, by_page)
    page = next(p for p in result.pages if p.page_path == "reference/alerts.mdx")
    assert page.status == "no_match"


def test_written_manifest_passes_doctor(repo):
    """The closed loop: everything infer writes must pass `docsync doctor` as-is."""
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.9,
            sources=[InferredSource(
                repo="svc", globs=["src/routes/alerts.py"], symbols=["get_alerts"]
            )],
        ),
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", kind="concept", confidence=0.8,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    infer_mod.write_infer(_run(docs, src, by_page), docs, _cfg())
    report = doctor(docs, {"svc": src})
    assert report.ok, (report.missing_pages, report.dead_globs)
    assert report.dead_globs == []
    assert report.missing_symbols == []


# ---------------------------------------------------------------------------
# Confidence gate + metering
# ---------------------------------------------------------------------------


def test_low_confidence_not_written(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", kind="reference", confidence=0.2,
            sources=[InferredSource(repo="svc", globs=["src/routes/alerts.py"])],
        ),
    }
    cfg = _cfg(judge_confidence_threshold=0.5)
    result = infer_mod.run_infer(
        [("svc", src)], docs, cfg, client=FakeJudge(by_page), encoder=_fake_encode,
    )
    page = next(p for p in result.pages if p.page_path == "reference/alerts.mdx")
    assert page.status == "low_confidence"
    assert infer_mod.write_infer(result, docs, cfg) == []


def test_metering_counts_one_call_per_page(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", confidence=0.9,
            sources=[InferredSource(repo="svc", globs=["src/routes/alerts.py"])],
        ),
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", confidence=0.9,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    result = _run(docs, src, by_page)
    assert len(result.pages) == 2
    assert result.usage is not None
    assert result.usage.calls == 2  # one judge call per page
    assert all(m.stage == "infer" for m in result.usage.by_model)


def test_max_pages_caps_work(repo):
    docs, src = repo
    by_page = {
        "reference/alerts.mdx": InferredAnchors(
            page_path="reference/alerts.mdx", confidence=0.9,
            sources=[InferredSource(repo="svc", globs=["src/routes/alerts.py"])],
        ),
        "concepts/overview.mdx": InferredAnchors(
            page_path="concepts/overview.mdx", confidence=0.9,
            sources=[InferredSource(repo="svc", globs=["src/routes/*.py"])],
        ),
    }
    result = infer_mod.run_infer(
        [("svc", src)], docs, _cfg(), max_pages=1,
        client=FakeJudge(by_page), encoder=_fake_encode,
    )
    assert len(result.pages) == 1


# ---------------------------------------------------------------------------
# Graceful degradation + glob collapse
# ---------------------------------------------------------------------------


def test_missing_embeddings_extra_raises_importerror(repo, monkeypatch):
    docs, src = repo
    monkeypatch.setattr(
        infer_mod.emb, "default_encoder",
        lambda *a, **k: (_ for _ in ()).throw(ImportError("no extra")),
    )
    with pytest.raises(ImportError):
        infer_mod.run_infer([("svc", src)], docs, _cfg(), client=FakeJudge({}), encoder=None)


def test_inferred_anchors_routes_through_json_backend():
    # Guard: InferredAnchors must NOT look like a single-required-str-field document
    # model, or the claude-code backend stuffs the whole JSON reply into page_path and
    # every anchor comes back empty (the dogfood bug).
    from docsync.llm_backends import _single_text_field

    assert _single_text_field(InferredAnchors) is None


def test_collapse_globs_exact_sibling_cover():
    paths = ["src/routes/a.py", "src/routes/b.py", "src/cli.py"]
    # Both siblings present and they are the ONLY .py in src/routes -> collapse.
    assert infer_mod._collapse_globs(
        ["src/routes/a.py", "src/routes/b.py"], paths
    ) == ["src/routes/*.py"]


def test_collapse_globs_keeps_literals_when_not_exact_cover():
    paths = ["src/routes/a.py", "src/routes/b.py", "src/routes/c.py"]
    # Only a + b proposed but c also exists -> collapsing would broaden; keep literals.
    out = infer_mod._collapse_globs(["src/routes/a.py", "src/routes/b.py"], paths)
    assert set(out) == {"src/routes/a.py", "src/routes/b.py"}
