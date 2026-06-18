"""Tests for the IA/narrative `docsync bootstrap` orchestrator (`docsync.bootstrap`).

No network: a fake client whose `.messages.parse(...)` returns a scripted `DocPlan`
for the plan call and an `AuthoredPage` for each author call, keyed by `output_format`.
Covers multi-repo ingest, the sectioned IA planner (dedupe/cap/ordering), kind-specific
authoring + cross-repo excerpts, ordered nav emit, and judge-gated manifest anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

from docsync.bootstrap import (
    _gather_excerpts,
    _repo_units,
    author_page,
    build_author_prompt,
    plan_docs,
    run_bootstrap,
    write_bootstrap,
)
from docsync.config import load_manifest
from docsync.ingest import walk_repos
from docsync.models import (
    AuthoredPage,
    DocPlan,
    DocsyncConfig,
    PlannedPage,
    PlannedSource,
)

_VALID_PAGE = (
    "---\ntitle: A Page\ndescription: A generated page.\n---\n\n# A Page\n\n"
    + ("This page documents the subsystem in detail. " * 8)
    + "\n\n<Note>Authoritative.</Note>\n"
)


# ---------------------------------------------------------------------------
# Fakes & fixtures
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, plan: DocPlan, page_text: str):
        self._plan = plan
        self._page_text = page_text
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        fmt = kwargs.get("output_format")

        class _Resp:
            usage = {"input_tokens": 10, "output_tokens": 5}

        if fmt is DocPlan:
            _Resp.parsed_output = self._plan
        elif fmt is AuthoredPage:
            _Resp.parsed_output = AuthoredPage(content=self._page_text)
        else:  # pragma: no cover
            raise AssertionError(f"unexpected output_format {fmt!r}")
        return _Resp()


class FakeClient:
    def __init__(self, plan: DocPlan, page_text: str = _VALID_PAGE):
        self.messages = _FakeMessages(plan, page_text)


def _make_repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    repo = tmp_path / name
    for rel, body in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(body)
    return repo


def _gw(tmp_path: Path) -> Path:
    return _make_repo(
        tmp_path, "gw",
        {
            "src/routes/alerts.py": "def get_alerts():\n    return []\n\nclass AlertManager:\n    pass\n",
            "src/routes/incidents.py": "def list_incidents():\n    return []\n",
        },
    )


def _eh(tmp_path: Path) -> Path:
    return _make_repo(
        tmp_path, "eh", {"src/consumer.py": "def consume():\n    return None\n"}
    )


def _docs_repo(tmp_path: Path, *, nav_pages: list[str] | None = None) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    nav = {
        "name": "Docs", "theme": "mint", "colors": {"primary": "#000"},
        "navigation": {"groups": [{"group": "Existing", "pages": nav_pages or []}]},
    }
    (docs / "docs.json").write_text(json.dumps(nav, indent=2) + "\n")
    return docs


def _page(path, *, section="Reference", kind="reference", order=0, repo="gw",
          globs=("src/routes/alerts.py",), symbols=("get_alerts",)) -> PlannedPage:
    return PlannedPage(
        page_path=path, title=f"Title {path}", kind=kind, section=section, order=order,
        summary="cover it",
        sources=[PlannedSource(repo=repo, globs=list(globs), symbols=list(symbols))],
    )


def _digests(tmp_path: Path):
    return walk_repos([("gw", _gw(tmp_path))])


# ---------------------------------------------------------------------------
# plan_docs — dedupe, cap, sections
# ---------------------------------------------------------------------------


def test_plan_dedupes_existing_disk_and_nav(tmp_path):
    docs = _docs_repo(tmp_path, nav_pages=["services/api-gateway"])
    (docs / "reference").mkdir()
    (docs / "reference" / "alerts.mdx").write_text("# existing\n")
    client = FakeClient(DocPlan(pages=[
        _page("reference/alerts.mdx"),               # collides on disk
        _page("services/api-gateway.mdx"),           # collides in nav
        _page("reference/incidents.mdx"),            # survives
    ]))
    plan, skipped = plan_docs(_digests(tmp_path), docs, DocsyncConfig(), client=client)
    assert [p.page_path for p in plan.pages] == ["reference/incidents.mdx"]
    assert set(skipped) == {"reference/alerts.mdx", "services/api-gateway.mdx"}


def test_plan_cap_after_dedupe_and_normalizes_ext(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(DocPlan(pages=[_page("a"), _page("b"), _page("c")]))
    plan, _ = plan_docs(_digests(tmp_path), docs, DocsyncConfig(), client=client, max_pages=2)
    assert [p.page_path for p in plan.pages] == ["a.mdx", "b.mdx"]  # ext added, capped


def test_plan_ordered_sections_follow_reading_flow(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(DocPlan(pages=[
        _page("reference/a.mdx", section="Reference", order=0),
        _page("getting-started/intro.mdx", section="Getting Started", kind="guide", order=0),
        _page("architecture/flow.mdx", section="Architecture", kind="concept", order=0),
    ]))
    plan, _ = plan_docs(_digests(tmp_path), docs, DocsyncConfig(), client=client)
    assert [s for s, _ in plan.ordered_sections()] == [
        "Getting Started", "Architecture", "Reference",
    ]


# ---------------------------------------------------------------------------
# Multi-repo ingest + cross-repo authoring
# ---------------------------------------------------------------------------


def test_gather_excerpts_resolves_across_repos(tmp_path):
    digests = walk_repos([("gw", _gw(tmp_path)), ("eh", _eh(tmp_path))])
    units = _repo_units(digests)
    page = _page("concepts/flow.mdx", kind="concept",
                 repo="eh", globs=("src/*.py",), symbols=())
    excerpts = _gather_excerpts(page, units)
    assert excerpts and all(label.startswith("eh/") for label, _ in excerpts)
    assert "def consume" in excerpts[0][1]


def test_author_prompt_is_kind_specific(tmp_path):
    concept = _page("c.mdx", kind="concept")
    reference = _page("r.mdx", kind="reference")
    sys_c, _ = build_author_prompt(concept, [("gw/x.py", "code")])
    sys_r, _ = build_author_prompt(reference, [("gw/x.py", "code")])
    assert "CONCEPT" in sys_c and "narrative" in sys_c
    assert "REFERENCE" in sys_r
    assert sys_c != sys_r


def test_run_bootstrap_multi_repo_meters_plan_and_author(tmp_path):
    docs = _docs_repo(tmp_path)
    repos = [("gw", _gw(tmp_path)), ("eh", _eh(tmp_path))]
    client = FakeClient(DocPlan(pages=[
        _page("getting-started/intro.mdx", section="Getting Started", kind="guide"),
        _page("reference/alerts.mdx"),
    ]))
    result = run_bootstrap(repos, docs, DocsyncConfig(), client=client)
    assert len(result.authored()) == 2
    assert result.repo == "gw, eh"
    assert result.usage.calls == 3  # 1 plan + 2 author
    assert {m.stage for m in result.usage.by_model} == {"plan", "author"}


def test_run_bootstrap_plan_only_skips_authoring(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(DocPlan(pages=[_page("reference/a.mdx")]))
    result = run_bootstrap(
        [("gw", _gw(tmp_path))], docs, DocsyncConfig(), client=client, plan_only=True
    )
    assert result.plan.pages and not result.outcomes
    assert {m.stage for m in result.usage.by_model} == {"plan"}


def test_run_bootstrap_invalid_page_dropped(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(DocPlan(pages=[_page("reference/a.mdx")]), page_text="too short")
    result = run_bootstrap([("gw", _gw(tmp_path))], docs, DocsyncConfig(), client=client)
    assert result.authored() == []
    assert "dropped by validation" in result.outcomes[0].note


def test_run_bootstrap_parallel_preserves_order(tmp_path):
    docs = _docs_repo(tmp_path)
    paths = [f"reference/p{i}.mdx" for i in range(6)]
    client = FakeClient(DocPlan(pages=[_page(p) for p in paths]))
    cfg = DocsyncConfig()
    cfg.max_parallel_requests = 4
    result = run_bootstrap([("gw", _gw(tmp_path))], docs, cfg, client=client)
    assert [o.page_path for o in result.outcomes] == paths


# ---------------------------------------------------------------------------
# write_bootstrap — ordered nav + judge-gated manifest
# ---------------------------------------------------------------------------


def test_write_bootstrap_orders_nav_and_gates_manifest(tmp_path):
    docs = _docs_repo(tmp_path)
    repos = [("gw", _gw(tmp_path))]
    client = FakeClient(DocPlan(pages=[
        _page("reference/alerts.mdx", section="Reference", order=0),
        _page("getting-started/intro.mdx", section="Getting Started", kind="guide", order=0),
        _page("architecture/flow.mdx", section="Architecture", kind="concept", order=0,
              globs=("src/routes/*.py",), symbols=()),
    ]))
    result = run_bootstrap(repos, docs, DocsyncConfig(), client=client)
    touched = write_bootstrap(result, docs, DocsyncConfig())

    # Nav groups appear in reading-flow order (new sections after the existing one).
    nav = json.loads((docs / "docs.json").read_text())
    order = [g["group"] for g in nav["navigation"]["groups"]]
    assert order == ["Existing", "Getting Started", "Architecture", "Reference"]
    # ensure_valid_docs_json kept colors present.
    assert "colors" in nav

    # Manifest: concept/guide pages are judge_required; reference is not.
    manifest = load_manifest(docs)
    jr = {p.path: p.judge_required for p in manifest.pages}
    assert jr["architecture/flow.mdx"] is True
    assert jr["getting-started/intro.mdx"] is True
    assert jr["reference/alerts.mdx"] is False
    assert ".docsync/manifest.yml" in touched
    assert "docs.json" in touched


def test_ensure_valid_docs_json_seeds_empty_scaffold(tmp_path):
    # An empty scaffold (no docs.json) gets a valid one so the site renders.
    docs = tmp_path / "scratch"
    docs.mkdir()
    client = FakeClient(DocPlan(pages=[_page("reference/a.mdx")]))
    result = run_bootstrap([("gw", _gw(tmp_path))], docs, DocsyncConfig(), client=client)
    write_bootstrap(result, docs, DocsyncConfig())
    nav = json.loads((docs / "docs.json").read_text())
    assert "colors" in nav  # required field seeded
    assert any(g["group"] == "Reference" for g in nav["navigation"]["groups"])


def test_author_page_uses_edit_model(tmp_path):
    client = FakeClient(DocPlan(pages=[]))
    units = _repo_units(walk_repos([("gw", _gw(tmp_path))]))
    author_page(_page("r.mdx"), units, DocsyncConfig(), client=client)
    call = client.messages.calls[0]
    assert call["model"] == DocsyncConfig().models.edit_model
    assert call["output_format"] is AuthoredPage
