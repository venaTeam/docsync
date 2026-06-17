"""Tests for the `docsync bootstrap` orchestrator (`docsync.bootstrap`).

No network: a fake client whose `.messages.parse(...)` returns a scripted
`DocPlan` for the plan call and an `AuthoredPage` for each author call, keyed by
the `output_format` it's asked for. Exercises planning + collision dedupe, the
spend cap, parallel-author determinism, pool resilience, metering, and the
emit step (file write + nav registration + manifest merge).
"""

from __future__ import annotations

import json
from pathlib import Path

from docsync.bootstrap import (
    DEFAULT_NAV_GROUP,
    plan_docs,
    run_bootstrap,
    write_bootstrap,
)
from docsync.config import load_manifest
from docsync.models import (
    AuthoredPage,
    DocPlan,
    DocsyncConfig,
    PlannedPage,
)

# A minimal valid page body the author stage "returns" — passes validate_new_page.
_VALID_PAGE = (
    "---\ntitle: Alerts API\ndescription: The alerts route group.\n---\n\n"
    "# Alerts API\n\n"
    + ("This page documents the alerts endpoints in detail. " * 8)
    + "\n\n<Note>Authoritative reference.</Note>\n"
)


# ---------------------------------------------------------------------------
# Fakes
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
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unexpected output_format {fmt!r}")
        return _Resp()


class FakeClient:
    def __init__(self, plan: DocPlan, page_text: str = _VALID_PAGE):
        self.messages = _FakeMessages(plan, page_text)


def _src_repo(tmp_path: Path) -> Path:
    """A tiny read-only-ish source repo with two Python files."""
    repo = tmp_path / "svc"
    (repo / "src" / "routes").mkdir(parents=True, exist_ok=True)
    (repo / "src" / "routes" / "alerts.py").write_text(
        "def get_alerts():\n    return []\n\n\nclass AlertManager:\n    pass\n"
    )
    (repo / "src" / "routes" / "incidents.py").write_text("def list_incidents():\n    return []\n")
    return repo


def _docs_repo(tmp_path: Path, *, nav_pages: list[str] | None = None) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    nav = {"navigation": {"groups": [{"group": "Existing", "pages": nav_pages or []}]}}
    (docs / "docs.json").write_text(json.dumps(nav, indent=2) + "\n")
    return docs


def _plan(*page_paths: str) -> DocPlan:
    return DocPlan(
        pages=[
            PlannedPage(
                page_path=p,
                title=f"Title {p}",
                summary="cover it",
                source_paths=["src/routes/alerts.py"],
                symbols=["get_alerts"],
            )
            for p in page_paths
        ]
    )


# ---------------------------------------------------------------------------
# plan_docs — dedupe + cap
# ---------------------------------------------------------------------------


def test_plan_dedupes_against_existing_disk_page(tmp_path):
    docs = _docs_repo(tmp_path)
    (docs / "reference").mkdir()
    (docs / "reference" / "alerts.mdx").write_text("# existing\n")
    client = FakeClient(_plan("reference/alerts.mdx", "reference/incidents.mdx"))

    plan, skipped = plan_docs(_src_repo_digest(tmp_path), docs, DocsyncConfig(), client=client)

    kept = [p.page_path for p in plan.pages]
    assert kept == ["reference/incidents.mdx"]
    assert "reference/alerts.mdx" in skipped


def test_plan_dedupes_against_existing_nav_route(tmp_path):
    # The route "services/api-gateway" is already in the nav -> drop the colliding page.
    docs = _docs_repo(tmp_path, nav_pages=["services/api-gateway"])
    client = FakeClient(_plan("services/api-gateway.mdx", "reference/incidents.mdx"))

    plan, skipped = plan_docs(_src_repo_digest(tmp_path), docs, DocsyncConfig(), client=client)

    assert [p.page_path for p in plan.pages] == ["reference/incidents.mdx"]
    assert "services/api-gateway.mdx" in skipped


def test_plan_dedupes_intra_plan_duplicates(tmp_path):
    docs = _docs_repo(tmp_path)
    # Same page proposed twice (once without extension) -> one survives.
    client = FakeClient(_plan("reference/alerts", "reference/alerts.mdx"))

    plan, skipped = plan_docs(_src_repo_digest(tmp_path), docs, DocsyncConfig(), client=client)

    assert [p.page_path for p in plan.pages] == ["reference/alerts.mdx"]
    assert skipped == ["reference/alerts.mdx"]


def test_plan_normalizes_missing_extension(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts"))
    plan, _ = plan_docs(_src_repo_digest(tmp_path), docs, DocsyncConfig(), client=client)
    assert plan.pages[0].page_path == "reference/alerts.mdx"


def test_plan_cap_applied_after_dedupe(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("a.mdx", "b.mdx", "c.mdx"))
    plan, _ = plan_docs(_src_repo_digest(tmp_path), docs, DocsyncConfig(), client=client, max_pages=2)
    assert [p.page_path for p in plan.pages] == ["a.mdx", "b.mdx"]


# ---------------------------------------------------------------------------
# run_bootstrap — author + validate
# ---------------------------------------------------------------------------


def _src_repo_digest(tmp_path: Path):
    """Build a RepoDigest from the fixture source repo (used by plan_docs tests)."""
    from docsync.ingest import walk_repo

    return walk_repo(_src_repo(tmp_path), repo="svc")


def test_run_bootstrap_authors_and_validates(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts.mdx", "reference/incidents.mdx"))

    result = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client)

    authored = result.authored()
    assert len(authored) == 2
    assert all(o.applied and o.new_content for o in authored)
    # Cost was metered (plan + 2 author calls).
    assert result.usage is not None and result.usage.calls == 3
    # Per-stage attribution separates plan from author.
    stages = {m.stage for m in result.usage.by_model}
    assert "plan" in stages and "author" in stages


def test_run_bootstrap_plan_only_skips_authoring(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts.mdx"))
    result = run_bootstrap(
        _src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client, plan_only=True
    )
    assert result.plan.pages and not result.outcomes
    # Only the planner was called.
    assert result.usage.calls == 1
    assert {m.stage for m in result.usage.by_model} == {"plan"}


def test_run_bootstrap_invalid_page_dropped(tmp_path):
    docs = _docs_repo(tmp_path)
    # Author returns a too-short stub with no frontmatter -> validation drops it.
    client = FakeClient(_plan("reference/alerts.mdx"), page_text="too short")
    result = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client)
    assert result.authored() == []
    assert "dropped by validation" in result.outcomes[0].note


def test_run_bootstrap_parallel_preserves_order(tmp_path):
    docs = _docs_repo(tmp_path)
    paths = [f"reference/p{i}.mdx" for i in range(6)]
    client = FakeClient(_plan(*paths))
    cfg = DocsyncConfig()
    cfg.max_parallel_requests = 4
    result = run_bootstrap(_src_repo(tmp_path), docs, cfg, repo="svc", client=client)
    assert [o.page_path for o in result.outcomes] == paths  # map() preserves input order


# ---------------------------------------------------------------------------
# write_bootstrap — emit (files + nav + manifest)
# ---------------------------------------------------------------------------


def test_write_bootstrap_emits_pages_nav_and_manifest(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts.mdx", "reference/incidents.mdx"))
    result = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client)

    touched = write_bootstrap(result, docs, DocsyncConfig())

    # Pages written to disk.
    assert (docs / "reference" / "alerts.mdx").exists()
    assert (docs / "reference" / "incidents.mdx").exists()
    # Nav got a new group with the extensionless routes.
    nav = json.loads((docs / "docs.json").read_text())
    groups = {g["group"]: g["pages"] for g in nav["navigation"]["groups"]}
    assert DEFAULT_NAV_GROUP in groups
    assert "reference/alerts" in groups[DEFAULT_NAV_GROUP]
    assert "Existing" in groups  # untouched
    # Manifest anchors emitted and round-trip.
    manifest = load_manifest(docs)
    paths = {p.path for p in manifest.pages}
    assert {"reference/alerts.mdx", "reference/incidents.mdx"} <= paths
    page = manifest.page("reference/alerts.mdx")
    assert page.sources[0].repo == "svc"
    assert "src/routes/alerts.py" in page.sources[0].globs
    # Touched list includes pages, nav, and manifest.
    assert "docs.json" in touched
    assert ".docsync/manifest.yml" in touched


def test_write_bootstrap_idempotent_rerun(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts.mdx"))
    result = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client)
    write_bootstrap(result, docs, DocsyncConfig())
    nav_after_first = (docs / "docs.json").read_text()

    # A second run: plan_docs now sees the page on disk + in nav -> nothing to do.
    client2 = FakeClient(_plan("reference/alerts.mdx"))
    result2 = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client2)
    assert result2.authored() == []
    assert "reference/alerts.mdx" in result2.skipped
    # Nav unchanged by the no-op second pass.
    assert (docs / "docs.json").read_text() == nav_after_first


def test_write_bootstrap_skips_existing_file_without_force(tmp_path):
    docs = _docs_repo(tmp_path)
    client = FakeClient(_plan("reference/alerts.mdx"))
    result = run_bootstrap(_src_repo(tmp_path), docs, DocsyncConfig(), repo="svc", client=client)
    # Pre-create the target file with sentinel content.
    (docs / "reference").mkdir()
    (docs / "reference" / "alerts.mdx").write_text("SENTINEL\n")
    write_bootstrap(result, docs, DocsyncConfig(), force=False)
    assert (docs / "reference" / "alerts.mdx").read_text() == "SENTINEL\n"  # not overwritten
