"""Integration test for the orchestration: diff -> impact -> edits -> validate.

Uses a fake Anthropic client so no network is touched. Exercises the real
anchor mapping, edit application, and validation gates end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync import pipeline
from docsync.config import load_config, load_manifest, save_cursors
from docsync.critique import CritiqueVerdict
from docsync.models import (
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    EditOp,
    FileStatus,
    JudgeVerdict,
    Manifest,
    PageEdit,
)

PAGE = """\
---
title: "API Gateway"
description: "REST API service"
---

# API Gateway

| Method | Path | Purpose |
| --- | --- | --- |
| POST | /alerts | Ingest an alert |

The service is configured via `setup_routers`.
"""

MANIFEST_YML = """\
pages:
  - path: services/api-gateway.mdx
    sources:
      - repo: keephq/keep-api-gateway
        globs: ["src/routes/router_setup.py"]
        symbols: ["setup_routers"]
    max_diff_lines: 40
"""


class _FakeMessages:
    def __init__(self, outputs):
        self._outputs = outputs

    def parse(self, *, output_format, **kwargs):
        out = self._outputs[output_format.__name__]
        return type("Resp", (), {"parsed_output": out})()


class FakeClient:
    """Returns a scripted JudgeVerdict for the judge and PageEdit for the editor."""

    def __init__(self, verdict: JudgeVerdict, edit: PageEdit):
        self.messages = _FakeMessages({"JudgeVerdict": verdict, "PageEdit": edit})


@pytest.fixture()
def docs_repo(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "services").mkdir(parents=True)
    (root / "services" / "api-gateway.mdx").write_text(PAGE, encoding="utf-8")
    (root / ".docsync").mkdir()
    (root / ".docsync" / "manifest.yml").write_text(MANIFEST_YML, encoding="utf-8")
    return root


def _diff() -> CodeDiff:
    return CodeDiff(
        repo="keephq/keep-api-gateway",
        base_sha="aaaaaaaa",
        head_sha="bbbbbbbb",
        pr_number=42,
        pr_title="add /alerts/bulk route",
        files=[
            ChangedFile(
                path="src/routes/router_setup.py",
                status=FileStatus.MODIFIED,
                hunks=["@@ def setup_routers(app): @@\n+    app.include_router(bulk)"],
                changed_symbols=["setup_routers"],
            )
        ],
    )


def test_pipeline_applies_validated_edit(docs_repo: Path):
    config = load_config(docs_repo)
    manifest = load_manifest(docs_repo)
    assert isinstance(config, DocsyncConfig)
    assert isinstance(manifest, Manifest)

    # The edit adds a table row — small, within the 40-line budget, no structural change.
    edit = PageEdit(
        edits=[
            EditOp(
                find="| POST | /alerts | Ingest an alert |",
                replace=(
                    "| POST | /alerts | Ingest an alert |\n"
                    "| POST | /alerts/bulk | Bulk-ingest alerts |"
                ),
                rationale="router_setup.py registers the new /alerts/bulk route",
            )
        ]
    )
    client = FakeClient(JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), edit)

    result = pipeline.run(_diff(), docs_repo, config, manifest, client=client)

    changed = result.changed()
    assert len(changed) == 1
    o = changed[0]
    assert o.page_path == "services/api-gateway.mdx"
    assert o.applied is True
    assert "/alerts/bulk" in o.new_content
    # title/description preserved (frontmatter gate).
    assert 'title: "API Gateway"' in o.new_content


class _CritiqueClient:
    """JudgeVerdict, PageEdit, and a scripted CritiqueVerdict (default-on critique)."""

    def __init__(self, verdict: JudgeVerdict, edit: PageEdit, critique: CritiqueVerdict):
        self.messages = _FakeMessages(
            {"JudgeVerdict": verdict, "PageEdit": edit, "CritiqueVerdict": critique}
        )


def _critique_setup(docs_repo: Path):
    config = load_config(docs_repo)
    manifest = load_manifest(docs_repo)
    edit = PageEdit(
        edits=[
            EditOp(
                find="| POST | /alerts | Ingest an alert |",
                replace=(
                    "| POST | /alerts | Ingest an alert |\n"
                    "| POST | /alerts/bulk | Bulk-ingest alerts |"
                ),
                rationale="router_setup.py registers the new /alerts/bulk route",
            )
        ]
    )
    # The critique rejects the only op (claims it's not justified by the diff).
    critique = CritiqueVerdict(
        faithful=False,
        rejected_finds=["| POST | /alerts | Ingest an alert |"],
        reason="op not justified by the diff",
    )
    client = _CritiqueClient(
        JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), edit, critique
    )
    return config, manifest, client


def test_pipeline_self_critique_on_by_default_drops_unfaithful_op(docs_repo: Path):
    # No self_critique arg → config.self_critique (True) applies. The critique rejects
    # the sole op, so nothing survives and the page is dropped, not edited.
    config, manifest, client = _critique_setup(docs_repo)
    assert config.self_critique is True  # the new default

    result = pipeline.run(_diff(), docs_repo, config, manifest, client=client)

    assert result.changed() == []
    o = result.outcomes[0]
    assert o.applied is False
    assert "dropped by self-critique" in o.note


def test_pipeline_no_self_critique_keeps_op(docs_repo: Path):
    # Explicit override turns critique off → the op is never re-checked and the edit
    # applies, proving the default is what drops it (not some other gate).
    config, manifest, client = _critique_setup(docs_repo)

    result = pipeline.run(
        _diff(), docs_repo, config, manifest, client=client, self_critique=False
    )

    changed = result.changed()
    assert len(changed) == 1
    assert "/alerts/bulk" in changed[0].new_content


def test_pipeline_drops_oversize_edit(docs_repo: Path):
    config = load_config(docs_repo)
    manifest = load_manifest(docs_repo)
    # Replace the whole page body — exceeds the diff-size guardrail.
    huge = "\n".join(f"new line {i}" for i in range(200))
    edit = PageEdit(
        edits=[EditOp(find="# API Gateway", replace=huge, rationale="rewrite")]
    )
    client = FakeClient(JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), edit)

    result = pipeline.run(_diff(), docs_repo, config, manifest, client=client)
    assert result.changed() == []
    dropped = [o for o in result.outcomes if not o.applied]
    assert dropped and "validation" in dropped[0].note


def test_pipeline_min_confidence_gates_edit(docs_repo: Path):
    manifest = load_manifest(docs_repo)
    # Disable anchor autopass so the page goes through the judge and carries the
    # judge's confidence (anchor autopass would pin it to 1.0 and never gate).
    config = DocsyncConfig(anchor_autopass=False)
    edit = PageEdit(
        edits=[
            EditOp(
                find="| POST | /alerts | Ingest an alert |",
                replace=(
                    "| POST | /alerts | Ingest an alert |\n"
                    "| POST | /alerts/bulk | Bulk-ingest alerts |"
                ),
                rationale="new route",
            )
        ]
    )
    client = FakeClient(
        JudgeVerdict(page_path="x", affected=True, confidence=0.6, reason="r"), edit
    )

    # Floor above the judge's 0.6 -> page skipped before the edit stage.
    gated = pipeline.run(_diff(), docs_repo, config, manifest, min_confidence=0.8, client=client)
    assert gated.changed() == []
    assert any("below floor" in o.note for o in gated.outcomes)

    # Floor below it -> the edit applies normally.
    ok = pipeline.run(_diff(), docs_repo, config, manifest, min_confidence=0.5, client=client)
    assert len(ok.changed()) == 1


TWO_PAGE_MANIFEST = """\
pages:
  - path: a.mdx
    sources:
      - repo: keephq/keep-api-gateway
        globs: ["src/routes/router_setup.py"]
        symbols: ["setup_routers"]
    max_diff_lines: 40
  - path: b.mdx
    sources:
      - repo: keephq/keep-api-gateway
        globs: ["src/routes/router_setup.py"]
        symbols: ["setup_routers"]
    max_diff_lines: 40
"""

PAGE_AB = '---\ntitle: "T"\ndescription: "D"\n---\n\nThe OLD value.\n'


@pytest.fixture()
def two_page_repo(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.mdx").write_text(PAGE_AB, encoding="utf-8")
    (root / "b.mdx").write_text(PAGE_AB, encoding="utf-8")
    (root / ".docsync").mkdir()
    (root / ".docsync" / "manifest.yml").write_text(TWO_PAGE_MANIFEST, encoding="utf-8")
    return root


def _ab_client() -> FakeClient:
    edit = PageEdit(
        edits=[EditOp(find="The OLD value.", replace="The NEW value.", rationale="r")]
    )
    return FakeClient(
        JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), edit
    )


def test_pipeline_max_pages_cap(two_page_repo: Path):
    # Both pages anchor-autopass (confidence 1.0); cap=1 edits the first (sorted),
    # reports the second without an edit call.
    manifest = load_manifest(two_page_repo)
    result = pipeline.run(
        _diff(), two_page_repo, DocsyncConfig(), manifest,
        max_pages=1, client=_ab_client(),
    )
    assert [o.page_path for o in result.changed()] == ["a.mdx"]
    capped = [o for o in result.outcomes if "cap" in o.note]
    assert [o.page_path for o in capped] == ["b.mdx"]
    assert capped[0].applied is False


def test_pipeline_parallel_is_deterministic(two_page_repo: Path):
    # max_parallel 4 + 2 pages -> the thread-pool path; map() preserves order.
    manifest = load_manifest(two_page_repo)
    config = DocsyncConfig(max_parallel_requests=4)
    r1 = pipeline.run(_diff(), two_page_repo, config, manifest, client=_ab_client())
    r2 = pipeline.run(_diff(), two_page_repo, config, manifest, client=_ab_client())

    order1 = [o.page_path for o in r1.outcomes]
    assert order1 == [o.page_path for o in r2.outcomes] == ["a.mdx", "b.mdx"]
    assert len(r1.changed()) == 2


def test_write_changes_returns_repo_root_relative_paths(tmp_path: Path):
    # Self-hosted layout: docs live in a `docs/` subdir, not at the repo root. The
    # returned paths must be prefixed with docs_root so `git add` (run from the repo
    # root in pr.open_pr) finds them — the bug that crashed the live self-docs loop.
    repo = tmp_path / "repo"
    (repo / "docs" / "services").mkdir(parents=True)
    (repo / "docs" / "services" / "api-gateway.mdx").write_text(PAGE, encoding="utf-8")
    (repo / ".docsync").mkdir()
    (repo / ".docsync" / "manifest.yml").write_text(MANIFEST_YML, encoding="utf-8")

    config = DocsyncConfig(docs_root="docs")
    manifest = load_manifest(repo)
    edit = PageEdit(
        edits=[
            EditOp(
                find="| POST | /alerts | Ingest an alert |",
                replace=(
                    "| POST | /alerts | Ingest an alert |\n"
                    "| POST | /alerts/bulk | Bulk-ingest alerts |"
                ),
                rationale="new route",
            )
        ]
    )
    client = FakeClient(JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), edit)

    result = pipeline.run(_diff(), repo, config, manifest, client=client)
    written = pipeline.write_changes(result, repo, config)

    assert written == ["docs/services/api-gateway.mdx"]
    # Each returned path resolves to a real file from the repo root (what git add needs).
    assert (repo / written[0]).exists()


def test_pipeline_idempotency_cursor(docs_repo: Path):
    # The cursor itself is enforced by the CLI, but confirm the helper round-trips.
    save_cursors(docs_repo, {"keephq/keep-api-gateway": "bbbbbbbb"})
    from docsync.config import already_processed

    assert already_processed(docs_repo, "keephq/keep-api-gateway", "bbbbbbbb") is True
    assert already_processed(docs_repo, "keephq/keep-api-gateway", "cccccccc") is False
