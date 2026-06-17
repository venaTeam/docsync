"""Integration test for the orchestration: diff -> impact -> edits -> validate.

Uses a fake Anthropic client so no network is touched. Exercises the real
anchor mapping, edit application, and validation gates end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync import pipeline
from docsync.config import load_config, load_manifest, save_cursors
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


def test_pipeline_idempotency_cursor(docs_repo: Path):
    # The cursor itself is enforced by the CLI, but confirm the helper round-trips.
    save_cursors(docs_repo, {"keephq/keep-api-gateway": "bbbbbbbb"})
    from docsync.config import already_processed

    assert already_processed(docs_repo, "keephq/keep-api-gateway", "bbbbbbbb") is True
    assert already_processed(docs_repo, "keephq/keep-api-gateway", "cccccccc") is False
