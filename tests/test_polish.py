"""Tests for the opt-in readability polish pass (`docsync.polish`).

No network: fake clients whose `.messages.parse(output_format=PageEdit)` return a scripted
`PageEdit`. Cover the prompt contract, the safe-application gates (frontmatter frozen, body
not gutted, ops must apply, structural validation), metering under the `"polish"` stage, and
the bootstrap/pipeline wiring (only fires when `readability_pass` is set).
"""

from __future__ import annotations

import json
from pathlib import Path

from docsync import bootstrap as bootstrap_mod
from docsync import pipeline, polish, style
from docsync.cost import MeteredClient, UsageMeter
from docsync.models import (
    AuthoredPage,
    ChangedFile,
    CodeDiff,
    DocPlan,
    DocsyncConfig,
    EditOp,
    FileStatus,
    JudgeVerdict,
    PageEdit,
    PlannedPage,
    PlannedSource,
)
from docsync.validate import get_adapter

# A valid page with frontmatter + a body long enough that "gutting" it is detectable.
_INTRO = (
    "Some intro paragraph that is reasonably long so the body has real length. "
    "It documents the endpoints in detail across several sentences here for weight."
)
_BODY = f"# Alerts\n\n{_INTRO}\n\n## Endpoints\n\nDetails about each endpoint live here.\n"
PAGE = f"---\ntitle: Alerts\ndescription: The alerts API.\n---\n\n{_BODY}"

ADAPTER = get_adapter("x.mdx")


class _Resp:
    usage = {"input_tokens": 10, "output_tokens": 5}

    def __init__(self, parsed):
        self.parsed_output = parsed


class _PolishClient:
    """Returns one scripted `PageEdit` for every parse call."""

    def __init__(self, edit: PageEdit):
        self._edit = edit
        self.calls: list[dict] = []

        class _Messages:
            def parse(_self, **kwargs):
                self.calls.append(kwargs)
                return _Resp(self._edit)

        self.messages = _Messages()


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def test_polish_prompt_carries_style_rules_and_fact_freeze():
    system, user = polish.build_polish_prompt("reference/alerts.mdx", PAGE, "reference")
    assert style.INVERTED_PYRAMID in system
    assert style.SCANNABILITY in system
    assert style.kind_structure("reference") in system
    assert "do not add, remove, or change any FACT" in system
    assert "NEVER change the YAML frontmatter" in system
    assert "reference/alerts.mdx" in user and "```mdx" in user


def test_polish_prompt_structure_is_kind_specific():
    sys_c, _ = polish.build_polish_prompt("c.mdx", PAGE, "concept")
    sys_r, _ = polish.build_polish_prompt("r.mdx", PAGE, "reference")
    assert "CONCEPT" in sys_c and "REFERENCE" in sys_r and sys_c != sys_r


# ---------------------------------------------------------------------------
# polish_text — safe application + fallbacks
# ---------------------------------------------------------------------------


def test_polish_text_applies_clean_readability_edit():
    edit = PageEdit(edits=[EditOp(
        find="Some intro paragraph",
        replace="Alerts lets you ingest and query alerts. Some intro paragraph",
        rationale="lead with the point",
    )])
    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_PolishClient(edit)
    )
    assert polished is True
    assert "Alerts lets you ingest and query alerts." in out
    assert "polished" in note
    # frontmatter untouched
    assert ADAPTER.split_frontmatter(out)[0] == ADAPTER.split_frontmatter(PAGE)[0]


def test_polish_text_rejects_frontmatter_change():
    edit = PageEdit(edits=[EditOp(find="title: Alerts", replace="title: Changed", rationale="x")])
    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_PolishClient(edit)
    )
    assert polished is False and out == PAGE
    assert "frontmatter" in note


def test_polish_text_rejects_gutted_body():
    edit = PageEdit(edits=[EditOp(find=_BODY, replace="# Alerts\n\nhi\n", rationale="nuke")])
    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_PolishClient(edit)
    )
    assert polished is False and out == PAGE
    assert "too short" in note


def test_polish_text_falls_back_when_op_not_applicable():
    edit = PageEdit(edits=[EditOp(find="NOT IN THE PAGE", replace="x", rationale="bad")])
    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_PolishClient(edit)
    )
    assert polished is False and out == PAGE
    assert "not applicable" in note


def test_polish_text_noop_when_no_edits():
    edit = PageEdit(edits=[], no_change_reason="already reads well")
    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_PolishClient(edit)
    )
    assert polished is False and out == PAGE
    assert "already reads well" in note


def test_polish_text_survives_client_error():
    class _Boom:
        class messages:
            @staticmethod
            def parse(**kwargs):
                raise RuntimeError("api down")

    out, polished, note = polish.polish_text(
        "r.mdx", PAGE, "reference", DocsyncConfig(), ADAPTER, client=_Boom()
    )
    assert polished is False and out == PAGE
    assert "polish skipped" in note


def test_polish_page_uses_edit_model_and_meters_polish_stage():
    edit = PageEdit(edits=[EditOp(find="Some intro paragraph", replace="Lead. Some intro paragraph", rationale="r")])
    meter = UsageMeter()
    client = MeteredClient(_PolishClient(edit), meter)
    polish.polish_page("r.mdx", PAGE, "reference", DocsyncConfig(), client=client)
    usage = meter.finalize()
    assert {m.stage for m in usage.by_model} == {"polish"}
    assert usage.by_model[0].model == DocsyncConfig().models.edit_model


# ---------------------------------------------------------------------------
# Wiring — only fires when readability_pass is set
# ---------------------------------------------------------------------------

_AUTHORED = (
    "---\ntitle: A Page\ndescription: A generated page.\n---\n\n# A Page\n\n"
    + ("This page documents the subsystem in detail. " * 8)
    + "\n\n<Note>Authoritative.</Note>\n"
)


class _BootstrapClient:
    """Scripts a DocPlan, an AuthoredPage, and a polish PageEdit by output_format."""

    def __init__(self, plan: DocPlan, polish_edit: PageEdit):
        self.calls: list[dict] = []

        class _Messages:
            def parse(_self, **kwargs):
                self.calls.append(kwargs)
                fmt = kwargs.get("output_format")
                if fmt is DocPlan:
                    return _Resp(plan)
                if fmt is AuthoredPage:
                    return _Resp(AuthoredPage(content=_AUTHORED))
                if fmt is PageEdit:
                    return _Resp(polish_edit)
                raise AssertionError(f"unexpected output_format {fmt!r}")

        self.messages = _Messages()


def _gw(tmp_path: Path) -> Path:
    repo = tmp_path / "gw"
    fp = repo / "src" / "routes" / "alerts.py"
    fp.parent.mkdir(parents=True)
    fp.write_text("def get_alerts():\n    return []\n")
    return repo


def _docs_repo(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    nav = {"name": "Docs", "theme": "mint", "colors": {"primary": "#000"},
           "navigation": {"groups": [{"group": "Existing", "pages": []}]}}
    (docs / "docs.json").write_text(json.dumps(nav) + "\n")
    return docs


def _plan() -> DocPlan:
    return DocPlan(pages=[PlannedPage(
        page_path="reference/alerts.mdx", title="Alerts", kind="reference",
        section="Reference", summary="cover it",
        sources=[PlannedSource(repo="gw", globs=["src/routes/alerts.py"], symbols=["get_alerts"])],
    )])


def test_bootstrap_runs_polish_when_enabled(tmp_path):
    # The polish edit inserts a lead sentence (paraphrase) — applies cleanly, keeps fm.
    polish_edit = PageEdit(edits=[EditOp(
        find="# A Page\n", replace="# A Page\n\nThe page, at a glance.\n", rationale="lead",
    )])
    client = _BootstrapClient(_plan(), polish_edit)
    config = DocsyncConfig(readability_pass=True)
    result = bootstrap_mod.run_bootstrap([("gw", _gw(tmp_path))], _docs_repo(tmp_path), config, client=client)

    assert len(result.authored()) == 1
    assert "The page, at a glance." in result.authored()[0].new_content
    assert "polished" in result.authored()[0].note
    assert "polish" in {m.stage for m in result.usage.by_model}


def test_bootstrap_skips_polish_by_default(tmp_path):
    client = _BootstrapClient(_plan(), PageEdit(edits=[]))
    result = bootstrap_mod.run_bootstrap(
        [("gw", _gw(tmp_path))], _docs_repo(tmp_path), DocsyncConfig(), client=client
    )
    assert len(result.authored()) == 1
    # No PageEdit call was ever made, and no polish stage was metered.
    assert all(c.get("output_format") is not PageEdit for c in client.calls)
    assert "polish" not in {m.stage for m in result.usage.by_model}


# ---------------------------------------------------------------------------
# Pipeline wiring — polish the validated edit (call-counting fake)
# ---------------------------------------------------------------------------

_PIPE_PAGE = (
    "---\ntitle: \"API Gateway\"\ndescription: \"REST API service\"\n---\n\n"
    "# API Gateway\n\n| Method | Path | Purpose |\n| --- | --- | --- |\n"
    "| POST | /alerts | Ingest an alert |\n\nConfigured via `setup_routers`.\n"
)
_PIPE_MANIFEST = (
    "pages:\n  - path: services/api-gateway.mdx\n    sources:\n"
    "      - repo: keephq/keep-api-gateway\n        globs: [\"src/routes/router_setup.py\"]\n"
    "        symbols: [\"setup_routers\"]\n    max_diff_lines: 40\n"
)


class _PipelineClient:
    """JudgeVerdict for the judge; first PageEdit = surgical edit, second = polish."""

    def __init__(self, verdict, surgical: PageEdit, polish_edit: PageEdit):
        self._verdict = verdict
        self._edits = [surgical, polish_edit]

        class _Messages:
            def parse(_self, **kwargs):
                fmt = kwargs.get("output_format")
                if fmt is JudgeVerdict:
                    return _Resp(self._verdict)
                return _Resp(self._edits.pop(0))

        self.messages = _Messages()


def _pipe_repo(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "services").mkdir(parents=True)
    (root / "services" / "api-gateway.mdx").write_text(_PIPE_PAGE, encoding="utf-8")
    (root / ".docsync").mkdir()
    (root / ".docsync" / "manifest.yml").write_text(_PIPE_MANIFEST, encoding="utf-8")
    return root


def _pipe_diff() -> CodeDiff:
    return CodeDiff(
        repo="keephq/keep-api-gateway", base_sha="aaaaaaaa", head_sha="bbbbbbbb",
        files=[ChangedFile(
            path="src/routes/router_setup.py", status=FileStatus.MODIFIED,
            hunks=["@@ def setup_routers(app): @@\n+    app.include_router(bulk)"],
            changed_symbols=["setup_routers"],
        )],
    )


def test_pipeline_polishes_validated_edit_when_enabled(tmp_path):
    from docsync.config import load_config, load_manifest

    docs = _pipe_repo(tmp_path)
    config = load_config(docs)
    config.readability_pass = True
    manifest = load_manifest(docs)

    surgical = PageEdit(edits=[EditOp(
        find="| POST | /alerts | Ingest an alert |",
        replace="| POST | /alerts | Ingest an alert |\n| POST | /alerts/bulk | Bulk-ingest |",
        rationale="new route",
    )])
    # Polish leads the page with a summary sentence (paraphrase) — applies post-edit.
    polish_edit = PageEdit(edits=[EditOp(
        find="# API Gateway\n", replace="# API Gateway\n\nThe gateway ingests alerts over REST.\n",
        rationale="lead",
    )])
    client = _PipelineClient(
        JudgeVerdict(page_path="x", affected=True, confidence=0.9, reason="r"), surgical, polish_edit
    )
    result = pipeline.run(_pipe_diff(), docs, config, manifest, client=client)
    changed = result.changed()
    assert len(changed) == 1
    assert "/alerts/bulk" in changed[0].new_content           # surgical edit survived
    assert "The gateway ingests alerts over REST." in changed[0].new_content  # polish applied
    assert "polished" in changed[0].note
