"""The light/medium/high thoroughness dial.

Covers the style helpers, the config resolver, the diff-size budget scaling in
validate, the prompt directives in edit/author/planner, and the CLI override.
No network, no LLM.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from docsync import style
from docsync.bootstrap import build_author_prompt, build_plan_prompt
from docsync.cli import app
from docsync.edits import _build_system_prompt as build_edit_system_prompt
from docsync.models import (
    DocsyncConfig,
    ManifestPage,
    PlannedPage,
    PlannedSource,
    RepoDigest,
    SourceUnit,
)
from docsync.validate import _check_diff_size

runner = CliRunner()


# --- style helpers ------------------------------------------------------------


def test_thoroughness_directive_distinct_per_level():
    light = style.thoroughness_directive("light")
    medium = style.thoroughness_directive("medium")
    high = style.thoroughness_directive("high")
    assert "LIGHT" in light and "MEDIUM" in medium and "HIGH" in high
    assert len({light, medium, high}) == 3


def test_thoroughness_directive_defaults_to_medium():
    assert style.thoroughness_directive("nonsense") == style.thoroughness_directive("medium")


def test_tokens_for_scales_with_level():
    base = 8000
    assert style.tokens_for("light", base) < base
    assert style.tokens_for("medium", base) == base
    assert style.tokens_for("high", base) > base


def test_tokens_for_floors_small_budgets():
    assert style.tokens_for("light", 100) >= 1024


# --- config resolver ----------------------------------------------------------


def test_thoroughness_for_global_default():
    assert DocsyncConfig().thoroughness_for() == "medium"
    assert DocsyncConfig(thoroughness="high").thoroughness_for() == "high"


def test_thoroughness_for_per_kind_override():
    config = DocsyncConfig(thoroughness="light", thoroughness_by_kind={"reference": "high"})
    assert config.thoroughness_for("reference") == "high"  # overridden
    assert config.thoroughness_for("concept") == "light"   # falls back to global
    assert config.thoroughness_for() == "light"            # no kind → global


# --- validate diff-size budget ------------------------------------------------


def _net_lines(n_changed: int) -> tuple[str, str]:
    """An original and a new text differing by ~n_changed net lines."""
    original = "\n".join(f"line {i}" for i in range(200))
    new = "\n".join((f"CHANGED {i}" if i < n_changed else f"line {i}") for i in range(200))
    return original, new


def test_diff_size_medium_matches_legacy_default():
    # medium == the old hard-coded 60/0.5 fallback: a 70-line change over budget.
    original, new = _net_lines(70)  # 70 removed + 70 added = 140 net > 60
    assert _check_diff_size(original, new, None, "medium")  # non-empty == failure


def test_diff_size_high_is_more_permissive_than_light():
    # A change that light rejects but high allows.
    original, new = _net_lines(45)  # 90 net changed lines
    assert _check_diff_size(original, new, None, "light")   # over light's 40-line budget
    assert not _check_diff_size(original, new, None, "high")  # within high's 100-line budget


def test_diff_size_explicit_manifest_override_wins_over_level():
    original, new = _net_lines(45)
    page = ManifestPage(path="p.mdx", max_diff_lines=500, max_diff_pct=1.0)
    # Even under light, an explicit generous per-page budget is respected.
    assert not _check_diff_size(original, new, page, "light")


# --- prompt directives --------------------------------------------------------


def _planned(kind: str = "reference") -> PlannedPage:
    return PlannedPage(
        page_path="reference/x.mdx", title="X", kind=kind, section="Reference",
        summary="cover it",
        sources=[PlannedSource(repo="gw", globs=["src/*.py"], symbols=["get_x"])],
    )


def test_author_prompt_carries_thoroughness_directive():
    sys_high, _ = build_author_prompt(_planned(), [("gw/x.py", "code")], "high")
    sys_light, _ = build_author_prompt(_planned(), [("gw/x.py", "code")], "light")
    assert "HIGH" in sys_high
    assert "LIGHT" in sys_light


def test_plan_prompt_carries_size_target():
    digests = [RepoDigest(repo="gw", root="/tmp", units=[SourceUnit(path="a.py", kind="python")])]
    sys_light, _ = build_plan_prompt(
        digests, existing_routes=[], existing_pages=set(), thoroughness="light"
    )
    sys_high, _ = build_plan_prompt(
        digests, existing_routes=[], existing_pages=set(), thoroughness="high"
    )
    assert "3-6 pages" in sys_light
    assert "COMPREHENSIVE" in sys_high


def test_edit_system_prompt_carries_thoroughness_directive():
    assert "HIGH" in build_edit_system_prompt(False, "high")
    assert "LIGHT" in build_edit_system_prompt(False, "light")


# --- CLI flag -----------------------------------------------------------------


def _docs_repo(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / ".docsync").mkdir(parents=True)
    (root / ".docsync" / "manifest.yml").write_text("pages: []\n", encoding="utf-8")
    return root


def test_run_rejects_invalid_thoroughness(tmp_path: Path):
    docs = _docs_repo(tmp_path)
    result = runner.invoke(
        app, ["run", "--docs-repo", str(docs), "--thoroughness", "extreme", "--no-preflight"]
    )
    assert result.exit_code != 0
    assert "thoroughness" in result.output.lower()
