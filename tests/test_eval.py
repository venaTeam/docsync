"""Unit tests for the regression-eval harness (`docsync.eval`).

No network: ``run_eval`` is exercised with an injected ``diff_fn`` returning a
synthetic ``CodeDiff`` and a tiny in-memory ``Manifest`` that deterministically
anchors exactly one page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync.eval import (
    CaseResult,
    EvalReport,
    GoldenCase,
    aggregate,
    load_golden,
    run_eval,
    score_case,
)
from docsync.models import (
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    FileStatus,
    Manifest,
    ManifestPage,
    ManifestSource,
)

GOLDEN_PATH = Path(__file__).parent / "eval" / "golden.json"

REPO = "keephq/keep-api-gateway"


# ---------------------------------------------------------------------------
# score_case
# ---------------------------------------------------------------------------


def test_score_case_perfect():
    expected = {"a.mdx", "b.mdx"}
    actual = {"a.mdx", "b.mdx"}
    assert score_case(expected, actual) == (2, 0, 0)


def test_score_case_all_miss():
    expected = {"a.mdx", "b.mdx"}
    actual = {"x.mdx", "y.mdx"}
    # 0 tp, all actual are fp, all expected are fn
    assert score_case(expected, actual) == (0, 2, 2)


def test_score_case_mixed():
    expected = {"a.mdx", "b.mdx", "c.mdx"}
    actual = {"a.mdx", "z.mdx"}
    # a -> tp; z -> fp; b,c -> fn
    assert score_case(expected, actual) == (1, 1, 2)


def test_score_case_empty_expected_true_negative():
    # No expected pages and no actual pages -> nothing counted at all.
    assert score_case(set(), set()) == (0, 0, 0)


def test_score_case_empty_expected_overselect():
    # No expected pages but the mapper over-selected -> pure false positives,
    # nothing missed (fn=0 since there was nothing to find).
    assert score_case(set(), {"a.mdx", "b.mdx"}) == (0, 2, 0)


# ---------------------------------------------------------------------------
# aggregate (micro-averaged)
# ---------------------------------------------------------------------------


def _case(tp: int, fp: int, fn: int) -> CaseResult:
    return CaseResult(label="x", repo=REPO, tp=tp, fp=fp, fn=fn)


def test_aggregate_perfect():
    results = [_case(2, 0, 0), _case(1, 0, 0)]
    precision, recall, f1 = aggregate(results)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_aggregate_all_miss():
    results = [_case(0, 2, 2), _case(0, 1, 1)]
    precision, recall, f1 = aggregate(results)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


def test_aggregate_mixed_micro_average():
    # Summed: tp=3, fp=1, fn=2 -> P=3/4=0.75, R=3/5=0.6
    results = [_case(1, 1, 2), _case(2, 0, 0)]
    precision, recall, f1 = aggregate(results)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.6)
    assert f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))


def test_aggregate_empty_is_zero_not_error():
    # All true negatives (no tp/fp/fn anywhere) -> guarded div-by-zero -> 0.0.
    results = [_case(0, 0, 0), _case(0, 0, 0)]
    assert aggregate(results) == (0.0, 0.0, 0.0)
    assert aggregate([]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# load_golden
# ---------------------------------------------------------------------------


def test_load_golden_roundtrips_fixture():
    cases = load_golden(GOLDEN_PATH)
    assert len(cases) == 4
    assert all(isinstance(c, GoldenCase) for c in cases)

    by_repo = {c.repo: c for c in cases}
    assert set(by_repo) == {
        "venaTeam/keep-event-handler",
        "venaTeam/keep-api-gateway",
        "venaTeam/keep-workflows",
        "venaTeam/keep-ui",
    }

    positive = by_repo["venaTeam/keep-event-handler"]
    assert positive.expected_pages == [
        "services/event-handler.mdx",
        "operations/authentication.mdx",
        "architecture/data-flow.mdx",
    ]
    # The three true-negative-at-edit / mapping cases carry no expected pages.
    assert by_repo["venaTeam/keep-api-gateway"].expected_pages == []
    assert by_repo["venaTeam/keep-workflows"].expected_pages == []
    assert by_repo["venaTeam/keep-ui"].expected_pages == []


def test_load_golden_accepts_cases_wrapper(tmp_path: Path):
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        '{"cases": [{"repo": "keephq/x", "base": "b", "head": "h", '
        '"label": "l", "expected_pages": ["p.mdx"]}]}',
        encoding="utf-8",
    )
    cases = load_golden(wrapped)
    assert len(cases) == 1
    assert cases[0].repo == "keephq/x"
    assert cases[0].expected_pages == ["p.mdx"]


# ---------------------------------------------------------------------------
# run_eval (mode="map", injected diff_fn — no network)
# ---------------------------------------------------------------------------


def _manifest_anchoring_one_page() -> Manifest:
    """Manifest where 'api/alerts.mdx' anchors on the alerts route, and a second
    page anchors on an unrelated route that the synthetic diff never touches."""
    return Manifest(
        pages=[
            ManifestPage(
                path="api/alerts.mdx",
                sources=[ManifestSource(repo=REPO, globs=["src/routes/alerts.py"])],
            ),
            ManifestPage(
                path="api/incidents.mdx",
                sources=[ManifestSource(repo=REPO, globs=["src/routes/incidents.py"])],
            ),
        ]
    )


def _diff_touching_alerts(repo: str, base: str, head: str) -> CodeDiff:
    """Injected diff_fn: deterministically touches only the alerts route file."""
    return CodeDiff(
        repo=repo,
        base_sha=base,
        head_sha=head,
        files=[
            ChangedFile(path="src/routes/alerts.py", status=FileStatus.MODIFIED),
        ],
    )


def test_run_eval_map_mode_scores_one_anchored_page():
    manifest = _manifest_anchoring_one_page()
    config = DocsyncConfig()
    cases = [
        GoldenCase(
            repo=REPO,
            base="base1",
            head="head1",
            label="touches alerts route",
            expected_pages=["api/alerts.mdx"],
        )
    ]

    report = run_eval(
        cases,
        docs_repo=Path("."),
        config=config,
        manifest=manifest,
        mode="map",
        diff_fn=_diff_touching_alerts,
    )

    assert isinstance(report, EvalReport)
    assert report.n_cases == 1

    case = report.cases[0]
    assert case.actual == ["api/alerts.mdx"]  # exactly one page anchored
    assert (case.tp, case.fp, case.fn) == (1, 0, 0)

    # Perfect single case -> micro P/R/F1 all 1.0.
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


def test_run_eval_map_mode_counts_false_positive_on_true_negative_case():
    """A case with no expected pages whose diff still anchors a page -> over-select."""
    manifest = _manifest_anchoring_one_page()
    cases = [
        GoldenCase(
            repo=REPO,
            base="b",
            head="h",
            label="true negative at edit stage",
            expected_pages=[],
        )
    ]

    report = run_eval(
        cases,
        docs_repo=Path("."),
        config=DocsyncConfig(),
        manifest=manifest,
        mode="map",
        diff_fn=_diff_touching_alerts,
    )

    case = report.cases[0]
    assert case.actual == ["api/alerts.mdx"]
    assert (case.tp, case.fp, case.fn) == (0, 1, 0)
    assert report.precision == 0.0  # over-selected, nothing expected
    assert report.recall == 0.0  # no expected pages -> 0/0 guarded to 0.0


def test_run_eval_catches_per_case_errors_without_aborting():
    def _boom(repo: str, base: str, head: str) -> CodeDiff:
        raise RuntimeError("synthetic diff failure")

    cases = [
        GoldenCase(repo=REPO, base="b", head="h", label="explodes", expected_pages=["x.mdx"]),
    ]
    report = run_eval(
        cases,
        docs_repo=Path("."),
        config=DocsyncConfig(),
        manifest=_manifest_anchoring_one_page(),
        mode="map",
        diff_fn=_boom,
    )

    assert report.n_cases == 1
    case = report.cases[0]
    assert case.actual == []
    assert "ERROR" in case.label
    assert (case.tp, case.fp, case.fn) == (0, 0, 1)  # expected page missed


def test_run_eval_rejects_unknown_mode():
    with pytest.raises(ValueError):
        run_eval(
            [],
            docs_repo=Path("."),
            config=DocsyncConfig(),
            manifest=Manifest(),
            mode="bogus",
            diff_fn=_diff_touching_alerts,
        )
