"""Unit tests for Stage 3 doc-impact mapping (`docsync.impact`).

No real API calls and no `sentence_transformers` import — the judge is exercised
with a fake client, and embeddings are not enabled in these tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync.impact import (
    filter_docs_paths,
    find_anchor_candidates,
    map_impact,
)
from docsync.models import (
    CandidateSource,
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    FileStatus,
    JudgeVerdict,
    Manifest,
    ManifestPage,
    ManifestSource,
)

REPO = "keephq/keep-api-gateway"
OTHER_REPO = "keephq/keep-ui"


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def make_diff(
    *,
    repo: str = REPO,
    paths: list[str] | None = None,
    symbols: list[str] | None = None,
) -> CodeDiff:
    files = []
    for p in paths or []:
        files.append(
            ChangedFile(
                path=p,
                status=FileStatus.MODIFIED,
                changed_symbols=list(symbols or []),
            )
        )
    if not files:
        files.append(
            ChangedFile(
                path="src/placeholder.py",
                status=FileStatus.MODIFIED,
                changed_symbols=list(symbols or []),
            )
        )
    return CodeDiff(repo=repo, base_sha="a" * 40, head_sha="b" * 40, files=files)


def make_manifest(*pages: ManifestPage) -> Manifest:
    return Manifest(pages=list(pages))


def page(path: str, *sources: ManifestSource, **kw) -> ManifestPage:
    return ManifestPage(path=path, sources=list(sources), **kw)


def source(
    *, repo: str = REPO, globs: list[str] | None = None, symbols: list[str] | None = None
) -> ManifestSource:
    return ManifestSource(repo=repo, globs=globs or [], symbols=symbols or [])


# ---------------------------------------------------------------------------
# find_anchor_candidates
# ---------------------------------------------------------------------------


def test_anchor_glob_match():
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(
        page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])),
        page("api/incidents.mdx", source(globs=["src/routes/incidents.py"])),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    assert cands[0].page_path == "api/alerts.mdx"
    assert cands[0].source == CandidateSource.ANCHOR
    assert cands[0].score == 1.0
    assert "alerts.py" in cands[0].reason


def test_anchor_glob_wildcard():
    diff = make_diff(paths=["src/routes/alerts.py", "src/routes/incidents.py"])
    manifest = make_manifest(page("api/routes.mdx", source(globs=["src/routes/*.py"])))
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    # Both changed paths match the one glob → score 2.
    assert cands[0].score == 2.0


def test_anchor_symbol_exact_match():
    diff = make_diff(paths=["src/x.py"], symbols=["process_alert", "ignore_me"])
    manifest = make_manifest(
        page("api/alerts.mdx", source(symbols=["process_alert"])),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    assert "process_alert (=process_alert)" in cands[0].reason


def test_anchor_symbol_trailing_star_prefix_match():
    diff = make_diff(paths=["src/config.py"], symbols=["ENV_HOST", "ENV_PORT", "OTHER"])
    manifest = make_manifest(
        page("config/env.mdx", source(symbols=["ENV_*"])),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    # ENV_HOST and ENV_PORT match ENV_*, OTHER does not.
    assert cands[0].score == 2.0
    assert "ENV_HOST (~ENV_*)" in cands[0].reason
    assert "ENV_PORT (~ENV_*)" in cands[0].reason
    assert "OTHER" not in cands[0].reason


def test_anchor_repo_scoping_excludes_other_repo():
    # The page's source is anchored to a *different* repo than the diff.
    diff = make_diff(repo=REPO, paths=["src/routes/alerts.py"])
    manifest = make_manifest(
        page("api/alerts.mdx", source(repo=OTHER_REPO, globs=["src/routes/alerts.py"])),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert cands == []


def test_anchor_repo_scoping_mixed_sources():
    diff = make_diff(repo=REPO, paths=["src/routes/alerts.py"], symbols=["process_alert"])
    manifest = make_manifest(
        page(
            "api/alerts.mdx",
            source(repo=OTHER_REPO, globs=["src/routes/alerts.py"]),  # wrong repo, ignored
            source(repo=REPO, symbols=["process_alert"]),  # right repo, matches
        ),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    assert cands[0].score == 1.0
    assert "process_alert" in cands[0].reason
    assert "alerts.py" not in cands[0].reason


def test_anchor_combined_path_and_symbol_score():
    diff = make_diff(paths=["src/routes/alerts.py"], symbols=["process_alert"])
    manifest = make_manifest(
        page(
            "api/alerts.mdx",
            source(globs=["src/routes/*.py"], symbols=["process_alert"]),
        ),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    assert cands[0].score == 2.0
    assert "paths:" in cands[0].reason
    assert "symbols:" in cands[0].reason


def test_anchor_previous_path_for_renames():
    diff = CodeDiff(
        repo=REPO,
        base_sha="a" * 40,
        head_sha="b" * 40,
        files=[
            ChangedFile(
                path="src/routes/alerts_v2.py",
                previous_path="src/routes/alerts.py",
                status=FileStatus.RENAMED,
            )
        ],
    )
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    cands = find_anchor_candidates(diff, manifest)
    # The previous_path is included in changed_paths(), so the rename still matches.
    assert len(cands) == 1


def test_anchor_no_match_returns_empty():
    diff = make_diff(paths=["src/unrelated.py"], symbols=["nothing"])
    manifest = make_manifest(
        page("api/alerts.mdx", source(globs=["src/routes/alerts.py"], symbols=["process_alert"])),
    )
    assert find_anchor_candidates(diff, manifest) == []


# ---------------------------------------------------------------------------
# Empty-repo wildcard (mono / single convenience)
# ---------------------------------------------------------------------------


def test_anchor_empty_repo_matches_any_diff():
    # A source with no `repo` is a wildcard: it matches regardless of the diff's repo.
    diff = make_diff(repo=REPO, paths=["src/routes/alerts.py"])
    manifest = make_manifest(
        page("api/alerts.mdx", ManifestSource(globs=["src/routes/alerts.py"])),
    )
    cands = find_anchor_candidates(diff, manifest)
    assert len(cands) == 1
    assert cands[0].page_path == "api/alerts.mdx"


def test_anchor_explicit_repo_still_scopes_when_others_are_wildcard():
    diff = make_diff(repo=REPO, paths=["src/routes/alerts.py"])
    manifest = make_manifest(
        # explicit, wrong repo → excluded even though a wildcard would have matched
        page("api/alerts.mdx", source(repo=OTHER_REPO, globs=["src/routes/alerts.py"])),
    )
    assert find_anchor_candidates(diff, manifest) == []


# ---------------------------------------------------------------------------
# filter_docs_paths (mono-repo diff filtering)
# ---------------------------------------------------------------------------


def test_filter_docs_paths_drops_docs_subtree():
    diff = make_diff(paths=["src/routes/alerts.py", "docs/reference/alerts.mdx"])
    filtered = filter_docs_paths(diff, "docs")
    kept = filtered.changed_paths()
    assert "src/routes/alerts.py" in kept
    assert "docs/reference/alerts.mdx" not in kept


def test_filter_docs_paths_handles_renames_via_previous_path():
    diff = CodeDiff(
        repo=REPO,
        base_sha="a" * 40,
        head_sha="b" * 40,
        files=[
            ChangedFile(
                path="docs/new.mdx",
                previous_path="docs/old.mdx",
                status=FileStatus.RENAMED,
            ),
            ChangedFile(path="src/x.py", status=FileStatus.MODIFIED),
        ],
    )
    filtered = filter_docs_paths(diff, "docs")
    assert [f.path for f in filtered.files] == ["src/x.py"]


def test_filter_docs_paths_root_docs_is_noop():
    # docs at the repo root ('.') — there's no code subtree to separate.
    diff = make_diff(paths=["index.mdx", "guide.mdx"])
    filtered = filter_docs_paths(diff, ".")
    assert filtered is diff


def test_filter_docs_paths_no_docs_files_returns_same_object():
    diff = make_diff(paths=["src/a.py", "src/b.py"])
    assert filter_docs_paths(diff, "docs") is diff


# ---------------------------------------------------------------------------
# Fake judge client
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, verdict_for):
        self._verdict_for = verdict_for
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        # Derive which page is being judged from the user message.
        user_msg = kwargs["messages"][0]["content"]
        verdict = self._verdict_for(user_msg)

        class _Resp:
            parsed_output = verdict

        return _Resp()


class FakeClient:
    """Stub whose `.messages.parse(...)` returns an object with `.parsed_output`."""

    def __init__(self, verdict_for):
        self.messages = _FakeMessages(verdict_for)


def _page_in(user_msg: str, page_path: str) -> bool:
    return page_path in user_msg


# ---------------------------------------------------------------------------
# map_impact decision logic
# ---------------------------------------------------------------------------


def test_anchor_autopass_keeps_page_without_calling_judge(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig()  # anchor_autopass defaults True
    assert config.anchor_autopass is True

    def _verdict_for(_msg):  # pragma: no cover — must never be called
        raise AssertionError("judge should not be called for an autopass anchor")

    client = FakeClient(_verdict_for)
    impacted = map_impact(diff, manifest, tmp_path, config, client=client)

    assert len(impacted) == 1
    assert impacted[0].page_path == "api/alerts.mdx"
    assert impacted[0].source == CandidateSource.ANCHOR
    assert impacted[0].confidence == 1.0
    # Judge was never invoked.
    assert client.messages.calls == []


def test_anchor_judged_when_autopass_disabled_high_confidence_kept(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    def _verdict_for(msg):
        return JudgeVerdict(
            page_path="", affected=True, confidence=0.9, reason="route changed"
        )

    client = FakeClient(_verdict_for)
    impacted = map_impact(diff, manifest, tmp_path, config, client=client)

    assert len(impacted) == 1
    assert impacted[0].page_path == "api/alerts.mdx"
    assert impacted[0].confidence == 0.9
    # Judge WAS called this time (autopass disabled).
    assert len(client.messages.calls) == 1
    # Haiku model is used, no thinking/effort params passed.
    call = client.messages.calls[0]
    assert call["model"] == config.models.judge_model == "claude-haiku-4-5"
    assert "thinking" not in call
    assert "output_config" not in call
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_low_confidence_verdict_is_dropped(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    def _verdict_for(_msg):
        return JudgeVerdict(
            page_path="", affected=True, confidence=0.3, reason="weak signal"
        )

    impacted = map_impact(diff, manifest, tmp_path, config, client=FakeClient(_verdict_for))
    assert impacted == []


def test_not_affected_verdict_is_dropped(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    def _verdict_for(_msg):
        return JudgeVerdict(
            page_path="", affected=False, confidence=0.99, reason="internal refactor"
        )

    impacted = map_impact(diff, manifest, tmp_path, config, client=FakeClient(_verdict_for))
    assert impacted == []


def test_judge_required_page_routes_through_judge_despite_anchor(tmp_path: Path):
    # A narrative page with a broad anchor + judge_required must NOT autopass: the judge
    # decides, so an unrelated subsystem change can be skipped. The reference page next
    # to it (no judge_required) still autopasses without a judge call.
    diff = make_diff(paths=["src/services/dedup/engine.py"])
    manifest = make_manifest(
        page("concepts/dedup.mdx", source(globs=["src/services/dedup/*"]), judge_required=True),
        page("reference/dedup.mdx", source(globs=["src/services/dedup/*"])),
    )
    config = DocsyncConfig()  # anchor_autopass True

    def _verdict_for(msg):
        # Judge says the narrative page is NOT actually invalidated by this change.
        return JudgeVerdict(page_path="", affected=False, confidence=0.9, reason="no narrative change")

    client = FakeClient(_verdict_for)
    impacted = map_impact(diff, manifest, tmp_path, config, client=client)

    paths = {i.page_path for i in impacted}
    # Reference page autopassed; narrative page was judged and dropped (affected=False).
    assert paths == {"reference/dedup.mdx"}
    # The judge was invoked exactly once — for the judge_required page only.
    assert len(client.messages.calls) == 1
    assert "concepts/dedup.mdx" in client.messages.calls[0]["messages"][0]["content"]


def test_judge_required_page_kept_when_judge_confirms(tmp_path: Path):
    diff = make_diff(paths=["src/services/dedup/engine.py"])
    manifest = make_manifest(
        page("concepts/dedup.mdx", source(globs=["src/services/dedup/*"]), judge_required=True),
    )
    config = DocsyncConfig(judge_confidence_threshold=0.5)

    def _verdict_for(_msg):
        return JudgeVerdict(page_path="", affected=True, confidence=0.8, reason="flow changed")

    impacted = map_impact(diff, manifest, tmp_path, config, client=FakeClient(_verdict_for))
    assert [i.page_path for i in impacted] == ["concepts/dedup.mdx"]
    assert impacted[0].confidence == 0.8  # judged confidence, not autopass 1.0


def test_judge_verdict_page_path_pinned_to_candidate(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    def _verdict_for(_msg):
        # Model echoes the WRONG page_path — map_impact must pin to the candidate.
        return JudgeVerdict(
            page_path="api/WRONG.mdx", affected=True, confidence=0.8, reason="ok"
        )

    impacted = map_impact(diff, manifest, tmp_path, config, client=FakeClient(_verdict_for))
    assert len(impacted) == 1
    assert impacted[0].page_path == "api/alerts.mdx"


def test_judge_exception_yields_non_affected_and_does_not_crash(tmp_path: Path):
    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    def _verdict_for(_msg):
        raise RuntimeError("simulated API failure")

    # Should not raise, and the failed page is dropped (treated as non-affected).
    impacted = map_impact(diff, manifest, tmp_path, config, client=FakeClient(_verdict_for))
    assert impacted == []


def test_no_candidates_returns_empty_without_judge(tmp_path: Path):
    diff = make_diff(paths=["src/unrelated.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False)

    def _verdict_for(_msg):  # pragma: no cover — never called
        raise AssertionError("judge should not be called with no candidates")

    client = FakeClient(_verdict_for)
    impacted = map_impact(diff, manifest, tmp_path, config, client=client)
    assert impacted == []
    assert client.messages.calls == []


def test_mixed_autopass_anchor_and_judged_page(tmp_path: Path):
    # Two anchored pages; with autopass on, both keep without a judge call.
    diff = make_diff(
        paths=["src/routes/alerts.py", "src/routes/incidents.py"],
    )
    manifest = make_manifest(
        page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])),
        page("api/incidents.mdx", source(globs=["src/routes/incidents.py"])),
    )
    config = DocsyncConfig()  # autopass on

    def _verdict_for(_msg):  # pragma: no cover
        raise AssertionError("no judge expected")

    client = FakeClient(_verdict_for)
    impacted = map_impact(diff, manifest, tmp_path, config, client=client)
    assert {p.page_path for p in impacted} == {"api/alerts.mdx", "api/incidents.mdx"}
    assert client.messages.calls == []


def test_page_text_is_read_and_passed_to_judge(tmp_path: Path):
    # Write a real .mdx so the judge user message includes its text.
    docs = tmp_path
    (docs / "api").mkdir()
    (docs / "api" / "alerts.mdx").write_text("# Alerts\n\nThe POST /alerts route.\n")

    diff = make_diff(paths=["src/routes/alerts.py"])
    manifest = make_manifest(page("api/alerts.mdx", source(globs=["src/routes/alerts.py"])))
    config = DocsyncConfig(anchor_autopass=False, judge_confidence_threshold=0.5)

    captured = {}

    def _verdict_for(msg):
        captured["msg"] = msg
        return JudgeVerdict(page_path="", affected=True, confidence=0.7, reason="ok")

    map_impact(diff, manifest, docs, config, client=FakeClient(_verdict_for))
    assert "POST /alerts route" in captured["msg"]
    assert "api/alerts.mdx" in captured["msg"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_anchor_matches_local_path_against_canonical_repo():
    """Regression: a local checkout path must reconcile with a manifest's owner/name.

    Found by dogfooding `docsync map` against the real Keep repos — diff_local sets
    repo to the checkout path, which must still match `keephq/keep-api-gateway`.
    """
    from docsync.models import (
        ChangedFile,
        CodeDiff,
        FileStatus,
        Manifest,
        ManifestPage,
        ManifestSource,
    )
    from docsync.impact import find_anchor_candidates

    diff = CodeDiff(
        repo="/Users/dev/keep-namespace/keep-api-gateway",  # local path, not owner/name
        base_sha="a",
        head_sha="b",
        files=[
            ChangedFile(
                path="src/config/config.py",
                status=FileStatus.MODIFIED,
                changed_symbols=["KEEP_EXTRACT_IDENTITY"],
            )
        ],
    )
    manifest = Manifest(
        pages=[
            ManifestPage(
                path="services/api-gateway.mdx",
                sources=[
                    ManifestSource(
                        repo="keephq/keep-api-gateway",
                        globs=["src/config/config.py"],
                        symbols=["KEEP_*"],
                    )
                ],
            )
        ]
    )
    cands = find_anchor_candidates(diff, manifest)
    assert [c.page_path for c in cands] == ["services/api-gateway.mdx"]

    # A genuinely different repo must NOT match (basename differs).
    diff2 = diff.model_copy(update={"repo": "keephq/keep-ui"})
    assert find_anchor_candidates(diff2, manifest) == []
