"""Tests for Stage 2 — diff extraction (docsync.diff).

No network / git calls: the unidiff parse path is exercised through the
``parse_patchset`` helper with inline, byte-accurate unified-diff fixtures
(generated from real ``git diff`` output), and symbol extraction is tested
directly with realistic hunk strings.
"""

from __future__ import annotations

from docsync.diff import (
    _split_github_patch,
    extract_changed_symbols,
    parse_patchset,
)
from docsync.models import FileStatus

# ---------------------------------------------------------------------------
# Fixtures (byte-accurate; context lines carry the leading space unidiff needs)
# ---------------------------------------------------------------------------

# A hunk whose enclosing-scope heading is `def register_routes(app):`, plus an
# added module-level `ROUTES = [...]` assignment and an added `def new_handler`.
HUNK_WITH_HEADING = "\n".join(
    [
        "@@ -2,6 +2,8 @@ def register_routes(app):",
        "     a = 1",
        "     b = 2",
        "     c = 3",
        '+    app.add_url_rule("/metrics", "metrics", metrics)',
        '+    app.add_url_rule("/live", "live", live)',
        "     d = 4",
        "     e = 5",
        "     f = 6",
    ]
)

# A full multi-file unified diff: a modified .py (module assignment + new def)
# and a pure rename. Mirrors `git diff --unified=3 -M`.
SAMPLE_DIFF = (
    "diff --git a/src/app/routes.py b/src/app/routes.py\n"
    "index 39a5add..f80a20a 100644\n"
    "--- a/src/app/routes.py\n"
    "+++ b/src/app/routes.py\n"
    "@@ -1,8 +1,15 @@\n"
    " import os\n"
    " import sys\n"
    " \n"
    '+ROUTES = ["health", "ready"]\n'
    "+\n"
    " \n"
    " def register_routes(app):\n"
    '     app.add_url_rule("/health", "health", health)\n'
    '     app.add_url_rule("/ready", "ready", ready)\n'
    '+    app.add_url_rule("/metrics", "metrics", metrics)\n'
    "     return app\n"
    "+\n"
    "+\n"
    "+def new_handler():\n"
    '+    return "ok"\n'
    "diff --git a/src/old.py b/src/renamed.py\n"
    "similarity index 100%\n"
    "rename from src/old.py\n"
    "rename to src/renamed.py\n"
)


# ---------------------------------------------------------------------------
# extract_changed_symbols
# ---------------------------------------------------------------------------


def test_symbols_from_heading_and_added_lines():
    """Heading def + added module assignment + added def, with no full file text."""
    hunk = "\n".join(
        [
            "@@ -10,6 +10,8 @@ def register_routes(app):",
            "     existing = 1",
            '+ROUTES = ["health", "ready"]',
            "+def new_handler():",
            '+    return "ok"',
        ]
    )
    symbols = extract_changed_symbols("src/app/routes.py", [hunk])
    # Heading symbol comes first, then symbols found in the added lines.
    assert symbols[0] == "register_routes"
    assert "ROUTES" in symbols
    assert "new_handler" in symbols


def test_symbols_from_real_heading_hunk():
    symbols = extract_changed_symbols("f.py", [HUNK_WITH_HEADING])
    # Only the enclosing function is named; the added lines are plain calls.
    assert symbols == ["register_routes"]


def test_symbols_capture_class_and_async_def():
    hunk = "\n".join(
        [
            "@@ -1,2 +1,5 @@",
            "+class Widget:",
            "+    pass",
            "+async def fetch():",
            "+    return 1",
        ]
    )
    symbols = extract_changed_symbols("m.py", [hunk])
    assert symbols == ["Widget", "fetch"]


def test_symbols_module_assignment_only_top_level():
    """Indented assignments are locals, not module-level symbols."""
    hunk = "\n".join(
        [
            "@@ -1,3 +1,5 @@",
            "+TOP_LEVEL = 1",
            "+    indented = 2",
            "+ALSO: list = []",
        ]
    )
    symbols = extract_changed_symbols("m.py", [hunk])
    assert "TOP_LEVEL" in symbols
    assert "ALSO" in symbols
    assert "indented" not in symbols


def test_symbols_from_removed_lines():
    hunk = "\n".join(
        [
            "@@ -1,3 +1,1 @@",
            "-def deleted_fn():",
            "-    return 1",
            " kept = 0",
        ]
    )
    symbols = extract_changed_symbols("m.py", [hunk])
    assert symbols == ["deleted_fn"]


def test_symbols_dedup_order_preserving():
    hunk_a = "@@ -1,1 +1,2 @@ def foo():\n+x = 1"
    hunk_b = "@@ -5,1 +6,2 @@ def foo():\n+def bar():"
    symbols = extract_changed_symbols("m.py", [hunk_a, hunk_b])
    # `foo` appears in both headings but is recorded once, first.
    assert symbols == ["foo", "x", "bar"]


def test_symbols_skips_equals_comparison():
    """`==` must not be mistaken for an assignment target."""
    hunk = "@@ -1,1 +1,1 @@\n+CONST == other"
    assert extract_changed_symbols("m.py", [hunk]) == []


def test_symbols_non_python_returns_empty():
    assert extract_changed_symbols("README.md", [HUNK_WITH_HEADING]) == []
    assert extract_changed_symbols("config.yaml", [HUNK_WITH_HEADING]) == []


def test_symbols_empty_hunks():
    assert extract_changed_symbols("m.py", []) == []


# ---------------------------------------------------------------------------
# parse_patchset (the unidiff parse path of diff_local, sans subprocess)
# ---------------------------------------------------------------------------


def test_parse_patchset_files_status_hunks_symbols():
    cd = parse_patchset(
        SAMPLE_DIFF,
        repo="owner/name",
        base="base123",
        head="head456",
        pr_number=42,
        pr_title="Add metrics route",
    )

    assert cd.repo == "owner/name"
    assert cd.base_sha == "base123"
    assert cd.head_sha == "head456"
    assert cd.pr_number == 42
    assert cd.pr_title == "Add metrics route"
    assert len(cd.files) == 2

    by_path = {f.path: f for f in cd.files}

    routes = by_path["src/app/routes.py"]
    assert routes.status is FileStatus.MODIFIED
    assert routes.previous_path is None
    assert len(routes.hunks) == 1
    assert routes.hunks[0].startswith("@@ -1,8 +1,15 @@")
    # Module-level ROUTES assignment and the new_handler def are captured.
    assert "ROUTES" in routes.changed_symbols
    assert "new_handler" in routes.changed_symbols

    renamed = by_path["src/renamed.py"]
    assert renamed.status is FileStatus.RENAMED
    assert renamed.previous_path == "src/old.py"
    # A pure (100% similarity) rename has no hunks and no symbols.
    assert renamed.hunks == []
    assert renamed.changed_symbols == []


def test_parse_patchset_added_file():
    diff = (
        "diff --git a/src/new_mod.py b/src/new_mod.py\n"
        "new file mode 100644\n"
        "index 0000000..1111111\n"
        "--- /dev/null\n"
        "+++ b/src/new_mod.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def brand_new():\n"
        "+    return 1\n"
    )
    cd = parse_patchset(diff, repo="o/r", base="a", head="b")
    assert len(cd.files) == 1
    f = cd.files[0]
    assert f.status is FileStatus.ADDED
    assert f.path == "src/new_mod.py"
    assert f.previous_path is None
    assert f.changed_symbols == ["brand_new"]


def test_parse_patchset_removed_file():
    diff = (
        "diff --git a/src/gone.py b/src/gone.py\n"
        "deleted file mode 100644\n"
        "index 1111111..0000000\n"
        "--- a/src/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-def old_fn():\n"
        "-    return 1\n"
    )
    cd = parse_patchset(diff, repo="o/r", base="a", head="b")
    assert len(cd.files) == 1
    f = cd.files[0]
    assert f.status is FileStatus.REMOVED
    assert f.path == "src/gone.py"
    assert "old_fn" in f.changed_symbols


def test_parse_patchset_empty_diff():
    cd = parse_patchset("", repo="o/r", base="a", head="b")
    assert cd.files == []
    assert cd.repo == "o/r"


# ---------------------------------------------------------------------------
# GitHub patch splitting (the diff_github hunk path, sans subprocess)
# ---------------------------------------------------------------------------


def test_split_github_patch_multiple_hunks():
    patch = (
        "@@ -1,3 +1,4 @@ def first():\n"
        " a = 1\n"
        "+b = 2\n"
        "@@ -10,2 +11,3 @@ def second():\n"
        " c = 3\n"
        "+d = 4\n"
    )
    hunks = _split_github_patch(patch)
    assert len(hunks) == 2
    assert hunks[0].startswith("@@ -1,3 +1,4 @@ def first():")
    assert hunks[1].startswith("@@ -10,2 +11,3 @@ def second():")


def test_split_github_patch_empty():
    assert _split_github_patch(None) == []
    assert _split_github_patch("") == []


def test_github_style_hunks_feed_symbol_extraction():
    """The GitHub patch path produces hunks symbol-extraction can read."""
    patch = (
        "@@ -1,2 +1,4 @@ def register_routes(app):\n"
        " existing = 1\n"
        '+ROUTES = ["a"]\n'
        "+def handler():\n"
    )
    hunks = _split_github_patch(patch)
    symbols = extract_changed_symbols("src/app/routes.py", hunks)
    assert symbols[0] == "register_routes"
    assert "ROUTES" in symbols
    assert "handler" in symbols
