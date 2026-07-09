"""Tests for the deterministic docstring machinery (`docsync.pysymbols`).

No network and no LLM: this module is pure stdlib `ast`. The three jobs are exercised
directly — locate (`iter_targets`), render/splice (`render_docstring`, `splice_docstring`,
`splice_docstrings`), and the safety gate (`assert_only_docstrings_changed`) — plus the
pluggable `resolve_style` registry. Assertions prefer reparsing with `ast.get_docstring`
over brittle string matches, with one golden-string check for single-line rendering.
"""

from __future__ import annotations

import ast

import pytest

from docsync.pysymbols import (
    assert_only_docstrings_changed,
    iter_targets,
    render_docstring,
    resolve_style,
    splice_docstring,
    splice_docstrings,
    symbol_source,
)

# A source with every locator case: a public function, a public class with a public
# method, a private method, a dunder method, and an already-documented function.
_SAMPLE = '''\
def foo():
    return 1


class Bar:
    def run(self):
        return 2

    def _private(self):
        return 3

    def __dunder__(self):
        return 4


def documented():
    """Already has one."""
    return 5
'''


def _targets_by_qualname(source, **kwargs):
    return {t.qualname: t for t in iter_targets(source, **kwargs)}


def _docstring_of(source, qualname):
    """Reparse `source` and return `ast.get_docstring` for a top-level or method name."""
    tree = ast.parse(source)
    top, _, sub = qualname.partition(".")
    for node in tree.body:
        if getattr(node, "name", None) != top:
            continue
        if not sub:
            return ast.get_docstring(node)
        for child in node.body:
            if getattr(child, "name", None) == sub:
                return ast.get_docstring(child)
    raise AssertionError(f"symbol {qualname!r} not found")


# ---------------------------------------------------------------------------
# iter_targets — the locator
# ---------------------------------------------------------------------------


def test_iter_targets_finds_public_symbols_and_skips_dunders():
    got = _targets_by_qualname(_SAMPLE)
    assert set(got) == {"foo", "Bar", "Bar.run"}
    # dunders never surface, even with everything opened up.
    everything = _targets_by_qualname(
        _SAMPLE, include_private=True, overwrite=True
    )
    assert "Bar.__dunder__" not in everything


def test_iter_targets_private_excluded_by_default_included_when_requested():
    assert "Bar._private" not in _targets_by_qualname(_SAMPLE)
    assert "Bar._private" in _targets_by_qualname(_SAMPLE, include_private=True)


def test_iter_targets_documented_excluded_by_default_included_on_overwrite():
    assert "documented" not in _targets_by_qualname(_SAMPLE)
    with_overwrite = _targets_by_qualname(_SAMPLE, overwrite=True)
    assert "documented" in with_overwrite
    assert with_overwrite["documented"].has_docstring is True
    assert with_overwrite["documented"].existing_span is not None


def test_iter_targets_kinds_filter():
    only_funcs = _targets_by_qualname(_SAMPLE, targets=("function",))
    assert set(only_funcs) == {"foo"}
    for t in only_funcs.values():
        assert t.kind == "function"


def test_iter_targets_module_kind():
    # _SAMPLE has no module docstring, so a module target is offered.
    mods = [t for t in iter_targets(_SAMPLE, targets=("module",)) if t.kind == "module"]
    assert len(mods) == 1
    assert mods[0].qualname == "<module>"

    documented_module = '"""Module doc."""\n\n\ndef foo():\n    return 1\n'
    # With an existing module docstring and no overwrite, no module target.
    assert not [
        t
        for t in iter_targets(documented_module, targets=("module",))
        if t.kind == "module"
    ]
    # overwrite re-offers it.
    assert [
        t
        for t in iter_targets(
            documented_module, targets=("module",), overwrite=True
        )
        if t.kind == "module"
    ]


def test_iter_targets_body_position_indentation():
    got = _targets_by_qualname(_SAMPLE)
    # A top-level def body is indented 4; a method body 8.
    assert got["foo"].body_col == 4
    assert got["Bar.run"].body_col == 8
    # body_lineno points at the first body statement, not the def line.
    assert got["foo"].body_lineno == 2  # `    return 1`


def test_iter_targets_invalid_source_returns_empty():
    assert iter_targets("def f(:\n    pass\n") == []


def test_iter_targets_inline_body_flagged():
    got = _targets_by_qualname("def one(): return 1\n")
    assert got["one"].inline_body is True


def test_symbol_source_returns_span_lines():
    got = _targets_by_qualname(_SAMPLE)
    src = symbol_source(_SAMPLE, got["foo"])
    assert src == "def foo():\n    return 1"


# ---------------------------------------------------------------------------
# render_docstring
# ---------------------------------------------------------------------------


def test_render_docstring_single_line_golden():
    assert render_docstring("summary", "    ") == ['    """summary"""']


def test_render_docstring_multiline_shape():
    lines = render_docstring("first line\nsecond line", "    ")
    assert lines[0] == '    """first line'
    assert lines[1] == "    second line"
    assert lines[-1] == '    """'


def test_render_docstring_escapes_triple_quotes_and_still_parses():
    body = 'He said """hi""" loudly'
    src = "def f():\n" + "\n".join(render_docstring(body, "    ")) + "\n    return 1\n"
    tree = ast.parse(src)  # must not raise
    assert ast.get_docstring(tree.body[0]) is not None


# ---------------------------------------------------------------------------
# splice_docstring / splice_docstrings
# ---------------------------------------------------------------------------


def test_splice_into_undocumented_function():
    src = "def foo():\n    return 1\n"
    target = _targets_by_qualname(src)["foo"]
    out = splice_docstring(src, target, "Return one.")
    ast.parse(out)  # valid
    assert _docstring_of(out, "foo") == "Return one."


def test_splice_replaces_existing_docstring_without_duplicating():
    target = _targets_by_qualname(_SAMPLE, overwrite=True)["documented"]
    out = splice_docstring(_SAMPLE, target, "Fresh summary.")
    assert _docstring_of(out, "documented") == "Fresh summary."
    assert "Already has one." not in out
    # exactly one docstring literal for `documented`.
    tree = ast.parse(out)
    node = next(n for n in tree.body if getattr(n, "name", None) == "documented")
    strings = [
        s
        for s in node.body
        if isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Constant)
        and isinstance(s.value.value, str)
    ]
    assert len(strings) == 1


def test_splice_docstrings_multiple_targets_all_applied():
    got = _targets_by_qualname(_SAMPLE)
    pairs = [
        (got["foo"], "Return one."),
        (got["Bar"], "The Bar class."),
        (got["Bar.run"], "Run and return two."),
    ]
    out, applied = splice_docstrings(_SAMPLE, pairs)
    assert {t.qualname for t in applied} == {"foo", "Bar", "Bar.run"}
    ast.parse(out)
    assert _docstring_of(out, "foo") == "Return one."
    assert _docstring_of(out, "Bar") == "The Bar class."
    assert _docstring_of(out, "Bar.run") == "Run and return two."


def test_splice_docstrings_bottom_up_order_independent():
    # Feed the pairs top-down; the splicer must still apply bottom-up internally so a
    # later insertion below never invalidates an earlier target's line numbers.
    got = _targets_by_qualname(_SAMPLE)
    forward = [
        (got["foo"], "aaa"),
        (got["Bar"], "bbb"),
        (got["Bar.run"], "ccc"),
    ]
    out_forward, _ = splice_docstrings(_SAMPLE, forward)
    out_reverse, _ = splice_docstrings(_SAMPLE, list(reversed(forward)))
    assert out_forward == out_reverse
    assert _docstring_of(out_forward, "foo") == "aaa"
    assert _docstring_of(out_forward, "Bar.run") == "ccc"


def test_splice_method_indentation():
    got = _targets_by_qualname(_SAMPLE)
    out = splice_docstring(_SAMPLE, got["Bar.run"], "Run it.")
    assert '        """Run it."""' in out


def test_splice_inline_body_raises():
    target = _targets_by_qualname("def one(): return 1\n")["one"]
    with pytest.raises(ValueError):
        splice_docstring("def one(): return 1\n", target, "Return one.")


# ---------------------------------------------------------------------------
# assert_only_docstrings_changed — the safety gate
# ---------------------------------------------------------------------------


def test_gate_passes_for_clean_splice():
    src = "def foo():\n    return 1\n"
    target = _targets_by_qualname(src)["foo"]
    out = splice_docstring(src, target, "Return one.")
    assert_only_docstrings_changed(src, out)  # must not raise


def test_gate_raises_when_code_line_changed():
    src = "def foo():\n    return 1\n"
    target = _targets_by_qualname(src)["foo"]
    out = splice_docstring(src, target, "Return one.")
    # Corrupt a real line of code on top of the (otherwise clean) splice.
    corrupted = out.replace("return 1", "return 2")
    with pytest.raises(ValueError):
        assert_only_docstrings_changed(src, corrupted)


def test_gate_raises_when_modified_does_not_parse():
    src = "def foo():\n    return 1\n"
    with pytest.raises(ValueError):
        assert_only_docstrings_changed(src, "def foo(:\n    return 1\n")


# ---------------------------------------------------------------------------
# resolve_style — the pluggable format registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["google", "numpy", "rest"])
def test_resolve_style_builtins(fmt):
    style = resolve_style(fmt)
    assert style.name == fmt
    assert style.prompt_fragment


def test_resolve_style_custom():
    style = resolve_style("custom", custom_prompt="MY FORMAT")
    assert style.name == "custom"
    assert style.prompt_fragment == "MY FORMAT"


def test_resolve_style_custom_without_prompt_raises():
    with pytest.raises(ValueError):
        resolve_style("custom")


def test_resolve_style_unknown_falls_back_to_google():
    assert resolve_style("klingon").name == "google"
