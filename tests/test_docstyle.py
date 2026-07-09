"""Tests for the docstring-style utility (`docsync.docstyle`) and the read-only
`pysymbols.collect_docstrings` helper it builds on.

No network: `infer_style` is exercised with a fake client whose `.messages.parse(...)`
returns a scripted `DocstringStyleSpec` keyed on `output_format`. `collect_docstrings`
is pure stdlib `ast`, and the render/scaffold/write/config helpers are deterministic.
Tiny source repos are built under `tmp_path`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docsync import docstyle
from docsync.docstyle import (
    DEFAULT_STYLE_FILE,
    config_snippet,
    infer_style,
    render_style_markdown,
    sample_docstrings,
    scaffold_template,
    write_style_file,
)
from docsync.models import DocsyncConfig, DocstringStyleSpec
from docsync.pysymbols import collect_docstrings


# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, spec: DocstringStyleSpec, calls: list[dict]):
        self._spec = spec
        self._calls = calls

    def parse(self, *, output_format, **kwargs):
        assert output_format is DocstringStyleSpec
        self._calls.append(kwargs)
        return type(
            "Resp",
            (),
            {
                "parsed_output": self._spec,
                "usage": {"input_tokens": 30, "output_tokens": 12},
            },
        )()


class FakeClient:
    def __init__(self, spec: DocstringStyleSpec):
        self.calls: list[dict] = []
        self.messages = _FakeMessages(spec, self.calls)


def _write_repo(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    repo = tmp_path / name
    for rel, body in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(body)
    return repo


def _system_text(call: dict) -> str:
    """Flatten a recorded call's `system` (a list of text blocks) to one string."""
    system = call["system"]
    if isinstance(system, str):
        return system
    return "\n".join(b.get("text", "") for b in system)


# ---------------------------------------------------------------------------
# collect_docstrings — the read-only counterpart to iter_targets
# ---------------------------------------------------------------------------


_MIXED_SRC = '''\
def documented():
    """A documented function."""
    return 1


def undocumented():
    return 2


def _private():
    """A private but documented function."""
    return 3


class Widget:
    """A documented class."""

    def __init__(self):
        """Dunder docstring that should never surface."""
        self.x = 1
'''


def test_collect_docstrings_returns_public_documented_only():
    samples = collect_docstrings(_MIXED_SRC)
    by_qual = {s.qualname: s for s in samples}
    # Public documented symbols only.
    assert set(by_qual) == {"documented", "Widget"}
    # Undocumented public symbols are excluded.
    assert "undocumented" not in by_qual
    # Private excluded by default; dunder never included.
    assert "_private" not in by_qual
    assert "Widget.__init__" not in by_qual
    # The actual docstring text is captured.
    assert by_qual["documented"].docstring == "A documented function."
    assert by_qual["documented"].kind == "function"
    assert by_qual["Widget"].docstring == "A documented class."


def test_collect_docstrings_include_private():
    by_qual = {s.qualname: s for s in collect_docstrings(_MIXED_SRC, include_private=True)}
    assert "_private" in by_qual
    assert by_qual["_private"].docstring == "A private but documented function."
    # Dunder still never surfaces, even with private included.
    assert "Widget.__init__" not in by_qual


def test_collect_docstrings_module_target():
    with_doc = '"""Module-level docstring."""\n\n\ndef foo():\n    return 1\n'
    samples = collect_docstrings(with_doc, targets=("module",))
    assert len(samples) == 1
    assert samples[0].kind == "module"
    assert samples[0].qualname == "<module>"
    assert samples[0].docstring == "Module-level docstring."

    without_doc = "def foo():\n    return 1\n"
    assert collect_docstrings(without_doc, targets=("module",)) == []


def test_collect_docstrings_invalid_source_returns_empty():
    assert collect_docstrings("def f(:\n    pass\n") == []


# ---------------------------------------------------------------------------
# sample_docstrings — bounded, format-revealing sampling across repos
# ---------------------------------------------------------------------------


def test_sample_docstrings_caps_at_max_samples(tmp_path):
    # Build more documented functions than the cap.
    body = "".join(
        f'def fn{i}():\n    """Docstring number {i}."""\n    return {i}\n\n\n' for i in range(6)
    )
    repo = _write_repo(tmp_path, "demo", {"mod.py": body})

    samples = sample_docstrings([("demo", str(repo))], DocsyncConfig(), max_samples=3)
    assert len(samples) == 3


def test_sample_docstrings_prefers_section_rich(tmp_path):
    rich = (
        "def rich(a, b):\n"
        '    """Do a thing.\n'
        "\n"
        "    Args:\n"
        "        a: First.\n"
        "        b: Second.\n"
        "\n"
        "    Returns:\n"
        "        A result.\n"
        '    """\n'
        "    return a\n"
    )
    trivial = "".join(
        f'def trivial{i}():\n    """One liner {i}."""\n    return {i}\n\n\n' for i in range(4)
    )
    repo = _write_repo(tmp_path, "demo", {"mod.py": rich + "\n\n" + trivial})

    samples = sample_docstrings([("demo", str(repo))], DocsyncConfig(), max_samples=1)
    assert len(samples) == 1
    # The section-rich docstring sorts ahead of the one-liners.
    assert samples[0].qualname == "rich"


# ---------------------------------------------------------------------------
# infer_style — the LLM path (fake client)
# ---------------------------------------------------------------------------


def test_infer_style_happy_path(tmp_path):
    src = 'def greet(name):\n    """Say hello to someone."""\n    return name\n'
    repo = _write_repo(tmp_path, "demo", {"mod.py": src})
    spec = DocstringStyleSpec(
        name="house", guidance="Lead with a summary.", example="Do a thing."
    )
    client = FakeClient(spec)

    out_spec, samples, usage = infer_style([("demo", str(repo))], DocsyncConfig(), client=client)

    assert out_spec.name == "house"
    assert samples  # learned from at least one docstring
    # The judge model is used and metered under the docstring-style stage.
    assert client.calls[0]["model"] == DocsyncConfig().models.judge_model
    assert {m.stage for m in usage.by_model} == {"docstring-style"}
    # The sampled docstrings are threaded into the prompt.
    call = client.calls[0]
    combined = _system_text(call) + "\n" + call["messages"][0]["content"]
    assert "Existing docstrings sampled" in combined


def test_infer_style_raises_without_docstrings(tmp_path):
    src = "def undocumented():\n    return 1\n"
    repo = _write_repo(tmp_path, "demo", {"mod.py": src})
    spec = DocstringStyleSpec(name="x", guidance="y", example="z")
    client = FakeClient(spec)

    with pytest.raises(ValueError):
        infer_style([("demo", str(repo))], DocsyncConfig(), client=client)


# ---------------------------------------------------------------------------
# Rendering / scaffold / config helpers (no LLM)
# ---------------------------------------------------------------------------


def test_render_style_markdown_contains_spec_fields():
    spec = DocstringStyleSpec(
        name="terse-oneline",
        guidance="Keep it to one imperative line.",
        example="Fetch a user by id.",
    )
    text = render_style_markdown(spec)
    assert spec.name in text
    assert spec.guidance in text
    assert spec.example in text
    assert "Example (docstring body only" in text


def test_scaffold_template_is_editable_stub():
    text = scaffold_template()
    assert "my-house-style" in text


def test_config_snippet_activates_custom_format():
    snippet = config_snippet()
    assert "format: custom" in snippet
    assert DEFAULT_STYLE_FILE in snippet


# ---------------------------------------------------------------------------
# write_style_file
# ---------------------------------------------------------------------------


def test_write_style_file_dry_run_does_not_write(tmp_path):
    path = write_style_file(tmp_path, "hi", out=".docsync/docstring_style.md", dry_run=True)
    assert path == tmp_path / ".docsync" / "docstring_style.md"
    assert not path.exists()


def test_write_style_file_writes_and_creates_parents(tmp_path):
    path = write_style_file(tmp_path, "hi", out=".docsync/docstring_style.md", dry_run=False)
    assert path.exists()
    assert path.read_text() == "hi"


def test_default_style_file_constant():
    assert docstyle.DEFAULT_STYLE_FILE == ".docsync/docstring_style.md"
