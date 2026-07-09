"""Tests for the code-level docstring orchestrator (`docsync.docstrings`).

No network: a fake client whose `.messages.parse(...)` returns a scripted
`FileDocstrings` keyed on `output_format`, echoing back only the docstrings whose
qualname appears in the generate prompt. Covers backfill + diff-mode scoping, the
model/stage metering, symbols the model declines, custom-format injection, and the
overwrite_existing target gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

from docsync.docstrings import run_docstrings, write_docstrings
from docsync.models import (
    ChangedFile,
    CodeDiff,
    DocsyncConfig,
    FileDocstrings,
    FileStatus,
    SymbolDocstring,
)


# ---------------------------------------------------------------------------
# Fakes & helpers
# ---------------------------------------------------------------------------


class _FakeMessages:
    def __init__(self, docstrings_by_qual: dict[str, str], calls: list[dict]):
        self._map = docstrings_by_qual
        self._calls = calls

    def parse(self, *, output_format, **kwargs):
        assert output_format is FileDocstrings
        self._calls.append(kwargs)
        # The user prompt lists "qualname: X" lines; return docstrings for the
        # symbols we know that were actually asked for in this file's prompt.
        user = kwargs["messages"][0]["content"]
        items = [
            SymbolDocstring(qualname=q, docstring=d)
            for q, d in self._map.items()
            if f"qualname: {q}" in user
        ]
        return type(
            "Resp",
            (),
            {
                "parsed_output": FileDocstrings(items=items),
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )()


class FakeClient:
    def __init__(self, docstrings_by_qual: dict[str, str]):
        self.calls: list[dict] = []
        self.messages = _FakeMessages(docstrings_by_qual, self.calls)


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


def _docstring_of(text: str, qualname: str) -> str | None:
    """The docstring of `qualname` (dotted path) in parsed source `text`, or None."""
    node: ast.AST = ast.parse(text)
    for part in qualname.split("."):
        node = next(n for n in node.body if getattr(n, "name", None) == part)
    return ast.get_docstring(node)


# A module with an undocumented public function and a documented class whose
# method is undocumented — so exactly `add` and `Calc.mul` are backfill targets.
_BACKFILL_SRC = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "class Calc:\n"
    '    """A calculator."""\n'
    "\n"
    "    def mul(self, a, b):\n"
    "        return a * b\n"
)


# ---------------------------------------------------------------------------
# Backfill happy path
# ---------------------------------------------------------------------------


def test_backfill_documents_and_writes(tmp_path):
    repo = _write_repo(tmp_path, "demo", {"calc.py": _BACKFILL_SRC})
    client = FakeClient({"add": "Add two numbers.", "Calc.mul": "Multiply."})

    result = run_docstrings([("demo", str(repo))], DocsyncConfig(), client=client)

    by_qual = {o.qualname: o.status for o in result.outcomes}
    assert by_qual["add"] == "documented"
    assert by_qual["Calc.mul"] == "documented"
    assert result.changed_paths() == ["calc.py"]
    assert result.usage.calls == 1  # one file -> one generate call

    abs_path = str(repo / "calc.py")
    original = (repo / "calc.py").read_text()

    # Dry run reports the path but leaves the file untouched.
    assert write_docstrings(result, dry_run=True) == [abs_path]
    assert (repo / "calc.py").read_text() == original

    # A real write splices the docstrings into the source.
    assert write_docstrings(result, dry_run=False) == [abs_path]
    written = (repo / "calc.py").read_text()
    assert written != original
    assert _docstring_of(written, "add") == "Add two numbers."
    assert _docstring_of(written, "Calc.mul") == "Multiply."


def test_uses_edit_model_and_docstring_stage(tmp_path):
    repo = _write_repo(tmp_path, "demo", {"calc.py": _BACKFILL_SRC})
    client = FakeClient({"add": "Add two numbers.", "Calc.mul": "Multiply."})

    result = run_docstrings([("demo", str(repo))], DocsyncConfig(), client=client)

    assert client.calls[0]["model"] == DocsyncConfig().models.edit_model
    assert {m.stage for m in result.usage.by_model} == {"docstring"}


# ---------------------------------------------------------------------------
# Symbols the model declines
# ---------------------------------------------------------------------------


def test_declined_symbol_is_no_change_but_file_still_written(tmp_path):
    repo = _write_repo(tmp_path, "demo", {"calc.py": _BACKFILL_SRC})
    # Only `add` gets a docstring; the model returns nothing for `Calc.mul`.
    client = FakeClient({"add": "Add two numbers."})

    result = run_docstrings([("demo", str(repo))], DocsyncConfig(), client=client)

    by_qual = {o.qualname: o.status for o in result.outcomes}
    assert by_qual["add"] == "documented"
    assert by_qual["Calc.mul"] == "no_change"

    # The file still changes — `add` succeeded — so it is written.
    abs_path = str(repo / "calc.py")
    assert write_docstrings(result, dry_run=False) == [abs_path]
    written = (repo / "calc.py").read_text()
    assert _docstring_of(written, "add") == "Add two numbers."
    assert _docstring_of(written, "Calc.mul") is None


# ---------------------------------------------------------------------------
# Diff mode scopes to touched symbols
# ---------------------------------------------------------------------------


def test_diff_mode_scopes_to_touched_symbols(tmp_path):
    src = (
        "def alpha():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def beta():\n"
        "    return 2\n"
    )
    repo = _write_repo(tmp_path, "demo", {"api.py": src})
    client = FakeClient({"alpha": "Return one.", "beta": "Return two."})
    diff = CodeDiff(
        repo="demo",
        base_sha="base",
        head_sha="head",
        files=[
            ChangedFile(
                path="api.py", status=FileStatus.MODIFIED, changed_symbols=["alpha"]
            )
        ],
    )

    result = run_docstrings([("demo", str(repo))], DocsyncConfig(), client=client, diff=diff)

    quals = {o.qualname for o in result.outcomes}
    assert quals == {"alpha"}
    assert [o.qualname for o in result.documented()] == ["alpha"]


# ---------------------------------------------------------------------------
# Custom format spec injection
# ---------------------------------------------------------------------------


def test_custom_format_prompt_is_injected(tmp_path):
    repo = _write_repo(tmp_path, "demo", {"calc.py": _BACKFILL_SRC})
    config = DocsyncConfig()
    config.docstrings.format = "custom"
    config.docstrings.style_prompt = "MY-CUSTOM-FORMAT-XYZ"
    client = FakeClient({"add": "Add two numbers.", "Calc.mul": "Multiply."})

    run_docstrings([("demo", str(repo))], config, client=client)

    assert "MY-CUSTOM-FORMAT-XYZ" in _system_text(client.calls[0])


# ---------------------------------------------------------------------------
# overwrite_existing target gate
# ---------------------------------------------------------------------------


def test_overwrite_existing_gates_documented_symbols(tmp_path):
    src = (
        "def greet(name):\n"
        '    """Say hi."""\n'
        "    return name\n"
    )
    repo = _write_repo(tmp_path, "demo", {"mod.py": src})
    client = FakeClient({"greet": "Greet a person by name."})

    # Default config: the only symbol already has a docstring -> nothing to do.
    default_result = run_docstrings([("demo", str(repo))], DocsyncConfig(), client=client)
    assert default_result.documented() == []

    # With overwrite_existing, the documented symbol becomes a target again.
    config = DocsyncConfig()
    config.docstrings.overwrite_existing = True
    overwrite_result = run_docstrings([("demo", str(repo))], config, client=client)
    assert [o.qualname for o in overwrite_result.documented()] == ["greet"]
