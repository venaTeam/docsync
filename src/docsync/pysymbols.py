"""Deterministic Python-source machinery for the docstring stage.

This module never calls an LLM. It does the three deterministic jobs around docstring
generation, so the model only ever writes docstring *text*:

    locate   → find the public symbols that want a docstring, with the exact source
               positions needed to place one (`iter_targets`).
    splice   → insert/replace docstrings in the raw file text at those positions,
               touching only the docstring lines (`splice_docstrings`).
    validate → prove we changed *only* docstrings by comparing docstring-stripped ASTs
               (`assert_only_docstrings_changed`).

`ast` is used purely to *locate* — we splice into the original file text, so every byte
outside the docstring lines is preserved (comments, spacing, quote styles) without the
byte-perfect round-trip a concrete-syntax-tree library (LibCST) would provide. That is a
deliberate MVP trade-off: no new dependency, and the operation is trivially localizable
because a docstring is always the first statement of a body.

Format is pluggable via the small `DocstringStyle` registry: `google` is the MVP; the
prompt fragment for each style is injected into the generate prompt. Placement is
format-agnostic (always a triple-quoted first body statement), so a user-defined format
needs no code here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Located symbols
# ---------------------------------------------------------------------------


@dataclass
class SymbolTarget:
    """One symbol that can receive a docstring, with the positions to place it.

    Line numbers are 1-based (as `ast` reports). `body_lineno`/`body_col` point at the
    first statement of the body — the insertion anchor and the indentation to match.
    `existing_span` is the inclusive (start, end) 1-based line range of the current
    docstring literal when `has_docstring`, else None.
    """

    qualname: str
    name: str
    kind: str  # "module" | "class" | "function" | "method"
    signature: str
    body_lineno: int
    body_col: int
    has_docstring: bool
    existing_span: tuple[int, int] | None = None
    # Inclusive 1-based line range of the symbol's own source (def/class .. end), for the
    # generate prompt. The module target spans the whole file.
    source_span: tuple[int, int] = (1, 1)
    # True when the def and its first body statement share a line (`def f(): return 1`).
    # The splicer can't place a docstring line here, so these are reported and skipped.
    inline_body: bool = False
    decorators: list[str] = field(default_factory=list)


def _signature(node: ast.AST) -> str:
    """A one-line signature for a top-level function or class node.

    Mirrors ``bootstrap._signature`` so surface strings read the same across stages.
    """
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}" + (f"({bases})" if bases else "")
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _docstring_span(node: ast.AST) -> tuple[int, int] | None:
    """(start, end) 1-based line range of a node's docstring literal, or None."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return (first.lineno, first.end_lineno or first.lineno)
    return None


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _wanted(name: str, *, include_private: bool) -> bool:
    """Public-symbol filter: dunders always out; `_private` out unless included."""
    if _is_dunder(name):
        return False
    if name.startswith("_") and not include_private:
        return False
    return True


def iter_targets(
    source_text: str,
    *,
    include_private: bool = False,
    targets: list[str] | tuple[str, ...] = ("class", "function", "method"),
    overwrite: bool = False,
) -> list[SymbolTarget]:
    """Locate documentable symbols in a Python source file.

    Walks top-level functions/classes and one level of methods inside classes. Applies
    the public-symbol filter (`include_private`), the requested `kinds` (`targets`), and
    skips symbols that already have a docstring unless `overwrite`. Returns targets in
    source order; a syntactically invalid file yields an empty list.

    The `module` kind (a file-level docstring) is included only when "module" is in
    `targets`; it is reported with `body_lineno` at the first real statement.
    """
    kinds = set(targets)
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError):
        return []

    out: list[SymbolTarget] = []

    if "module" in kinds:
        mod = _module_target(tree, overwrite=overwrite)
        if mod is not None:
            out.append(mod)

    for node in tree.body:
        if isinstance(node, _FUNC_TYPES) and "function" in kinds:
            t = _def_target(node, node.name, "function", include_private, overwrite)
            if t is not None:
                out.append(t)
        elif isinstance(node, ast.ClassDef):
            if "class" in kinds:
                t = _def_target(node, node.name, "class", include_private, overwrite)
                if t is not None:
                    out.append(t)
            if "method" in kinds:
                for sub in node.body:
                    if isinstance(sub, _FUNC_TYPES):
                        t = _def_target(
                            sub,
                            f"{node.name}.{sub.name}",
                            "method",
                            include_private,
                            overwrite,
                        )
                        if t is not None:
                            out.append(t)
    return out


def _module_target(tree: ast.Module, *, overwrite: bool) -> SymbolTarget | None:
    if not tree.body:
        return None
    existing = _docstring_span(tree)
    if existing and not overwrite:
        return None
    anchor = tree.body[0]
    end = max((getattr(n, "end_lineno", n.lineno) for n in tree.body), default=1)
    return SymbolTarget(
        qualname="<module>",
        name="<module>",
        kind="module",
        signature="module",
        body_lineno=anchor.lineno,
        body_col=anchor.col_offset,  # 0 for a module
        has_docstring=existing is not None,
        existing_span=existing,
        source_span=(1, end),
    )


def _def_target(
    node: ast.AST,
    qualname: str,
    kind: str,
    include_private: bool,
    overwrite: bool,
) -> SymbolTarget | None:
    name = node.name  # type: ignore[attr-defined]
    if not _wanted(name, include_private=include_private):
        return None
    existing = _docstring_span(node)
    if existing and not overwrite:
        return None
    body = node.body  # type: ignore[attr-defined]
    anchor = body[0]
    inline = anchor.lineno == node.lineno  # `def f(): return 1`
    return SymbolTarget(
        qualname=qualname,
        name=name,
        kind=kind,
        signature=_signature(node),
        body_lineno=anchor.lineno,
        body_col=anchor.col_offset,
        has_docstring=existing is not None,
        existing_span=existing,
        source_span=(node.lineno, getattr(node, "end_lineno", node.lineno)),
        inline_body=inline,
        decorators=[ast.unparse(d) for d in getattr(node, "decorator_list", [])],
    )


def symbol_source(source_text: str, target: SymbolTarget) -> str:
    """The raw source lines of a target (its `source_span`), for the generate prompt."""
    lines = source_text.splitlines()
    start, end = target.source_span
    return "\n".join(lines[start - 1 : end])


@dataclass
class DocSample:
    """One existing docstring, sampled from source to learn a repo's house style."""

    qualname: str
    kind: str  # "module" | "class" | "function" | "method"
    signature: str
    docstring: str


def collect_docstrings(
    source_text: str,
    *,
    include_private: bool = False,
    targets: list[str] | tuple[str, ...] = ("class", "function", "method"),
) -> list[DocSample]:
    """Extract the *existing* docstrings of public symbols in a Python source file.

    The read-only counterpart to `iter_targets`: where that finds symbols *lacking* a
    docstring to write one, this returns the docstrings symbols *already have* so the
    `docstring-style` utility can distil the repo's house format. Applies the same
    public-symbol filter and `kinds` selection; a syntactically invalid file yields [].
    """
    kinds = set(targets)
    try:
        tree = ast.parse(source_text)
    except (SyntaxError, ValueError):
        return []

    out: list[DocSample] = []

    if "module" in kinds:
        doc = ast.get_docstring(tree)
        if doc:
            out.append(DocSample("<module>", "module", "module", doc.strip()))

    def sample(node: ast.AST, qualname: str, kind: str) -> None:
        if not _wanted(node.name, include_private=include_private):  # type: ignore[attr-defined]
            return
        doc = ast.get_docstring(node)
        if doc:
            out.append(DocSample(qualname, kind, _signature(node), doc.strip()))

    for node in tree.body:
        if isinstance(node, _FUNC_TYPES) and "function" in kinds:
            sample(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            if "class" in kinds:
                sample(node, node.name, "class")
            if "method" in kinds:
                for sub in node.body:
                    if isinstance(sub, _FUNC_TYPES):
                        sample(sub, f"{node.name}.{sub.name}", "method")
    return out


# ---------------------------------------------------------------------------
# Rendering + splicing docstrings into source text
# ---------------------------------------------------------------------------


def render_docstring(body: str, indent: str) -> list[str]:
    """Render a docstring literal as a list of physical lines at `indent`.

    A single-line body becomes one triple-quoted summary line. A multi-line body opens on
    the quote line, indents each content line, and closes the quotes on their own line —
    the Google/PEP-257 shape. Escapes any embedded triple-double-quote so the literal
    stays valid; a body ending in a double-quote gets a space before the closing quotes.
    """
    text = body.strip("\n").rstrip()
    text = text.replace('"""', '\\"\\"\\"')
    lines = text.split("\n")
    if len(lines) == 1 and lines[0]:
        one = lines[0]
        tail = " " if one.endswith('"') else ""
        return [f'{indent}"""{one}{tail}"""']
    rendered = [f'{indent}"""{lines[0]}'.rstrip()]
    for line in lines[1:]:
        rendered.append(f"{indent}{line}".rstrip() if line else "")
    rendered.append(f'{indent}"""')
    return rendered


def splice_docstring(source_text: str, target: SymbolTarget, docstring: str) -> str:
    """Return `source_text` with `target`'s docstring set to `docstring`.

    Replaces the existing docstring literal (when `has_docstring`) or inserts a new one
    immediately before the first body statement, indented to match it. Everything else in
    the file is left byte-for-byte unchanged. Raises `ValueError` for an `inline_body`
    target (no line to place a docstring on) — callers should skip those.
    """
    if target.inline_body:
        raise ValueError(f"cannot splice a docstring into an inline body: {target.qualname}")

    newline = "\r\n" if "\r\n" in source_text else "\n"
    lines = source_text.split(newline)
    indent = " " * target.body_col
    block = render_docstring(docstring, indent)

    if target.has_docstring and target.existing_span is not None:
        start, end = target.existing_span
        lines[start - 1 : end] = block
    else:
        insert_at = target.body_lineno - 1  # 0-based line of the first body statement
        lines[insert_at:insert_at] = block
    return newline.join(lines)


def splice_docstrings(
    source_text: str, pairs: list[tuple[SymbolTarget, str]]
) -> tuple[str, list[SymbolTarget]]:
    """Apply many docstrings to one file, bottom-up so line numbers stay valid.

    `pairs` is (target, docstring). Splices are applied highest-line-first, so an earlier
    target's positions are never invalidated by a later insertion below it. Returns the
    new file text and the list of targets that were actually spliced (inline bodies are
    skipped and returned separately by the caller via `inline_body`).
    """
    applied: list[SymbolTarget] = []
    text = source_text
    ordered = sorted(pairs, key=lambda p: p[0].body_lineno, reverse=True)
    for target, docstring in ordered:
        if target.inline_body:
            continue
        text = splice_docstring(text, target, docstring)
        applied.append(target)
    return text, applied


# ---------------------------------------------------------------------------
# Validation — prove only docstrings changed
# ---------------------------------------------------------------------------


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove every docstring `Expr` node in-place and return the tree."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        if not isinstance(node, (ast.Module, ast.ClassDef, *_FUNC_TYPES)):
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body.pop(0)
            # A body reduced to empty (a stub that was only a docstring) needs a filler
            # so the comparison stays valid on both sides; use `...` on both.
            if not body:
                body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
    return tree


def assert_only_docstrings_changed(original: str, modified: str) -> None:
    """Raise `ValueError` unless `modified` differs from `original` only in docstrings.

    Parses both, strips every docstring literal, and compares `ast.dump()` (which ignores
    line/column positions by default). Equal dumps prove the splice touched nothing but
    docstrings; any structural difference — or a `modified` that no longer parses — is a
    splicing bug and the file must not be written.
    """
    try:
        new_tree = ast.parse(modified)
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise ValueError(f"modified source no longer parses: {exc}") from exc
    old_dump = ast.dump(_strip_docstrings(ast.parse(original)))
    new_dump = ast.dump(_strip_docstrings(new_tree))
    if old_dump != new_dump:
        raise ValueError("splice altered code structure, not only docstrings")


# ---------------------------------------------------------------------------
# Docstring style registry — the pluggable format
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocstringStyle:
    """A named docstring format: a prompt fragment the generate stage injects."""

    name: str
    prompt_fragment: str


_GOOGLE = DocstringStyle(
    name="google",
    prompt_fragment=(
        "Write Google-style docstrings.\n"
        "- Line 1 is a one-sentence imperative summary that fits on one line.\n"
        "- Optionally a blank line then a short description of behavior/why.\n"
        "- Then, only the sections the symbol needs, in this order, each separated by a\n"
        "  blank line: `Args:`, `Returns:` (or `Yields:` for generators), `Raises:`.\n"
        "- Under `Args:`, one entry per parameter as `name: description` (no types — the\n"
        "  signature already carries them); never list `self`/`cls`.\n"
        "- Under `Raises:`, one entry per exception as `ExceptionType: when it is raised`.\n"
        "- Omit `Returns:` when the function returns None; omit any empty section.\n"
        "- For a class, summarize the class; document instance state under `Attributes:`\n"
        "  when it helps. Do not document `__init__` here.\n"
        "Example body (text only, no surrounding quotes):\n"
        "Fetch a user by id.\n\n"
        "Args:\n"
        "    user_id: Primary key of the user to load.\n"
        "    strict: Raise instead of returning None when absent.\n\n"
        "Returns:\n"
        "    The matching User, or None when not found and strict is False.\n\n"
        "Raises:\n"
        "    LookupError: If strict is True and no user matches."
    ),
)

_NUMPY = DocstringStyle(
    name="numpy",
    prompt_fragment=(
        "Write NumPy-style docstrings: a one-line summary, then underlined sections —\n"
        "`Parameters`, `Returns`, `Raises` — where each header is underlined with dashes\n"
        "and each parameter is `name : type` on one line with its description indented\n"
        "on the next. Omit sections the symbol doesn't need."
    ),
)

_REST = DocstringStyle(
    name="rest",
    prompt_fragment=(
        "Write reStructuredText (Sphinx) docstrings: a one-line summary, then `:param\n"
        "name: description` for each parameter, `:type name: type`, `:returns:`,\n"
        "`:rtype:`, and `:raises ExceptionType: when` as needed. Omit what doesn't apply."
    ),
)

_STYLES: dict[str, DocstringStyle] = {s.name: s for s in (_GOOGLE, _NUMPY, _REST)}


def resolve_style(fmt: str, *, custom_prompt: str | None = None) -> DocstringStyle:
    """Resolve a `DocstringConfig.format` to a `DocstringStyle`.

    A built-in name (`google`/`numpy`/`rest`) returns its style. `custom` requires a
    `custom_prompt` (the user's inline or file-loaded format spec) and wraps it verbatim.
    An unknown format falls back to Google.
    """
    if fmt == "custom":
        if not custom_prompt:
            raise ValueError(
                "docstrings.format is 'custom' but no style_prompt/style_prompt_file was set"
            )
        return DocstringStyle(name="custom", prompt_fragment=custom_prompt)
    return _STYLES.get(fmt, _GOOGLE)
