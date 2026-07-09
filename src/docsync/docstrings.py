"""Code-level docstring generation — the `--docstring` stage.

Where `bootstrap.py`/`pipeline.py` write *prose docs* into a separate docs repo, this
writes *docstrings* into the source itself. It runs before the docs stage so the freshly
documented source becomes richer context for authoring/editing.

    D1 ingest   → read the source files (ingest.read_excerpt is per-file; we read whole)
    D2 locate   → public undocumented symbols + placement positions (pysymbols, no LLM)
    D3 generate → docstring TEXT per symbol, batched per file (Opus, metered)
    D4 place    → deterministic splice into the raw file text (pysymbols, no LLM)
    D5 validate → prove only docstrings changed (AST-equality gate); skip a file on failure
    D6 write    → in place, into the source checkout (write_docstrings; no PR)

Two modes: backfill (`diff=None`) documents every public undocumented symbol; diff-mode
(`diff` given) scopes to the symbols the merged PR touched. Every LLM call goes through
the injectable `client` (wrapped in `MeteredClient`), so cost lands on `result.usage`.
"""

from __future__ import annotations

from pathlib import Path

from . import ingest as ingest_mod
from . import llm
from . import pysymbols
from . import style
from .cost import MeteredClient, UsageMeter
from .models import (
    CodeDiff,
    DocsyncConfig,
    DocstringOutcome,
    DocstringResult,
    FileDocstrings,
)
from .pool import run_parallel

_DOCSTRING_MAX_TOKENS = 6_000
# Cap the source handed to the model per file so a huge file can't overflow the prompt.
_SYMBOL_SOURCE_MAX_CHARS = 6_000


def _load_custom_prompt(config: DocsyncConfig, docs_repo: Path | None) -> str | None:
    """Resolve a `format='custom'` style spec: inline `style_prompt` wins over the file."""
    ds = config.docstrings
    if ds.style_prompt:
        return ds.style_prompt
    if ds.style_prompt_file and docs_repo is not None:
        path = docs_repo / ds.style_prompt_file
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def _diff_symbols_by_path(diff: CodeDiff) -> dict[str, set[str]]:
    """Map each changed source path to the symbol names the diff touched in it."""
    out: dict[str, set[str]] = {}
    for f in diff.files:
        out[f.path] = set(f.changed_symbols)
        if f.previous_path:
            out[f.previous_path] = set(f.changed_symbols)
    return out


def _target_matches(target: pysymbols.SymbolTarget, changed: set[str]) -> bool:
    """Whether a located target was touched by the diff (by symbol/qualname/owner)."""
    if not changed:
        return False
    if target.name in changed or target.qualname in changed:
        return True
    # A method whose owning class was touched (e.g. the class body changed).
    if "." in target.qualname and target.qualname.split(".", 1)[0] in changed:
        return True
    return False


def _build_prompt(
    file_label: str,
    targets: list[pysymbols.SymbolTarget],
    source_text: str,
    docstyle: pysymbols.DocstringStyle,
    thoroughness: str,
) -> tuple[str, str]:
    """(system, user) for one file's docstring generation."""
    system = "\n\n".join(
        [
            "You write Python docstrings for the given symbols. You return ONLY docstring "
            "text (no surrounding quotes, no code, no signatures). Return one item per "
            "symbol keyed by its exact qualname; omit a symbol if the source doesn't "
            "support a faithful docstring rather than guessing.",
            style.GROUNDING,
            style.thoroughness_directive(thoroughness),
            docstyle.prompt_fragment,
        ]
    )
    parts = [f"File: `{file_label}`", "", "Document these symbols:"]
    for t in targets:
        src = pysymbols.symbol_source(source_text, t)
        if len(src) > _SYMBOL_SOURCE_MAX_CHARS:
            src = src[:_SYMBOL_SOURCE_MAX_CHARS] + "\n# … (truncated)"
        parts.append("")
        parts.append(f"### qualname: {t.qualname}  ({t.kind})")
        parts.append(f"signature: {t.signature}")
        parts.append("```python")
        parts.append(src)
        parts.append("```")
    return system, "\n".join(parts)


def _process_file(
    repo: str,
    root: Path,
    rel_path: str,
    targets: list[pysymbols.SymbolTarget],
    source_text: str,
    config: DocsyncConfig,
    docstyle: pysymbols.DocstringStyle,
    client,
) -> tuple[list[DocstringOutcome], tuple[str, str] | None]:
    """Generate + splice + validate one file. Returns (outcomes, (abs_path, new_text)|None).

    Never raises: a generation or validation failure is reported as outcomes and produces
    no file text (nothing is written for that file).
    """
    abs_path = str(root / rel_path)
    spliceable = [t for t in targets if not t.inline_body]
    outcomes: list[DocstringOutcome] = []

    def outcome(t: pysymbols.SymbolTarget, status: str, note: str = "") -> DocstringOutcome:
        return DocstringOutcome(
            repo=repo, path=rel_path, qualname=t.qualname, kind=t.kind, status=status, note=note
        )

    for t in targets:
        if t.inline_body:
            outcomes.append(outcome(t, "skipped", "inline body — no line to place a docstring"))

    if not spliceable:
        return outcomes, None

    system, user = _build_prompt(
        rel_path, spliceable, source_text, docstyle, config.thoroughness
    )
    model = config.docstrings.model or config.models.edit_model
    result: FileDocstrings = llm.parse(
        client,
        stage="docstring",
        model=model,
        max_tokens=_DOCSTRING_MAX_TOKENS,
        system=system,
        user=user,
        output_format=FileDocstrings,
        thinking=True,
        effort=config.models.edit_effort,
    )

    by_qual = {item.qualname: item.docstring for item in result.items if item.docstring.strip()}
    pairs: list[tuple[pysymbols.SymbolTarget, str]] = []
    for t in spliceable:
        text = by_qual.get(t.qualname)
        if text is None:
            outcomes.append(outcome(t, "no_change", "model returned no docstring"))
        else:
            pairs.append((t, text))

    if not pairs:
        return outcomes, None

    new_text, applied = pysymbols.splice_docstrings(source_text, pairs)
    try:
        pysymbols.assert_only_docstrings_changed(source_text, new_text)
    except ValueError as exc:
        for t, _ in pairs:
            outcomes.append(outcome(t, "invalid", f"validation gate: {exc}"))
        return outcomes, None

    for t, _ in pairs:
        outcomes.append(outcome(t, "documented"))
    return outcomes, (abs_path, new_text)


def run_docstrings(
    repos: list[tuple[str, str | Path]],
    config: DocsyncConfig,
    *,
    diff: CodeDiff | None = None,
    docs_repo: Path | None = None,
    client=None,
    meter: UsageMeter | None = None,
) -> DocstringResult:
    """Generate docstrings for the public symbols in `repos` (in-memory; no writes).

    `repos` is `(name, local_path)` pairs (same shape as bootstrap's). With `diff` given,
    only diff-touched symbols in changed files are documented (the `run --docstring`
    path); without it, every public undocumented symbol is (the `bootstrap --docstring`
    path). Returns a `DocstringResult` carrying per-symbol outcomes, the new file texts to
    write, and metered usage. `write_docstrings` performs the actual in-place write.
    """
    meter = meter or UsageMeter()
    client = MeteredClient(llm.get_client(client), meter)

    ds = config.docstrings
    custom = _load_custom_prompt(config, docs_repo)
    docstyle = pysymbols.resolve_style(ds.format, custom_prompt=custom)
    diff_by_path = _diff_symbols_by_path(diff) if diff is not None else None

    # D1/D2: read each repo and locate targets, respecting the diff scope + symbol cap.
    jobs: list[tuple[str, Path, str, list[pysymbols.SymbolTarget], str]] = []
    remaining = ds.max_symbols_per_run or None
    for name, path in repos:
        digest = ingest_mod.walk_repo(
            Path(path),
            repo=name,
            exclude_dirs=ingest_mod.resolve_exclude_dirs(config.ingest_exclude_dirs),
        )
        root = Path(digest.root)
        for unit in digest.units:
            if unit.kind != "python":
                continue
            if diff_by_path is not None and unit.path not in diff_by_path:
                continue
            try:
                source_text = (root / unit.path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            targets = pysymbols.iter_targets(
                source_text,
                include_private=ds.include_private,
                targets=ds.targets,
                overwrite=ds.overwrite_existing,
            )
            if diff_by_path is not None:
                changed = diff_by_path.get(unit.path, set())
                targets = [t for t in targets if _target_matches(t, changed)]
            if not targets:
                continue
            if remaining is not None:
                if remaining <= 0:
                    break
                targets = targets[:remaining]
                remaining -= len(targets)
            jobs.append((name, root, unit.path, targets, source_text))
        if remaining is not None and remaining <= 0:
            break

    # D3-D5: generate + splice + validate per file, in parallel.
    def worker(job):
        name, root, rel_path, targets, source_text = job
        return _process_file(
            name, root, rel_path, targets, source_text, config, docstyle, client
        )

    results = run_parallel(worker, jobs, config.max_parallel_requests)

    outcomes: list[DocstringOutcome] = []
    file_texts: dict[str, str] = {}
    for file_outcomes, written in results:
        outcomes.extend(file_outcomes)
        if written is not None:
            abs_path, new_text = written
            file_texts[abs_path] = new_text

    return DocstringResult(outcomes=outcomes, file_texts=file_texts, usage=meter.finalize())


def write_docstrings(result: DocstringResult, *, dry_run: bool = True) -> list[str]:
    """Write the validated docstring changes in place. Returns the affected abs paths.

    On a dry run nothing is written; the paths that *would* change are still returned so
    the caller can report them. Files are keyed by absolute path on the result.
    """
    paths = sorted(result.file_texts)
    if dry_run:
        return paths
    for path in paths:
        Path(path).write_text(result.file_texts[path], encoding="utf-8")
    return paths
