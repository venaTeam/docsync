"""`docsync docstring-style` — help a user define their own docstring format.

The docstring stage ships built-in formats (google/numpy/rest) and a `custom` escape
hatch that loads a user-written spec verbatim. Writing that spec by hand is the friction
this utility removes. Two ways to produce it:

    infer   → sample a repo's *existing* docstrings and let the judge model distil the
              house style into a reusable spec + a canonical example (the brownfield path,
              mirroring `docsync infer` for manifest anchors).
    blank   → scaffold an editable template with no LLM call (write it yourself).

Either way the result is written to a `style_prompt_file` (default `.docsync/
docstring_style.md`); point `docstrings.format: custom` + `style_prompt_file:` at it and
the generator writes to your format — no code change, because `pysymbols.resolve_style`
injects the file's text verbatim as the format's prompt fragment.
"""

from __future__ import annotations

from pathlib import Path

from . import ingest as ingest_mod
from . import llm
from . import pysymbols
from .cost import MeteredClient, UsageMeter
from .models import DocsyncConfig, DocstringStyleSpec, RunUsage
from .pysymbols import DocSample

_INFER_MAX_TOKENS = 2_000
_SAMPLE_DOC_MAX_CHARS = 800
# Where the rendered style file lands by default (relative to the docs repo).
DEFAULT_STYLE_FILE = ".docsync/docstring_style.md"


# ---------------------------------------------------------------------------
# Sampling the repo's existing docstrings
# ---------------------------------------------------------------------------


def _richness(sample: DocSample) -> tuple[int, int]:
    """Sort key preferring section-rich, multi-line docstrings (they show the format)."""
    lower = sample.docstring.lower()
    sections = sum(
        marker in lower
        for marker in ("args:", "returns:", "yields:", "raises:", "attributes:",
                       "parameters", "----", ":param", ":returns", ":rtype")
    )
    return (sections, sample.docstring.count("\n"))


def sample_docstrings(
    repos: list[tuple[str, str | Path]],
    config: DocsyncConfig,
    *,
    max_samples: int = 40,
) -> list[DocSample]:
    """Collect a diverse, bounded set of existing docstrings across `repos`.

    Walks each repo read-only, extracts existing docstrings from public symbols
    (`pysymbols.collect_docstrings`), then keeps the most format-revealing ones — the
    section-rich, multi-line docstrings first — capped at `max_samples`. Each sampled
    docstring is truncated so a big one can't dominate the prompt.
    """
    ds = config.docstrings
    collected: list[DocSample] = []
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
            try:
                text = (root / unit.path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            collected.extend(
                pysymbols.collect_docstrings(
                    text, include_private=ds.include_private, targets=ds.targets
                )
            )

    collected.sort(key=_richness, reverse=True)
    picked = collected[:max_samples]
    for s in picked:
        if len(s.docstring) > _SAMPLE_DOC_MAX_CHARS:
            s.docstring = s.docstring[:_SAMPLE_DOC_MAX_CHARS].rstrip() + "\n… (truncated)"
    return picked


# ---------------------------------------------------------------------------
# Inferring a style spec from the sample (LLM)
# ---------------------------------------------------------------------------


def build_infer_prompt(samples: list[DocSample]) -> tuple[str, str]:
    """(system, user) asking the judge to distil the samples into a reusable format."""
    system = (
        "You are given real docstrings from one codebase. Distil the codebase's docstring "
        "CONVENTIONS into a single reusable format spec that a generator can follow to "
        "write NEW docstrings in the same house style.\n"
        "- `name`: a short kebab-case label for the style.\n"
        "- `guidance`: imperative rules capturing the structure and tone the samples share "
        "— the summary line convention, which sections appear and how they're written "
        "(headers, indentation, `name: desc` vs `name : type`), voice, and length. Describe "
        "the CONSISTENT conventions; ignore one-off quirks. Write it as instructions to the "
        "generator, not as a description of the samples.\n"
        "- `example`: one fresh, canonical docstring BODY in that style (text only, no "
        "surrounding triple quotes, no code), for a plausible function with parameters, a "
        "return value, and a raised exception.\n"
        "Do not invent sections the samples never use."
    )
    lines = ["Existing docstrings sampled from the codebase:", ""]
    for s in samples:
        lines.append(f"### {s.qualname}  ({s.kind})")
        lines.append(f"signature: {s.signature}")
        lines.append('"""')
        lines.append(s.docstring)
        lines.append('"""')
        lines.append("")
    return system, "\n".join(lines)


def infer_style(
    repos: list[tuple[str, str | Path]],
    config: DocsyncConfig,
    *,
    client=None,
    meter: UsageMeter | None = None,
    max_samples: int = 40,
) -> tuple[DocstringStyleSpec, list[DocSample], RunUsage]:
    """Infer a `DocstringStyleSpec` from the existing docstrings in `repos`.

    Samples the repo, asks the judge model to distil the house style, and returns the
    spec, the samples it learned from (for reporting), and metered usage. Raises
    `ValueError` when the repo has no existing docstrings to learn from — there is nothing
    to infer, so the user should scaffold a blank template instead.
    """
    meter = meter or UsageMeter()
    client = MeteredClient(llm.get_client(client), meter)

    samples = sample_docstrings(repos, config, max_samples=max_samples)
    if not samples:
        raise ValueError(
            "no existing docstrings found to learn from — run with --blank to scaffold a "
            "template you fill in, or point --src-repo at documented code."
        )

    system, user = build_infer_prompt(samples)
    spec: DocstringStyleSpec = llm.parse(
        client,
        stage="docstring-style",
        model=config.models.judge_model,
        max_tokens=_INFER_MAX_TOKENS,
        system=system,
        user=user,
        output_format=DocstringStyleSpec,
    )
    return spec, samples, meter.finalize()


# ---------------------------------------------------------------------------
# Rendering the style file (the `style_prompt_file` custom format loads verbatim)
# ---------------------------------------------------------------------------

_GENERATED_HEADER = (
    "<!-- docsync docstring style — loaded verbatim as the `custom` format's prompt.\n"
    "     Edit freely; the whole file becomes the instruction the generator follows. -->"
)


def render_style_markdown(spec: DocstringStyleSpec) -> str:
    """Render an inferred spec to the style-file text used by `format: custom`.

    The entire file is injected verbatim as the format's prompt fragment, so it reads as
    direct instructions to the generator (rules first, then a worked example).
    """
    return "\n".join(
        [
            _GENERATED_HEADER,
            "",
            f"# Docstring style: {spec.name}",
            "",
            "Write docstrings following these conventions:",
            "",
            spec.guidance.strip(),
            "",
            "Example (docstring body only — no surrounding quotes):",
            "",
            spec.example.strip(),
            "",
        ]
    )


def scaffold_template() -> str:
    """A blank, editable style template (no LLM) for a user to write a format by hand."""
    return "\n".join(
        [
            _GENERATED_HEADER,
            "",
            "# Docstring style: my-house-style",
            "",
            "Write docstrings following these conventions:",
            "",
            "- Line 1 is a one-sentence imperative summary that fits on one line.",
            "- <describe when to add a longer description>",
            "- <list the sections you use and how each is formatted, e.g. Args as "
            "`name: description`>",
            "- <voice/tense, length limits, any house rules>",
            "",
            "Example (docstring body only — no surrounding quotes):",
            "",
            "Fetch a user by id.",
            "",
            "Args:",
            "    user_id: Primary key of the user to load.",
            "",
            "Returns:",
            "    The matching User, or None when absent.",
            "",
        ]
    )


def write_style_file(
    docs_repo: Path,
    text: str,
    *,
    out: str = DEFAULT_STYLE_FILE,
    dry_run: bool = True,
) -> Path:
    """Write the style file under `docs_repo` (creating parent dirs). Returns its path.

    On a dry run nothing is written; the resolved path is still returned so the caller can
    report where it would land.
    """
    path = docs_repo / out
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path


def config_snippet(out: str = DEFAULT_STYLE_FILE) -> str:
    """The `config.yml` lines that activate the written style file."""
    return (
        "docstrings:\n"
        "  format: custom\n"
        f"  style_prompt_file: {out}\n"
    )
