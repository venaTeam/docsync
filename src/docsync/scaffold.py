"""Onboarding helpers: scaffold a starter `.docsync/` and validate manifest drift.

Two operations sit in front of the rest of the pipeline:

1. :func:`init_docs_repo` — write a minimal, *valid* ``.docsync/`` skeleton
   (``config.yml`` + a commented ``manifest.yml`` template + ``state/cursors.json``)
   so adopting docsync no longer means hand-authoring both files from scratch.
2. :func:`doctor` — re-resolve an existing manifest against real source checkouts
   and report drift (dead globs, vanished symbols, missing pages, unmapped repos)
   that would silently break anchor mapping in :mod:`docsync.impact`.

Both reuse :mod:`docsync.config` constants for paths and :func:`docsync.impact._repo_key`
for repo-key normalization, so the diagnostics match how mapping actually matches.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .config import (
    CONFIG_FILE,
    CURSORS_FILE,
    MANIFEST_FILE,
    docsync_dir,
    load_config,
)
from .impact import _repo_key
from .models import DocsyncConfig

# ---------------------------------------------------------------------------
# 1. Scaffold — `docsync init`
# ---------------------------------------------------------------------------


def _config_template() -> str:
    """A minimal valid ``config.yml``, seeded from the real model defaults."""
    models = DocsyncConfig().models  # ModelConfig defaults — never hardcode ids
    return (
        "# docsync config — see DocsyncConfig in models.py for all options.\n"
        "models:\n"
        f"  edit_model: {models.edit_model}\n"
        f"  judge_model: {models.judge_model}\n"
        f"  edit_effort: {models.edit_effort}\n"
        "# Root of the docs tree, relative to this repo (where page paths resolve).\n"
        'docs_root: "."\n'
        "# GitHub handles requested as reviewers on opened doc-update PRs.\n"
        "reviewers: []\n"
        "# Labels applied to opened docs PRs (auto-created in this repo if missing).\n"
        "pr_labels: [docsync]\n"
        "# Ship-safety: skip the edit stage for pages below this impact confidence\n"
        "# (0-1). 0 = off. Raise (e.g. 0.7) for a conservative first rollout.\n"
        "min_edit_confidence: 0.0\n"
        "# Max concurrent LLM requests across the judge + edit stages.\n"
        "max_parallel_requests: 4\n"
        "# Cap on pages edited per run (0 = unlimited); highest-confidence first.\n"
        "max_pages_per_run: 0\n"
    )


def _minimal_config_template(docs_root: str, adapter: str = "") -> str:
    """A near-empty config that only pins non-default keys, with a pointer comment.

    Everything in :class:`DocsyncConfig` is optional with a sane default, so a minimal
    config need say nothing more than where the docs live (and only when that isn't the
    repo root) and which framework owns them (only when it isn't the default mintlify).
    Keeps the onboarding artifact small instead of an 8-field template.
    """
    lines = [
        "# docsync config — only non-default keys are shown.",
        "# See DocsyncConfig in models.py for every option and its default.",
    ]
    if docs_root and docs_root != ".":
        lines.append(f'docs_root: "{docs_root}"')
    if adapter and adapter != "mintlify":
        lines.append(f"adapter: {adapter}")
    return "\n".join(lines) + "\n"


# --- detection (zero-config onboarding) ---------------------------------------


def detect_docs_root(docs_repo: Path) -> str:
    """Best-effort guess of `docs_root` (relative to *docs_repo*).

    Prefers the directory holding a Mintlify ``docs.json``/``mint.json`` (the adapter
    manifest); falls back to the common ancestor of the doc-page tree (``.mdx`` or
    ``.md``); defaults to ``"."`` when nothing is found. The ``.docsync`` dir is ignored.
    """
    from .adapters.mintlify import _DOCS_JSON, _MINT_JSON

    docs_repo = Path(docs_repo)
    for name in (_DOCS_JSON, _MINT_JSON):
        hits = [p for p in docs_repo.rglob(name) if ".docsync" not in p.parts]
        if hits:
            shallowest = min(hits, key=lambda p: len(p.relative_to(docs_repo).parts))
            rel = shallowest.parent.relative_to(docs_repo).as_posix()
            return rel or "."

    mdx = [
        p
        for ext in ("*.mdx", "*.md")
        for p in docs_repo.rglob(ext)
        if ".docsync" not in p.parts
    ]
    if not mdx:
        return "."
    common = mdx[0].parent.relative_to(docs_repo).parts
    for page in mdx[1:]:
        parts = page.parent.relative_to(docs_repo).parts
        n = 0
        while n < len(common) and n < len(parts) and common[n] == parts[n]:
            n += 1
        common = common[:n]
    return "/".join(common) or "."


def detect_adapter(docs_repo: Path, docs_root: str) -> str:
    """Return the docs adapter name recognizable from the tree, else ``""``.

    A Mintlify `docs.json`/`mint.json` (or an `.mdx` tree) ⇒ ``"mintlify"``; a Docusaurus
    config or an `.md`-only tree ⇒ ``"markdown"`` (the plain-Markdown adapter). The result
    seeds `DocsyncConfig.adapter` in a `--minimal` init and is echoed to the adopter.
    """
    from .adapters.mintlify import _DOCS_JSON, _MINT_JSON

    root = Path(docs_repo) / docs_root
    if (root / _DOCS_JSON).exists() or (root / _MINT_JSON).exists():
        return "mintlify"
    if (root / "docusaurus.config.js").exists() or (root / "docusaurus.config.ts").exists():
        return "markdown"

    def _tree(ext: str) -> bool:
        return any(".docsync" not in p.parts for p in root.rglob(ext))

    if _tree("*.mdx"):
        return "mintlify"
    if _tree("*.md"):
        return "markdown"
    return ""


def _manifest_template() -> str:
    """A commented starter manifest with one illustrative page entry.

    The example is intentionally a template: it points at a placeholder repo and
    page so :func:`doctor` flags it until the adopter edits it to real values.
    """
    return (
        "# docsync manifest — maps each doc page to the source code it documents.\n"
        "# Edit the example below: set `path` to a real page under docs_root, and\n"
        "# each source's `repo`/`globs`/`symbols` to the code that page describes.\n"
        "# Anchors here drive impact mapping; keep them honest (run `docsync doctor`).\n"
        "pages:\n"
        "  # --- EXAMPLE (replace me) ------------------------------------------\n"
        "  - path: example-page.mdx          # page path, relative to docs_root\n"
        "    sources:\n"
        "      - repo: owner/your-service    # matches the source repo (owner/name)\n"
        "        globs:                      # fnmatch globs over changed file paths\n"
        '          - "src/routes/*.py"\n'
        "        symbols:                    # symbol names (trailing * = prefix)\n"
        "          - get_app\n"
        "    max_diff_lines: 60              # per-page net-changed-lines guardrail\n"
    )


def init_docs_repo(
    docs_repo: Path,
    *,
    force: bool = False,
    minimal: bool = False,
    detect: bool = False,
    docs_root: str | None = None,
) -> list[Path]:
    """Scaffold a starter ``.docsync/`` skeleton in *docs_repo*.

    Default (``minimal=False``): the full template — ``config.yml`` (seeded from real
    model defaults), a commented placeholder ``manifest.yml``, and ``state/cursors.json``.

    ``minimal=True`` (zero-config onboarding): a minimal ``config.yml`` (only non-default
    keys) with ``docs_root`` auto-detected when ``detect`` is set (or taken from an
    explicit ``docs_root``), plus ``state/cursors.json`` — and **no** placeholder manifest
    (a doctor-flagged placeholder would defeat the point; use ``docsync infer`` or hand-
    author the manifest instead). Existing files are left untouched unless *force*.

    Returns the paths created (or overwritten), in deterministic order.
    """
    base = docsync_dir(docs_repo)
    base.mkdir(parents=True, exist_ok=True)

    config_path = base / CONFIG_FILE
    manifest_path = base / MANIFEST_FILE
    cursors_path = base / CURSORS_FILE

    if minimal:
        root = docs_root if docs_root is not None else (
            detect_docs_root(docs_repo) if detect else "."
        )
        # Seed the adapter too when detecting, so a non-mintlify site is configured
        # without the adopter having to know the field exists.
        adapter = detect_adapter(docs_repo, root) if detect else ""
        artifacts: list[tuple[Path, str]] = [
            (config_path, _minimal_config_template(root, adapter)),
            (cursors_path, "{}\n"),
        ]
    else:
        artifacts = [
            (config_path, _config_template()),
            (manifest_path, _manifest_template()),
            (cursors_path, "{}\n"),
        ]

    created: list[Path] = []
    for path, content in artifacts:
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)

    return created


# ---------------------------------------------------------------------------
# 2. Doctor — `docsync doctor`
# ---------------------------------------------------------------------------


class DeadGlob(BaseModel):
    """A manifest glob that no longer matches any file in its source checkout."""

    page: str
    repo: str
    glob: str


class MissingSymbol(BaseModel):
    """A declared symbol not found textually in any glob-matched source file."""

    page: str
    repo: str
    symbol: str


class DoctorReport(BaseModel):
    """Structured result of validating a manifest against real checkouts.

    ``ok`` is True iff there are no *hard* issues (missing pages or dead globs).
    Missing symbols are warnings — symbol-only drift degrades anchor recall but
    does not by itself break the mapping — and do not flip ``ok``.
    """

    ok: bool = True
    missing_pages: list[str] = Field(default_factory=list)
    dead_globs: list[DeadGlob] = Field(default_factory=list)
    missing_symbols: list[MissingSymbol] = Field(default_factory=list)
    unmapped_repos: list[str] = Field(default_factory=list)


def _resolve_checkout(source_repo: str, checkouts: dict[str, Path]) -> Path | None:
    """Find the checkout for *source_repo*, matching on normalized repo key.

    Uses the same :func:`docsync.impact._repo_key` normalization mapping relies on,
    so ``owner/name`` and a bare ``name`` checkout key reconcile identically.
    """
    target = _repo_key(source_repo)
    for key, path in checkouts.items():
        if _repo_key(key) == target:
            return Path(path)
    return None


def doctor(docs_repo: Path, checkouts: dict[str, Path]) -> DoctorReport:
    """Validate the loaded manifest still resolves against real source checkouts.

    For each manifest page source: every glob must match at least one file in the
    matching checkout (else it is a dead glob), and every declared symbol should
    appear at least textually in some glob-matched file (else a warning). Each page
    ``path`` must exist under ``docs_repo/<docs_root>``. Sources whose repo has no
    provided checkout are reported as unmapped (and skipped for glob/symbol checks).

    Args:
        docs_repo: Root of the docs repo holding ``.docsync/manifest.yml``.
        checkouts: Map of source-repo identifier (``owner/name`` or bare name) to a
            local checkout path. Keys are normalized via the impact module's key.

    Returns:
        A :class:`DoctorReport`. ``ok`` is False iff there are hard issues
        (missing pages or dead globs); missing symbols are warnings only.

    Raises:
        FileNotFoundError: If the docs repo has no ``.docsync/manifest.yml``.
    """
    from .config import load_manifest  # local import: keep module import-light

    manifest = load_manifest(docs_repo)
    config = load_config(docs_repo)
    docs_root = (Path(docs_repo) / config.docs_root).resolve()

    report = DoctorReport()
    seen_unmapped: set[str] = set()

    for page in manifest.pages:
        # --- page path must exist under docs_root --------------------------
        if not (docs_root / page.path).exists():
            report.missing_pages.append(page.path)

        for source in page.sources:
            checkout = _resolve_checkout(source.repo, checkouts)
            if checkout is None:
                if source.repo not in seen_unmapped:
                    seen_unmapped.add(source.repo)
                    report.unmapped_repos.append(source.repo)
                continue

            # --- globs must each match >= 1 file ---------------------------
            matched_files: list[Path] = []
            for glob in source.globs:
                hits = [p for p in checkout.glob(glob) if p.is_file()]
                if hits:
                    matched_files.extend(hits)
                else:
                    report.dead_globs.append(
                        DeadGlob(page=page.path, repo=source.repo, glob=glob)
                    )

            # --- symbols should appear textually in some matched file ------
            if source.symbols:
                corpus = _read_corpus(matched_files)
                for symbol in source.symbols:
                    needle = symbol[:-1] if symbol.endswith("*") else symbol
                    if needle and needle not in corpus:
                        report.missing_symbols.append(
                            MissingSymbol(
                                page=page.path, repo=source.repo, symbol=symbol
                            )
                        )

    report.ok = not report.missing_pages and not report.dead_globs
    return report


def _read_corpus(files: list[Path]) -> str:
    """Concatenate the text of *files*, skipping unreadable/binary ones."""
    parts: list[str] = []
    for path in files:
        try:
            parts.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(parts)
