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
    )


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


def init_docs_repo(docs_repo: Path, *, force: bool = False) -> list[Path]:
    """Scaffold a starter ``.docsync/`` skeleton in *docs_repo*.

    Creates ``config.yml`` (seeded from real model defaults), a commented
    ``manifest.yml`` template, and ``state/cursors.json`` (``{}``). Existing files
    are left untouched unless *force* is set; only files actually written are
    returned.

    Args:
        docs_repo: Root of the docs repository to scaffold.
        force: Overwrite any of the three artifacts that already exist.

    Returns:
        The paths created (or overwritten), in deterministic order. Files skipped
        because they already exist and *force* is False are omitted.
    """
    base = docsync_dir(docs_repo)
    base.mkdir(parents=True, exist_ok=True)

    config_path = base / CONFIG_FILE
    manifest_path = base / MANIFEST_FILE
    cursors_path = base / CURSORS_FILE

    artifacts: list[tuple[Path, str]] = [
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
