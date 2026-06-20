"""Load docsync config + manifest from a docs repo's `.docsync/` directory.

Layout (all lives in the docs repo, e.g. keep-developer-docs):

    .docsync/
      config.yml          # DocsyncConfig (models, thresholds, reviewers)
      manifest.yml        # Manifest (page <-> source mapping) — the heart of mapping
      state/cursors.json  # last processed head_sha per source repo (idempotency)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML

from .models import DocsyncConfig, Manifest, ManifestPage

_yaml = YAML(typ="safe")

# A SEPARATE round-trip instance for *editing* manifest.yml in place: it preserves
# the file's curated comments and key order on dump (the safe `_yaml` above strips
# them). Never merge through `_yaml` or the hand-authored manifest comments are lost.
_rt_yaml = YAML()  # round-trip mode is the default
_rt_yaml.preserve_quotes = True
_rt_yaml.indent(mapping=2, sequence=4, offset=2)

DOCSYNC_DIR = ".docsync"
CONFIG_FILE = "config.yml"
MANIFEST_FILE = "manifest.yml"
CURSORS_FILE = "state/cursors.json"


def docsync_dir(docs_repo: Path) -> Path:
    return Path(docs_repo) / DOCSYNC_DIR


def load_config(docs_repo: Path) -> DocsyncConfig:
    """Load .docsync/config.yml; return defaults if it's absent.

    Raises :class:`ConfigError` (with the offending field) on an unknown key or a
    bad value, rather than leaking a raw Pydantic traceback — a typo'd field is a
    mistake worth surfacing, not silently ignoring.
    """
    path = docsync_dir(docs_repo) / CONFIG_FILE
    if not path.exists():
        return DocsyncConfig()
    data = _yaml.load(path.read_text()) or {}
    try:
        return DocsyncConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_config_error(path, exc)) from exc


class ConfigError(ValueError):
    """A `.docsync/config.yml` that doesn't validate (unknown key or bad value)."""


def _format_config_error(path: Path, exc: ValidationError) -> str:
    """A friendly, `.docsync/config.yml`-framed message for a config validation error."""
    lines = [f"invalid config in {path}:"]
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) or "(root)"
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown field (check the spelling; run `docsync explain` for valid fields)"
        lines.append(f"  - {field}: {msg}")
    return "\n".join(lines)


def load_manifest(docs_repo: Path) -> Manifest:
    """Load .docsync/manifest.yml. Raises FileNotFoundError if missing."""
    path = docsync_dir(docs_repo) / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. A docsync-enabled docs repo needs "
            f"{DOCSYNC_DIR}/{MANIFEST_FILE} mapping pages to their source code."
        )
    data = _yaml.load(path.read_text()) or {}
    return Manifest.model_validate(data)


# --- manifest merge (bootstrap: append anchors, comment-preserving) -----------


def _manifest_page_dict(page: ManifestPage) -> dict:
    """A plain dict for one new manifest page, omitting unmodified default knobs.

    `exclude_defaults` drops empty `globs`/`symbols` and any guardrail left at its
    default (`max_diff_lines`, `allow_frontmatter_edit`, …), keeping the appended YAML
    minimal. Fields are emitted in declaration order (`path`, `sources`, then knobs).
    """
    return page.model_dump(mode="json", exclude_defaults=True)


_FRESH_MANIFEST_HEADER = (
    "# docsync manifest — maps each doc page to the source code it documents.\n"
    "# Bootstrapped by `docsync bootstrap`; anchors drive impact mapping for the\n"
    "# update pipeline. Run `docsync doctor` to keep them honest.\n"
)


def merge_manifest_pages(docs_repo: Path, pages: list[ManifestPage]) -> list[str]:
    """Append `pages` to `.docsync/manifest.yml`, preserving existing comments.

    Idempotent on `path`: a page already in the manifest is skipped. Creates the
    manifest (with a header) if absent. Returns the page paths actually added.
    """
    path = docsync_dir(docs_repo) / MANIFEST_FILE
    fresh = not path.exists()
    data = {} if fresh else (_rt_yaml.load(path.read_text()) or {})
    if not data.get("pages"):
        data["pages"] = []

    existing = {p.get("path") for p in data["pages"] if isinstance(p, dict)}
    added: list[str] = []
    for page in pages:
        if page.path in existing:
            continue
        data["pages"].append(_manifest_page_dict(page))
        existing.add(page.path)
        added.append(page.path)

    if not added and not fresh:
        return []

    buf = io.StringIO()
    _rt_yaml.dump(data, buf)
    content = buf.getvalue()
    if fresh:
        content = _FRESH_MANIFEST_HEADER + content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return added


# --- repo topology (mono / single / poly) -------------------------------------


def resolve_repo_mode(
    config: DocsyncConfig,
    docs_repo: Path,
    diff_repo: str,
    manifest: Manifest,
) -> str:
    """Resolve the effective repo topology: ``"mono"`` | ``"single"`` | ``"poly"``.

    An explicit ``config.repo_mode`` short-circuits detection. Under ``"auto"``: mono
    when the diff's repo *is* the docs repo (their normalized keys match — so the source
    and docs share one checkout); else poly when the manifest anchors span more than one
    distinct (non-empty) source repo; else single.

    ``diff_repo`` is the resolved ``CodeDiff.repo`` (``owner/name`` or a local path), so
    detection works the same in CI (``--from-event``) as for a local ``--src-repo`` run —
    both reduce to "does the changed repo equal the docs repo?".
    """
    if config.repo_mode != "auto":
        return config.repo_mode

    from .impact import _repo_key  # local import: keep config import-light

    docs_key = _repo_key(Path(docs_repo).resolve().name)
    if _repo_key(diff_repo) == docs_key:
        return "mono"

    repos = {
        _repo_key(s.repo) for p in manifest.pages for s in p.sources if s.repo
    }
    return "poly" if len(repos) > 1 else "single"


# --- cursor (the only mutable persisted state; committed by the Action) -------


def load_cursors(docs_repo: Path) -> dict[str, str]:
    """Map of repo -> last processed head_sha."""
    path = docsync_dir(docs_repo) / CURSORS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_cursors(docs_repo: Path, cursors: dict[str, str]) -> None:
    path = docsync_dir(docs_repo) / CURSORS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cursors, indent=2, sort_keys=True) + "\n")


def already_processed(docs_repo: Path, repo: str, head_sha: str) -> bool:
    """Idempotency check: has this head_sha for this repo already produced a PR?"""
    return load_cursors(docs_repo).get(repo) == head_sha


def advance_cursor(docs_repo: Path, repo: str, head_sha: str) -> None:
    cursors = load_cursors(docs_repo)
    cursors[repo] = head_sha
    save_cursors(docs_repo, cursors)
