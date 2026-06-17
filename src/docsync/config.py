"""Load docsync config + manifest from a docs repo's `.docsync/` directory.

Layout (all lives in the docs repo, e.g. keep-developer-docs):

    .docsync/
      config.yml          # DocsyncConfig (models, thresholds, reviewers)
      manifest.yml        # Manifest (page <-> source mapping) — the heart of mapping
      state/cursors.json  # last processed head_sha per source repo (idempotency)
"""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

from .models import DocsyncConfig, Manifest

_yaml = YAML(typ="safe")

DOCSYNC_DIR = ".docsync"
CONFIG_FILE = "config.yml"
MANIFEST_FILE = "manifest.yml"
CURSORS_FILE = "state/cursors.json"


def docsync_dir(docs_repo: Path) -> Path:
    return Path(docs_repo) / DOCSYNC_DIR


def load_config(docs_repo: Path) -> DocsyncConfig:
    """Load .docsync/config.yml; return defaults if it's absent."""
    path = docsync_dir(docs_repo) / CONFIG_FILE
    if not path.exists():
        return DocsyncConfig()
    data = _yaml.load(path.read_text()) or {}
    return DocsyncConfig.model_validate(data)


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
