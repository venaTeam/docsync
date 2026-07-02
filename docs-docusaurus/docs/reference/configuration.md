---
title: Configuration Reference
description: The `.docsync/config.yml` loader API — how docsync finds, validates, and applies configuration, plus the manifest, cursor, and repo-topology helpers in `config.py`.
---

The `docsync.config` module loads and validates everything docsync reads from a docs repo's `.docsync/` directory: the `config.yml` settings file, the `manifest.yml` page↔source map, and the `state/cursors.json` idempotency record. Reach for this page when you need the exact behavior of `load_config`, the validation error it raises, or the helper functions that read and write `.docsync/` state.

All paths are resolved relative to the docs repo root you pass in. Configuration is **optional** — an absent `config.yml` yields a fully-defaulted `DocsyncConfig`.

## Directory layout

Everything docsync reads lives under `.docsync/` in the **docs** repo (not the source repo):

```
.docsync/
  config.yml          # DocsyncConfig — models, thresholds, repo_mode
  manifest.yml        # Manifest — page <-> source mapping
  state/cursors.json  # last processed head_sha per source repo (idempotency)
```

These locations are fixed by module constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| `DOCSYNC_DIR` | `.docsync` | Root config directory inside the docs repo. |
| `CONFIG_FILE` | `config.yml` | Settings file, loaded into `DocsyncConfig`. |
| `MANIFEST_FILE` | `manifest.yml` | Page-to-source map, loaded into `Manifest`. |
| `CURSORS_FILE` | `state/cursors.json` | Per-repo last-processed `head_sha`. |

## `load_config`

Loads `.docsync/config.yml` into a `DocsyncConfig`, returning **defaults when the file is absent**. This is the entry point every command uses to read configuration.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Path to the docs repo root; `config.yml` is read from `<docs_repo>/.docsync/config.yml`. |

**Returns** `DocsyncConfig`. If the file does not exist, returns `DocsyncConfig()` (all defaults). Otherwise the YAML is parsed (an empty file is treated as `{}`) and validated with `DocsyncConfig.model_validate`.

**Raises** `ConfigError` on validation failure — an unknown field or a bad value — instead of leaking a raw Pydantic traceback.

:::warning
Validation is strict: an unknown key (for example a misspelled field) is rejected, not silently ignored. The error message names the offending field and, for unknown fields, suggests running `docsync explain` to see the valid set.
:::

## `ConfigError`

```python
class ConfigError(ValueError): ...
```

Raised when `.docsync/config.yml` fails validation (an unknown key or a bad value). Subclasses `ValueError`. The message is built by `_format_config_error`, which frames each Pydantic error as `<field>: <msg>`, where `<field>` is the dotted location path (or `(root)`) and an `extra_forbidden` error becomes the friendlier `unknown field (check the spelling; run \`docsync explain\` for valid fields)`.

## `DocsyncConfig`

The validated model `load_config` returns. Every field is optional — `DocsyncConfig()` constructs a usable default configuration. The full field set is defined in `models.py`; the field this module reads directly is `repo_mode`.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `repo_mode` | `str` | `"auto"` | Repo topology selector consumed by `resolve_repo_mode`. When set to an explicit value it short-circuits auto-detection; `"auto"` triggers detection. |

`repo_mode` resolves to one of the values below (see [`resolve_repo_mode`](#resolve_repo_mode)):

| Value | Meaning |
|-------|---------|
| `auto` | Detect the topology from the diff repo and manifest. |
| `mono` | Source and docs share one checkout. |
| `single` | Docs map to exactly one distinct source repo. |
| `poly` | Docs map to more than one distinct source repo. |

## `docsync_dir`

Returns the `.docsync` directory path for a docs repo.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |

**Returns** `Path` — `Path(docs_repo) / ".docsync"`. Does not check existence or create the directory.

## `load_manifest`

Loads `.docsync/manifest.yml` into a `Manifest`. Unlike `load_config`, a missing manifest is an **error**, because a docsync-enabled docs repo requires page→source mappings.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |

**Returns** `Manifest` (via `Manifest.model_validate`; an empty file is treated as `{}`).

**Raises** `FileNotFoundError` if `.docsync/manifest.yml` is absent.

## `merge_manifest_pages`

Appends new pages to `.docsync/manifest.yml` while preserving the file's existing comments and key order. Used by `bootstrap` to register newly-authored pages.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |
| `pages` | `list[ManifestPage]` | yes | Pages to append. |

**Returns** `list[str]` — the page `path`s actually added.

**Behavior:**
- **Idempotent on `path`** — a page whose `path` already exists in the manifest is skipped.
- **Creates the manifest if absent**, prepending a generated header comment block.
- Each appended page is dumped with `model_dump(mode="json", exclude_defaults=True)`, so empty `globs`/`symbols` and any guardrail left at its default are omitted, keeping the YAML minimal.
- If nothing was added and the file already existed, returns `[]` without rewriting the file.

:::note
Merging uses a dedicated round-trip YAML instance (`_rt_yaml`) so hand-authored comments and key order survive the edit. The plain safe loader used elsewhere strips comments on dump.
:::

## `resolve_repo_mode` {#resolve_repo_mode}

Resolves the effective repo topology — `"mono"`, `"single"`, or `"poly"` — used to decide how the diff repo relates to the docs and manifest.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `config` | `DocsyncConfig` | yes | Provides `repo_mode`; a non-`"auto"` value is returned as-is. |
| `docs_repo` | `Path` | yes | Docs repo root; its resolved directory name is the docs key. |
| `diff_repo` | `str` | yes | The resolved `CodeDiff.repo` — `owner/name` or a local path. |
| `manifest` | `Manifest` | yes | Source for the set of distinct source repos across all page anchors. |

**Returns** `str` — one of `"mono"`, `"single"`, `"poly"`.

**Resolution order under `repo_mode == "auto"`:**
1. **`mono`** when the diff repo's normalized key equals the docs repo's key (source and docs share one checkout).
2. **`poly`** when the manifest's page `sources` span more than one distinct non-empty `repo`.
3. **`single`** otherwise.

An explicit `config.repo_mode` (anything other than `"auto"`) short-circuits all detection. Normalization uses `_repo_key` from `impact.py` (imported lazily to keep `config` import-light), so a CI `--from-event` run and a local `--src-repo` run reduce to the same comparison.

## Cursor state

The cursor file (`state/cursors.json`) is the only mutable persisted state docsync writes — a JSON map of source repo → last processed `head_sha`, committed by the GitHub Action to make reruns idempotent.

### `load_cursors`

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |

**Returns** `dict[str, str]` mapping `repo → head_sha`. Returns `{}` if the file is absent.

### `save_cursors`

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |
| `cursors` | `dict[str, str]` | yes | Full repo→`head_sha` map to persist. |

**Returns** `None`. Creates the parent directory if needed and writes pretty-printed JSON (`indent=2`, `sort_keys=True`) with a trailing newline.

### `already_processed`

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |
| `repo` | `str` | yes | Source repo key. |
| `head_sha` | `str` | yes | Candidate head commit. |

**Returns** `bool` — `True` if the stored cursor for `repo` already equals `head_sha` (this commit already produced a PR).

### `advance_cursor`

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | yes | Docs repo root. |
| `repo` | `str` | yes | Source repo key. |
| `head_sha` | `str` | yes | New head commit to record. |

**Returns** `None`. Loads the current cursors, sets `cursors[repo] = head_sha`, and writes them back via `save_cursors`.