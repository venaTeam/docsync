---
title: Manifest Schema Reference
description: The complete schema for `.docsync/manifest.yml` — the page-to-source anchors that drive impact mapping — plus the loader functions and validation behavior that read it.
---

The manifest (`.docsync/manifest.yml`) maps each documentation page to the source code it documents. Its anchors — globs and symbol names, per source repo — are what docsync matches a merged diff against to decide which pages a code change impacts. This page documents the schema models (`PlannedPage`, `PlannedSource`, `PageKind`), the loaders that read the file, and the related diff/digest models that anchors are matched against.

The manifest lives alongside two siblings in the docs repo's `.docsync/` directory; all three are read through `config.py`:

| Path | Purpose |
|------|---------|
| `.docsync/config.yml` | `DocsyncConfig` — models, thresholds, reviewers. |
| `.docsync/manifest.yml` | The page ↔ source mapping documented here. |
| `.docsync/state/cursors.json` | Last processed `head_sha` per source repo (idempotency). |

## Page anchor schema

A manifest page pairs a doc path with one or more source anchors. The planner emits these as `PlannedPage` / `PlannedSource` during `bootstrap`; the same shape is what the on-disk manifest stores.

### `PlannedPage`

One doc page, with its metadata and the code anchors that keep it live. Structured LLM output from the bootstrap planner.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `page_path` | `str` | yes | New `.mdx` path relative to `docs_root`, e.g. `reference/alerts.mdx`. |
| `title` | `str` | yes | Page title. |
| `kind` | `PageKind` | no (default `"reference"`) | Steers the author prompt and how the page is kept live. See [Page kinds](#page-kinds). |
| `section` | `str` | no (default `"Reference"`) | Nav group / section heading. |
| `order` | `int` | no (default `0`) | Position within the section, ascending. |
| `summary` | `str` | no (default `""`) | What the page should cover; steers the author stage. |
| `sources` | `list[PlannedSource]` | no (default `[]`) | Code anchors, one entry per source repo. |

**`judge_required` (property → `bool`)** — returns `True` when `kind` is `"concept"` or `"guide"`. Narrative pages anchor to a whole subsystem, so they route through the Haiku judge rather than anchor-autopassing into a (costly) Opus edit; an edit then fires only when a change actually invalidates the page.

### `PlannedSource`

One code anchor for a page, qualified by source repo so a single manifest can serve a multi-repo site.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | Which source repo this anchor belongs to. Matches a `RepoDigest.repo` / `CodeDiff.repo`. |
| `globs` | `list[str]` | no (default `[]`) | `fnmatch` globs evaluated over the diff's changed paths. |
| `symbols` | `list[str]` | no (default `[]`) | Symbol names to match; a trailing `*` means prefix match. |

A page is a candidate for editing when a changed path in the diff matches one of its `globs`, or a changed symbol matches one of its `symbols`, for the corresponding `repo`.

### Page kinds

`PageKind = Literal["concept", "guide", "reference"]`. The kind steers both the author prompt and the update routing.

| Kind | Anchoring | Update routing |
|------|-----------|----------------|
| `reference` | Precise (specific paths/symbols) | Autopasses into the edit stage on a match. |
| `concept` | Broad (a whole subsystem) | Routed through the judge (`judge_required` is `True`). |
| `guide` | Loose (task-oriented) | Routed through the judge (`judge_required` is `True`). |

:::note
The on-disk manifest page model (`ManifestPage`) carries the same `path` + `sources` shape and additionally accepts per-page guardrail knobs such as `max_diff_lines` and `allow_frontmatter_edit`. When `bootstrap` appends pages, defaulted knobs and empty `globs`/`symbols` are omitted from the written YAML to keep it minimal.
:::

## Loading the manifest and config

These functions read `.docsync/` from a docs repo checkout. `docs_repo` is the repo root; the `.docsync/` subdirectory is resolved internally.

### `load_manifest(docs_repo: Path) -> Manifest`

Loads and validates `.docsync/manifest.yml`. **Raises `FileNotFoundError` if the file is missing** — a docsync-enabled docs repo must have a manifest. Returns a validated `Manifest`.

### `load_config(docs_repo: Path) -> DocsyncConfig`

Loads `.docsync/config.yml`. **Returns `DocsyncConfig()` defaults if the file is absent.** On an unknown key or bad value it raises `ConfigError` (with the offending field) rather than leaking a raw Pydantic traceback.

### `docsync_dir(docs_repo: Path) -> Path`

Returns `docs_repo / ".docsync"`. The base path the other loaders build on.

### `ConfigError(ValueError)`

Raised by `load_config` when `.docsync/config.yml` fails validation (unknown field or bad value). The message is framed against the config file and lists each offending field; an `extra_forbidden` error is reported as an unknown field with a hint to run `docsync explain`.

### `merge_manifest_pages(docs_repo: Path, pages: list[ManifestPage]) -> list[str]`

Appends `pages` to `.docsync/manifest.yml` while **preserving existing comments and key order** (it uses a round-trip YAML loader, not the plain safe loader). Behavior:

- **Idempotent on `path`** — a page whose `path` already exists in the manifest is skipped.
- Creates the manifest (with a generated header) if it does not exist.
- Returns the list of page paths actually added (empty when nothing new was appended to an existing file).

## Repo topology resolution

### `resolve_repo_mode(config, docs_repo, diff_repo, manifest) -> str`

Resolves the effective repo topology, returning one of `"mono"`, `"single"`, or `"poly"`.

| Parameter | Type | Meaning |
|-----------|------|---------|
| `config` | `DocsyncConfig` | An explicit `config.repo_mode` other than `"auto"` short-circuits detection. |
| `docs_repo` | `Path` | The docs repo checkout. |
| `diff_repo` | `str` | The resolved `CodeDiff.repo` (`owner/name` or a local path). |
| `manifest` | `Manifest` | Source of the anchor repos counted for `"poly"` detection. |

Under `repo_mode == "auto"`:

- **`"mono"`** — the diff's repo *is* the docs repo (their normalized keys match).
- **`"poly"`** — the manifest's anchors span more than one distinct non-empty source repo.
- **`"single"`** — otherwise.

Because detection reduces to "does the changed repo equal the docs repo?", it behaves identically for a CI `--from-event` run and a local `--src-repo` run.

## Idempotency cursors

`state/cursors.json` is the only mutable persisted state, committed by the CI Action. It maps each source repo to the last `head_sha` that produced a PR.

| Function | Signature | Behavior |
|----------|-----------|----------|
| `load_cursors` | `(docs_repo: Path) -> dict[str, str]` | Returns the repo → `head_sha` map, or `{}` if the file is absent. |
| `save_cursors` | `(docs_repo: Path, cursors: dict[str, str]) -> None` | Writes the map (indented, key-sorted), creating parent dirs. |
| `already_processed` | `(docs_repo: Path, repo: str, head_sha: str) -> bool` | `True` if `repo`'s stored cursor equals `head_sha`. |
| `advance_cursor` | `(docs_repo: Path, repo: str, head_sha: str) -> None` | Sets `repo`'s cursor to `head_sha` and saves. |

## Models anchors are matched against

A manifest anchor is only meaningful against the diff (for `run`) or the repo digest (for `bootstrap`). These are the structures `globs` and `symbols` resolve against.

### `CodeDiff`

The structured result of comparing `base..head` in a single service repo.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | `owner/name` or a local path. Matched against `PlannedSource.repo`. |
| `base_sha` | `str` | yes | Base commit. |
| `head_sha` | `str` | yes | Head commit (the cursor value). |
| `pr_number` | `Optional[int]` | no (default `None`) | Source PR number. |
| `pr_title` | `Optional[str]` | no (default `None`) | Source PR title. |
| `files` | `list[ChangedFile]` | no (default `[]`) | Files touched by the diff. |

Methods:
- `changed_paths() -> list[str]` — every file's `path` plus its `previous_path` (when renamed). Globs match against this list.
- `all_symbols() -> list[str]` — deduped union of every file's `changed_symbols` (first-seen order). Symbols match against this list.

### `ChangedFile`

One file touched by a diff, with its hunks and affected symbols.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `path` | `str` | yes | File path. |
| `status` | `FileStatus` | yes | How the file changed. |
| `previous_path` | `Optional[str]` | no (default `None`) | Set when `status == RENAMED`. |
| `hunks` | `list[str]` | no (default `[]`) | Unified-diff hunk texts (`@@ … @@` blocks + context), one per hunk. |
| `changed_symbols` | `list[str]` | no (default `[]`) | Function / class / module-level assignment names the hunks touch; the cross-boundary signal used by impact mapping (survives line-number churn). |

### `FileStatus` (enum)

String enum of how a file changed.

| Value | String |
|-------|--------|
| `ADDED` | `"added"` |
| `MODIFIED` | `"modified"` |
| `REMOVED` | `"removed"` |
| `RENAMED` | `"renamed"` |

### `RepoDigest` and `SourceUnit`

The whole-repo snapshot `bootstrap` plans from (paths + symbol names only, no file bodies).

`RepoDigest`:

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | `owner/name` or local path (mirrors `CodeDiff.repo`). |
| `root` | `str` | yes | Absolute path the units were walked from. |
| `units` | `list[SourceUnit]` | no (default `[]`) | The documentable source files. |

`RepoDigest.all_symbols() -> list[str]` returns the deduped union of every unit's `symbols`.

`SourceUnit` — one documentable source file:

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `path` | `str` | yes | Repo-relative path, e.g. `src/routes/alerts.py`. |
| `kind` | `str` | yes | Coarse language/role tag: `"python"` \| `"typescript"` \| `"other"`. |
| `symbols` | `list[str]` | no (default `[]`) | Top-level defs / classes / exports. |

## Internal helpers

Two module-private helpers in `models.py` support the models above:

- `_dedupe_preserving_order(items: Iterable[str]) -> list[str]` — first-seen-wins dedupe (via `dict.fromkeys`), used by `all_symbols()`.
- `_prompt_tokens(usage) -> int` — billable prompt tokens = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` (used by cost accounting, not the manifest).

:::tip
After hand-editing `.docsync/manifest.yml`, run `docsync doctor` to validate the anchors against real checkouts — it catches globs and symbols that no longer resolve to any source.
:::

## Module constants

`config.py` resolves all paths from these constants:

| Constant | Value |
|----------|-------|
| `DOCSYNC_DIR` | `".docsync"` |
| `CONFIG_FILE` | `"config.yml"` |
| `MANIFEST_FILE` | `"manifest.yml"` |
| `CURSORS_FILE` | `"state/cursors.json"` |