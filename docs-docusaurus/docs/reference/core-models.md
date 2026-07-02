---
title: Core Data Models
description: The Pydantic v2 data contracts every docsync pipeline stage exchanges — CodeDiff, RepoDigest, DocPlan, and the page-planning types.
---

`docsync/src/docsync/models.py` defines the shared Pydantic v2 models that every pipeline stage passes between itself and the next: diff extraction emits a `CodeDiff`, bootstrap ingest emits a `RepoDigest`, and the planner emits a `DocPlan` of `PlannedPage`s. Reach for this page when you need the exact field names, types, defaults, and helper methods of those structures — for example to construct a fixture, read a stage's output, or understand what an anchor or a changed symbol carries.

This page documents the core models only. Authoring/config models (`AuthoredPage`, `ManifestSource`, …) and the section-ordering helpers live in the same module but are out of scope here.

:::note
docsync uses **Pydantic v2** (≥2.6), deliberately unlike Keep's v1 services. The Anthropic SDK's `messages.parse()` structured-output helper validates LLM responses against these models directly, so their shapes double as the LLM output contracts.
:::

## Orientation: which model belongs to which stage

| Model | Pipeline stage | Produced by |
|-------|----------------|-------------|
| `CodeDiff` / `ChangedFile` | Diff extraction (`run`) | `diff.py` |
| `RepoDigest` / `SourceUnit` | Whole-repo ingest (`bootstrap`) | `ingest.py` |
| `DocPlan` / `PlannedPage` / `PlannedSource` | Doc planning (`bootstrap`) | `bootstrap.py` (Haiku) |

## Enums and type aliases

### `FileStatus`

`str`-valued `Enum` describing how a file was touched in a diff.

| Value | String | Meaning |
|-------|--------|---------|
| `ADDED` | `"added"` | New file |
| `MODIFIED` | `"modified"` | Existing file changed |
| `REMOVED` | `"removed"` | File deleted |
| `RENAMED` | `"renamed"` | File moved; `ChangedFile.previous_path` is set |

### `PageKind`

`Literal["concept", "guide", "reference"]` — the documentation kind that steers the author prompt and how the page is kept live.

| Value | Anchoring & liveness |
|-------|----------------------|
| `reference` | Code-anchored API/data-model pages; precise anchors, autopass into the editor |
| `concept` | Narrative subsystem/architecture explanations; broad anchors, routed through the judge |
| `guide` | Task-oriented (getting-started/how-to); loosely anchored, routed through the judge |

## `ChangedFile`

One file touched by a diff, with its hunks and the code symbols affected.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `path` | `str` | yes | Path of the file in the diff |
| `status` | `FileStatus` | yes | How the file was changed |
| `previous_path` | `Optional[str]` | no (default `None`) | Prior path; set when `status == RENAMED` |
| `hunks` | `list[str]` | no (default `[]`) | Unified-diff hunk texts (`@@ … @@` blocks plus context), one per hunk |
| `changed_symbols` | `list[str]` | no (default `[]`) | Function / class / module-level assignment names whose body the hunks touch — the cross-boundary signal impact mapping uses; survives line-number churn |

## `CodeDiff`

The structured result of comparing `base..head` in a single service repo. This is the input to the `run` pipeline's impact stage.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | `owner/name` (e.g. `"keephq/keep-api-gateway"`) or a local path |
| `base_sha` | `str` | yes | Base commit SHA |
| `head_sha` | `str` | yes | Head commit SHA |
| `pr_number` | `Optional[int]` | no (default `None`) | Originating PR number, if known |
| `pr_title` | `Optional[str]` | no (default `None`) | Originating PR title, if known |
| `files` | `list[ChangedFile]` | no (default `[]`) | The files touched by the diff |

### Methods

| Method | Returns | Behavior |
|--------|---------|----------|
| `changed_paths()` | `list[str]` | Every file's `path`, plus its `previous_path` when set (renames contribute both paths). Order follows `files`; not deduplicated. |
| `all_symbols()` | `list[str]` | All `changed_symbols` across every file, deduplicated first-seen-wins (insertion order preserved). |

## `SourceUnit`

One documentable source file, distilled to what the planner needs. Lightweight by design — paths and symbol names only, no file bodies — so a whole repo's worth fits in the planner's context; excerpts are fetched per page at author time.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `path` | `str` | yes | Repo-relative path, e.g. `"src/routes/alerts.py"` |
| `kind` | `str` | yes | Coarse language/role tag: `"python"`, `"typescript"`, or `"other"` |
| `symbols` | `list[str]` | no (default `[]`) | Top-level defs / classes / exports |

## `RepoDigest`

The lightweight, whole-repo snapshot that `bootstrap` plans from.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | `owner/name` or local path (mirrors `CodeDiff.repo`) |
| `root` | `str` | yes | Absolute path the units were walked from |
| `units` | `list[SourceUnit]` | no (default `[]`) | The documentable source files |

### Methods

| Method | Returns | Behavior |
|--------|---------|----------|
| `all_symbols()` | `list[str]` | All `symbols` across every unit, deduplicated first-seen-wins. |

## `PlannedSource`

A code anchor for a planned page — repo-qualified, so a single site can span multiple source repos.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `repo` | `str` | yes | Which source repo (matches a `RepoDigest.repo` / `CodeDiff.repo`) |
| `globs` | `list[str]` | no (default `[]`) | `fnmatch` globs over changed paths |
| `symbols` | `list[str]` | no (default `[]`) | Symbol names; a trailing `*` means prefix match |

## `PlannedPage`

One doc page the planner proposes to author. This is the per-page unit of structured LLM output the planner returns.

| Field | Type | Required | Default | Meaning |
|-------|------|----------|---------|---------|
| `page_path` | `str` | yes | — | New `.mdx` path relative to `docs_root`, e.g. `"reference/alerts.mdx"` |
| `title` | `str` | yes | — | Page title |
| `kind` | `PageKind` | no | `"reference"` | Documentation kind (see [`PageKind`](#pagekind)) |
| `section` | `str` | no | `"Reference"` | Nav group / section heading |
| `order` | `int` | no | `0` | Position within the section (ascending) |
| `summary` | `str` | no | `""` | What the page should cover; steers the author stage |
| `sources` | `list[PlannedSource]` | no | `[]` | Multi-repo code anchors for the page |

### Properties

| Property | Returns | Behavior |
|----------|---------|----------|
| `judge_required` | `bool` | `True` when `kind` is `"concept"` or `"guide"`. Narrative pages anchor to a whole subsystem, so they route through the Haiku judge instead of anchor-autopassing into a costly Opus edit — an edit fires only when a change actually invalidates the page. |