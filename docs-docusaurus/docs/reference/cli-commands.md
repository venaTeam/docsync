---
title: CLI Reference
description: Complete reference for the docsync command-line interface — the run command, its flags, defaults, backends, and the internal helpers that load config and resolve a diff.
---

The docsync CLI (`docsync`) is the engine that drives every sync: it computes a diff, maps it to affected docs, generates surgical edits, validates them, and emits a PR or a report. This page documents the `run` command and its flags exhaustively, plus the internal helpers that load configuration and resolve the diff source.

The CLI is a [Typer](https://typer.tiangolo.com/) app exposed as `app`, declared in `cli.py`:

```python
app = typer.Typer(add_completion=False, help="Keep documentation in sync with code changes.")
```

The module docstring advertises three commands; `run` is the full pipeline and is fully documented below.

| Command | Purpose |
|---------|---------|
| `docsync run` | Full pipeline: diff → impact → edits → validate → (PR \| patch + report). |
| `docsync map` | Impact mapping only (no LLM edits) — cheap dry inspection. |
| `docsync index` | Build/refresh the embeddings index (optional recall-net). |

## `docsync run`

Runs the complete pipeline. By default it is a **dry run**: it computes and reports the result without writing files or opening a PR. Pass `--open-pr` to branch, commit, push, and open a docs PR/MR.

```bash
docsync run --docs-repo ./docs --src-repo owner/name --base <sha> --head <sha>
```

### Parameters

| Flag | Type | Required | Default | Meaning |
|------|------|----------|---------|---------|
| `--docs-repo` | `Path` | Yes | — | Path to the docs repo checkout. |
| `--src-repo` | `str` | No | `None` | Service repo: local path or GitHub `owner/name`. Omit when using `--from-event`. |
| `--base` | `str` | No | `None` | Base ref/sha (before). Omit with `--from-event`. |
| `--head` | `str` | No | `None` | Head ref/sha (after). Omit with `--from-event`. |
| `--from-event` | `Path` | No | `None` | GitHub event JSON (the CI `$GITHUB_EVENT_PATH`): derives repo/base/head/PR automatically. Auto-detected from `$GITHUB_EVENT_PATH` when no flags are given. |
| `--pr-number` | `int` | No | `None` | PR number to associate with the diff. |
| `--pr-title` | `str` | No | `None` | PR title to associate with the diff. |
| `--dry-run` / `--no-dry-run` | `bool` | No | `True` | Compute + report only; do not write or open a PR. |
| `--open-pr` / `--no-open-pr` | `bool` | No | `False` | Branch, commit, push, and open a docs PR (GitHub) or MR (GitLab); see `config.forge`. |
| `--use-embeddings` / `--no-use-embeddings` | `bool` | No | `True` | Embeddings recall-net: also surface drift on pages the manifest doesn't anchor. Degrades to anchors-only if the embeddings extra isn't installed. |
| `--check-links` / `--no-check-links` | `bool` | No | `False` | Run the active adapter's broken-link soft gate (no-op for adapters without a link checker, e.g. plain markdown). |
| `--self-critique` / `--no-self-critique` | `bool` | No | `None` (uses `config.self_critique`) | Adversarially re-check each generated edit against the diff (adds a judge-model call per page) and drop edits not justified by the change. On by default; `--no-self-critique` disables it. Overrides `config.self_critique`. |
| `--polish` / `--no-polish` | `bool` | No | `None` (uses `config.readability_pass`) | Readability pass: after each edit, run a fact-frozen pass that revises the page for a leading summary + scannable structure (adds an edit-model call and a larger diff). Overrides `config.readability_pass`. |
| `--min-confidence` | `float` | No | `None` (uses `config.min_edit_confidence`) | Skip the edit stage for pages below this impact confidence (0–1). Use for a conservative first rollout. |
| `--max-pages` | `int` | No | `None` (uses `config.max_pages_per_run`) | Cap pages sent to the edit stage (highest-confidence first; the rest are reported, not edited). |
| `--max-parallel` | `int` | No | `None` (uses `config.max_parallel_requests`) | Max concurrent LLM requests for the judge + edit stages. |
| `--preflight` / `--no-preflight` | `bool` | No | `True` | Pre-flight the manifest (doctor) and abort before any LLM spend if it references doc pages that don't exist. `--no-preflight` to bypass. |
| `--thoroughness` | `str` | No | `None` (uses `config.thoroughness`) | Generation thoroughness: `light` \| `medium` \| `high`. Controls edit depth and the diff-size budget. |
| `--report-path` | `Path` | No | `None` | Write the PR-body markdown here. |
| `--backend` | `str` | No | `"api"` | LLM backend: `api` (`ANTHROPIC_API_KEY`) or `claude-code` (dev: reuse the local Claude Code CLI auth, no API key). |

:::warning
Several flags whose default is shown as `None` are tri-state overrides: `None` means "fall back to the value in `config.yml`," not "off." For `--self-critique` and `--polish`, pass the explicit `--no-...` form to disable a behavior that config enables.
:::

### Diff source resolution

`run` selects its diff source by precedence (`--from-event` > explicit flags > CI auto-detect), implemented in `_resolve_diff`:

1. **`--from-event`** — parse the supplied event JSON via `events.diff_from_event`.
2. **Explicit flags** — all of `--src-repo`, `--base`, `--head` present → build the diff directly (`_build_diff`).
3. **CI auto-detect** — none of the above → `events.diff_from_ci`, which reads `CI_*` vars (GitLab, signalled by `$GITLAB_CI`) or the GitHub event JSON at `$GITHUB_EVENT_PATH`. On failure it raises `typer.BadParameter` instructing you to supply `--from-event` or all of `--src-repo` / `--base` / `--head`.

`_build_diff` chooses local vs. remote automatically: if `src_repo` is an existing path containing a `.git` directory it runs `diff_mod.diff_local`, otherwise it treats the value as a GitHub `owner/name` and runs `diff_mod.diff_github`.

### Mono-repo convenience

Docs and code can share a single checkout. When `--src-repo` is omitted but `--base` and `--head` are given (an explicit local run, not CI or `--from-event`), and `config.repo_mode` is `auto` or `mono`, `run` defaults the source repo to `--docs-repo`:

```bash
docsync run --docs-repo . --base X --head Y   # no --src-repo needed
```

### Return / output

In the default dry run, `run` produces a report (optionally written to `--report-path`) and opens nothing. With `--open-pr`, it branches, commits, pushes, and opens a docs PR on GitHub or an MR on GitLab, selected by `config.forge`.

## Internal helpers

These module-level functions back the command. They are not commands themselves but define the load-and-validate behavior every command shares.

### `_apply_thoroughness(config, value)`

Overrides `config.thoroughness` from the `--thoroughness` flag, validating the level.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `config` | config object | Yes | The loaded config to mutate in place. |
| `value` | `Optional[str]` | Yes | Requested level, or `None` to leave config untouched. |

Returns `None`. If `value` is not one of `light`, `medium`, `high` (the `_THOROUGHNESS_LEVELS` tuple), it raises `typer.BadParameter`. A `value` of `None` is a no-op.

### `_load_config(docs_repo)`

Loads config via `cfg.load_config(docs_repo)`. On a `cfg.ConfigError` (an invalid `config.yml`), it echoes a framed `docsync: <error>` message and raises `typer.Exit(2)`.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | Yes | Docs repo checkout to load `.docsync/config.yml` from. |

### `_load_manifest_or_hint(docs_repo)`

Loads the manifest via `cfg.load_manifest(docs_repo)`. On a missing manifest (`FileNotFoundError`), it echoes the error plus `Run \`docsync init\` first.` and raises `typer.Exit(2)`.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `docs_repo` | `Path` | Yes | Docs repo checkout to load `.docsync/manifest.yml` from. |

### `_build_diff(src_repo, base, head, pr_number, pr_title)`

Builds a diff from an explicit source. Routes to `diff_mod.diff_local` when `src_repo` is an existing path with a `.git` directory, else to `diff_mod.diff_github`.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `src_repo` | `str` | Yes | Local path or GitHub `owner/name`. |
| `base` | `str` | Yes | Base ref/sha. |
| `head` | `str` | Yes | Head ref/sha. |
| `pr_number` | `Optional[int]` | Yes | Associated PR number (may be `None`). |
| `pr_title` | `Optional[str]` | Yes | Associated PR title (may be `None`). |

### `_resolve_diff(src_repo, base, head, pr_number, pr_title, from_event)`

Picks the diff source by precedence and returns the resolved diff. See [Diff source resolution](#diff-source-resolution) for the precedence and the `typer.BadParameter` raised when nothing resolves.

| Parameter | Type | Required | Meaning |
|-----------|------|----------|---------|
| `src_repo` | `Optional[str]` | Yes | Explicit source repo, or `None`. |
| `base` | `Optional[str]` | Yes | Explicit base ref, or `None`. |
| `head` | `Optional[str]` | Yes | Explicit head ref, or `None`. |
| `pr_number` | `Optional[int]` | Yes | Associated PR number. |
| `pr_title` | `Optional[str]` | Yes | Associated PR title. |
| `from_event` | `Optional[Path]` | Yes | Event JSON path, or `None`. |

## Constants and exit codes

| Name | Value | Meaning |
|------|-------|---------|
| `_THOROUGHNESS_LEVELS` | `("light", "medium", "high")` | Accepted `--thoroughness` levels. |

| Exit code | Raised by | Condition |
|-----------|-----------|-----------|
| `2` | `_load_config` | `config.yml` is invalid (`cfg.ConfigError`). |
| `2` | `_load_manifest_or_hint` | Manifest is missing (`FileNotFoundError`). |
| `BadParameter` | `_apply_thoroughness`, `_resolve_diff` | Invalid `--thoroughness` level, or no resolvable diff source. |

## Backends

The `--backend` flag selects how LLM calls are dispatched, resolved at runtime via `llm_backends.get_client`.

| Backend | Requirement | Use |
|---------|-------------|-----|
| `api` (default) | `ANTHROPIC_API_KEY` | Standard API access. |
| `claude-code` | Local Claude Code CLI auth | Dev/dogfooding; reuses the CLI session, no API key. |

:::note
For the `.docsync/` config fields these flags override (`thoroughness`, `self_critique`, `readability_pass`, `min_edit_confidence`, `max_pages_per_run`, `max_parallel_requests`, `repo_mode`, `forge`), see the configuration reference for `DocsyncConfig`.
:::