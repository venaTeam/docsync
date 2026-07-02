---
title: "Quick Start: Your First Sync"
description: "Run your first docsync command in minutes — bootstrap a docs site from a code snapshot, or sync existing pages from a merged diff."
---

This page walks you through your first docsync run end to end: install the tool, point it at your repos, and either author a fresh docs site or sync existing pages from a code change. Reach for it when you've just installed docsync and want a concrete first result before wiring it into CI.

docsync has two generation flows, and which one you start with depends on whether docs already exist:

- **`bootstrap`** — greenfield. Author a whole sectioned docs site from a code snapshot. Use this when the docs repo is empty.
- **`run`** — diff-driven. Make surgical edits to *existing* `.mdx`/`.md` pages from a merged PR's diff. This is the live loop docsync runs in CI.

Pick `bootstrap` for a brand-new site; pick `run` once pages exist and you want to keep them in sync.

## Prerequisites

Before either flow, make sure you have:

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 + Poetry | docsync is a Poetry project with an in-project `.venv`. |
| Dependencies installed | `poetry install -E embeddings` — the `embeddings` extra adds the recall-net `run` uses by default. |
| An LLM backend | `--backend api` needs `ANTHROPIC_API_KEY`; `--backend claude-code` reuses your local Claude Code CLI auth (no API key). Defaults to `api`. |
| A docs repo checkout | The directory you pass to `--docs-repo`. |
| A `.docsync/` directory | `config.yml` + `manifest.yml`. Scaffold it with `docsync init` if it's missing (`run` prints this hint when the manifest isn't found). |

:::tip
For local dogfooding without an API key, set `--backend claude-code` to shell out to your installed `claude` CLI instead of spending against `ANTHROPIC_API_KEY`.
:::

## Flow A — Author a new site with `bootstrap`

Use this when the docs repo has no pages yet. `bootstrap` ingests your source repos, plans a sectioned information architecture, authors one page per plan entry, validates each, and writes pages + nav + manifest anchors.

1. **Install dependencies** (from the docsync checkout):

   ```bash
   poetry install -E embeddings
   ```

2. **Run bootstrap**, pointing at your docs repo and one or more source repos as `name=path`:

   ```bash
   poetry run docsync bootstrap \
     --docs-repo ./docs \
     --src-repo keep-api-gateway=../keep-api-gateway
   ```

3. **(Optional) Add a readability pass** with `--polish` to run a fact-frozen revision over each authored page:

   ```bash
   poetry run docsync bootstrap --docs-repo ./docs --src-repo name=path --polish
   ```

The result is written into the docs repo: the authored pages, an ordered nav grouped into sections (Getting Started → Concepts → Architecture → Reference → Operations), and `manifest.yml` anchors tying each page back to real code. Narrative pages anchor to broad subsystem globs and carry `judge_required`, so a later `run` won't fire an edit on every unrelated change.

## Flow B — Sync existing pages with `run`

Use this once pages exist. `run` maps a code diff to the pages it affects, generates surgical find/replace edits, and validates them. **By default it is a dry run** — it computes and reports without writing files or opening a PR.

1. **Preview the sync** for a known commit range. Pass the source repo as a local path or a GitHub `owner/name`, plus the `base` and `head` shas:

   ```bash
   poetry run docsync run \
     --docs-repo ./docs \
     --src-repo ../keep-api-gateway \
     --base <base-sha> \
     --head <head-sha> \
     --report-path sync-report.md
   ```

   The pipeline runs `diff → impact → edits → validate` and writes the PR-body markdown to `--report-path`. Nothing is committed.

2. **Inspect the report.** It lists each impacted page and whether its edit passed validation. Only pages with applied, validated edits are eligible to be written.

3. **Open the PR** once the preview looks right. Add `--open-pr` to branch, commit, push, and open a docs PR (or GitLab MR, per `config.forge`):

   ```bash
   poetry run docsync run \
     --docs-repo ./docs \
     --src-repo ../keep-api-gateway \
     --base <base-sha> \
     --head <head-sha> \
     --open-pr
   ```

Useful flags for a first, conservative rollout:

| Flag | Default | Effect |
|------|---------|--------|
| `--open-pr` | off | Write changes and open a PR/MR instead of reporting only. |
| `--min-confidence <0-1>` | config | Skip the edit stage for pages below this impact confidence. |
| `--max-pages <n>` | config | Cap pages sent to the edit stage (highest-confidence first). |
| `--no-self-critique` | on | Disable the adversarial re-check that drops edits not justified by the diff. |
| `--polish` | config | Add a fact-frozen readability pass after each edit. |
| `--backend claude-code` | `api` | Use the local Claude Code CLI instead of an API key. |

:::note
Before any LLM spend, `run` pre-flights the manifest (a `doctor` check) and aborts if it references doc pages that don't exist on disk. Keep anchors honest — run `docsync doctor` after editing the manifest, or pass `--no-preflight` to bypass the check.
:::

### Running from CI

In CI you don't pass `--src-repo`/`--base`/`--head` — docsync auto-detects them. On GitHub it reads the event JSON at `$GITHUB_EVENT_PATH`; on GitLab it reads the `CI_*` variables (signalled by `$GITLAB_CI`). You can also point it at a specific event file with `--from-event <path>`.

## Verify it worked

- **`bootstrap`**: confirm new pages appear under the section folders in your docs repo (e.g. `getting-started/`, `reference/`), that nav is populated, and that `.docsync/manifest.yml` now lists anchors for the authored pages.
- **`run` (dry run)**: open the file from `--report-path` and check the per-page outcomes — each impacted page shows its surgical edit and whether it passed validation. A page with no applicable edit is reported with a note (e.g. "model returned no edits") rather than changed.
- **`run --open-pr`**: confirm a docs PR/MR was opened against the docs repo with the expected edits.

## Next steps

- **Validate your manifest** — run `docsync doctor` against your checkouts whenever you change anchors.
- **Tune behavior** — edit `.docsync/config.yml` (models, thresholds, `docs_root`, `max_pages_per_run`, `readability_pass`).
- **Wire it into CI** — let merged PRs trigger `run` automatically and open `docs: sync …` PRs for human review.