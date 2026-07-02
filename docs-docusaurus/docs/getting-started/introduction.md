---
title: What is docsync?
description: docsync is a CI tool that reads a merged code change and makes surgical, reviewable LLM edits to your existing docs pages — even when the docs live in a separate repo.
---

docsync keeps documentation in sync with code by turning a merged code change into a reviewable docs pull request. It ingests a diff, maps it to the doc pages that change affects, makes **surgical** LLM edits to the existing `.mdx`, validates them, and opens a PR — typically against a *different* repo from the one that changed. Reach for this page when you're new to docsync and want to understand what it does and run it once end to end.

## What docsync lets you do

docsync's core job is the diff-driven sync loop: a code change lands, and docsync proposes the matching doc edits instead of letting docs drift. Its niche is the **cross-repo** shape — code in one repo, docs in another — which most doc tools don't handle.

- **Sync docs from a code change** (`docsync run`): the live loop — diff → impact → edits → validate → PR.
- **Preview impact only** (`docsync map`): which pages a change touches, no LLM edits — a cheap dry inspection.
- **Build the recall-net** (`docsync index`): refresh the optional embeddings index used to catch drift on pages the manifest doesn't anchor.

Use docsync when you maintain docs alongside an actively changing codebase and want updates triggered by commits rather than by hand. The rest of this page walks the first `run`.

## Prerequisites

| Requirement | Why |
|-------------|-----|
| A docs repo checkout | Passed as `--docs-repo`; docsync reads and edits pages here. |
| `.docsync/` config + manifest | docsync exits with a *"Run `docsync init` first"* hint if the manifest is missing. Run `docsync init` to scaffold it. |
| The source change to sync | A local checkout (path with a `.git`) or a GitHub `owner/name`, plus a `--base` and `--head` ref. |
| An LLM backend | `--backend api` needs `ANTHROPIC_API_KEY`; `--backend claude-code` reuses your local Claude Code CLI auth (no API key). |

## Make your first sync

These steps run docsync once from a local checkout. By default `run` only computes and reports — it does not write files or open a PR until you opt in.

1. **Scaffold config** (if you haven't already) so the manifest exists:
   ```bash
   docsync init
   ```

2. **Do a dry run** to see impact and proposed edits without changing anything:
   ```bash
   docsync run \
     --docs-repo ./docs \
     --src-repo owner/name \
     --base <base-sha> \
     --head <head-sha>
   ```
   `--src-repo` accepts a local path (a directory containing `.git`) or a GitHub `owner/name`. In a mono-repo where docs and code share one checkout, you can omit `--src-repo` and docsync defaults the source to `--docs-repo`.

3. **Review the report.** Add `--report-path report.md` to write the PR-body markdown to a file you can read.

4. **Open the docs PR** once the dry run looks right:
   ```bash
   docsync run --docs-repo ./docs --src-repo owner/name \
     --base <base-sha> --head <head-sha> --open-pr
   ```
   `--open-pr` branches, commits, pushes, and opens a PR (GitHub) or MR (GitLab).

:::warning
`run` is **dry-run by default** (`--dry-run` is `True`). It reports proposed edits but will not write files or open a PR until you pass `--open-pr`. If nothing changed in your docs repo after a run, this is almost always why.
:::

### Useful flags for the first run

| Flag | Default | Effect |
|------|---------|--------|
| `--open-pr` | off | Branch, commit, push, and open the docs PR/MR. |
| `--dry-run` | on | Compute and report only; no writes. |
| `--thoroughness` | from config | `light` / `medium` / `high` — controls edit depth and the diff-size budget. |
| `--max-pages` | from config | Cap pages sent to the edit stage (highest-confidence first; the rest are reported). |
| `--min-confidence` | from config | Skip the edit stage for pages below this impact confidence (0–1) — good for a conservative first rollout. |
| `--self-critique / --no-self-critique` | on | Re-check each edit against the diff and drop edits the change doesn't justify. |
| `--polish / --no-polish` | from config | Fact-frozen readability pass after each edit (larger diff, extra model call). |
| `--use-embeddings` | on | Recall-net for drift on un-anchored pages; degrades to anchors-only if the embeddings extra isn't installed. |
| `--check-links` | off | Run the adapter's broken-link soft gate. |
| `--preflight` | on | Validate the manifest (doctor) and abort *before any LLM spend* if it references doc pages that don't exist. |
| `--backend` | `api` | `api` (uses `ANTHROPIC_API_KEY`) or `claude-code` (local CLI auth). |

### Running in CI

In CI you don't need `--src-repo` / `--base` / `--head` — docsync auto-detects them. GitLab is detected via its `CI_*` env vars (signalled by `GITLAB_CI`); GitHub is read from the event JSON at `$GITHUB_EVENT_PATH`. You can also point at an event file explicitly with `--from-event <path>`.

## Verify it worked

- **Dry run:** the command exits cleanly and prints (or writes, via `--report-path`) a report listing impacted pages and proposed edits.
- **Manifest sanity:** with `--preflight` on (the default), a run aborts early with a clear message if the manifest points at doc pages that don't exist — a green preflight means anchors resolve.
- **`--open-pr`:** a new branch and a docs PR (GitHub) or MR (GitLab) appears against your docs repo, containing the surgical edits.

## Next steps

- **`docsync init`** — scaffold `.docsync/` config and manifest before your first run.
- **`docsync map`** — preview which pages a change impacts, with no LLM edits.
- **`docsync index`** — build or refresh the embeddings recall-net.
- **`docsync doctor`** — validate the manifest against your checkouts after editing anchors.