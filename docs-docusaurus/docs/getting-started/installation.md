---
title: Installation & Setup
description: Install docsync, verify the CLI, and scaffold a starter .docsync/ configuration directory so you can begin syncing docs with code.
---

This page gets docsync installed and onboarded: you install the package, confirm the CLI runs, and scaffold a `.docsync/` config directory in your docs repo. By the end you'll have a valid `config.yml`, a starter `manifest.yml`, and an idempotency cursor — the three files every other docsync command reads.

docsync keeps documentation in sync with code: it ingests a merged PR's diff, maps it to the doc pages it affects, makes surgical LLM edits, and opens a reviewable PR against the docs repo. Reach for this guide when you're adopting docsync in a docs repo for the first time, before running [`docsync run`](#next-steps).

## Prerequisites

| Requirement | Why |
|-------------|-----|
| **Python ≥ 3.10** | docsync's runtime. |
| **Poetry** | Dependency and virtualenv manager used to install docsync. |
| A **docs repo checkout** | A local path to the documentation repository you want docsync to manage. |
| **An LLM backend** | Either an `ANTHROPIC_API_KEY` (the `api` backend) or a working local `claude` CLI (the `claude-code` backend). |

:::note
The `claude-code` backend reuses your local Claude Code CLI authentication and needs no API key — handy for dogfooding. The `api` backend reads `ANTHROPIC_API_KEY` from the environment.
:::

## 1. Install docsync

From the docsync source checkout, install with Poetry. The `embeddings` extra pulls in `sentence-transformers`, which powers the optional embeddings recall-net (surfacing drift on pages the manifest doesn't anchor):

```bash
poetry install -E embeddings
```

If you don't need the recall-net, plain `poetry install` is enough — `docsync run` degrades to anchors-only when the embeddings extra isn't present.

## 2. Verify the installation

Confirm the CLI is wired up by asking the Typer `app` for its help. This lists the available commands:

```bash
poetry run docsync --help
```

You should see the command surface, including `run` (the diff-driven sync), `map`, `index`, `bootstrap`, `infer`, `doctor`, and `init`. If this prints the command list, docsync is installed correctly.

## 3. Scaffold `.docsync/`

Run `init` against your docs repo to write a minimal, valid `.docsync/` skeleton. This is `init_docs_repo` under the hood — it saves you from hand-authoring config and manifest from scratch:

```bash
poetry run docsync init --docs-repo ./docs
```

This creates three files:

| File | Contents |
|------|----------|
| `.docsync/config.yml` | A minimal `config.yml` seeded from the real model defaults (`edit_model`, `judge_model`, `edit_effort`, `docs_root`, reviewers, PR labels, confidence/parallelism/page caps). |
| `.docsync/manifest.yml` | A commented starter manifest with one **illustrative placeholder** page entry, mapping a doc page to the source code it documents. |
| `.docsync/state/cursors.json` | The idempotency cursor (last processed head per repo). |

### Zero-config detection

During scaffolding, docsync inspects the docs tree and fills in sensible defaults rather than asking you to. Three detectors run:

- **`detect_docs_root`** — guesses `docs_root` relative to the repo. It prefers the directory holding a Mintlify `docs.json`/`mint.json`, falls back to the common ancestor of the `.mdx`/`.md` page tree, and defaults to `"."` when nothing is found. The `.docsync` directory is ignored.
- **`detect_adapter`** — names the docs framework: `mintlify` (a `docs.json`/`mint.json`, or an `.mdx` tree), `docusaurus` (a `docusaurus.config.js`/`.ts` at the docs dir or its parent), or `markdown` (an `.md`-only tree). Seeds `DocsyncConfig.adapter`.
- **`detect_repo_mode`** — returns `"mono"` when source code lives alongside the docs *outside* `docs_root`, otherwise `"single"`. The `poly` (multi-repo) topology can't be inferred from one checkout, so it's never guessed here.

:::tip
A `--minimal` init writes a near-empty `config.yml` that pins only the non-default keys — `docs_root` (only when it isn't the repo root) and `adapter` (only when it isn't the default `mintlify`) — keeping the onboarding artifact small.
:::

## 4. Verify the scaffold worked

First confirm the files exist:

```bash
ls .docsync/
# config.yml  manifest.yml  state/
```

Then edit `.docsync/manifest.yml`: replace the placeholder example with a real page under `docs_root` and set each source's `repo`/`globs`/`symbols` to the code that page describes. Once you've done that, validate the manifest against your real source checkouts:

```bash
poetry run docsync doctor --docs-repo ./docs
```

`doctor` re-resolves the manifest and reports drift — dead globs, vanished symbols, missing pages, unmapped repos — using the same matching logic as impact mapping.

:::warning
A freshly scaffolded `manifest.yml` points at a **placeholder** repo and page on purpose, so `doctor` will flag it as invalid until you edit it to real values. That failure is expected on a brand-new scaffold — it's the signal to fill in your real anchors.
:::

## Next steps

- **Sync docs from a code change** — run the full pipeline (diff → impact → edits → validate → PR) with `docsync run --docs-repo ./docs --src-repo owner/name --base <sha> --head <sha>`.
- **Preview impact only** — use `docsync map` for a cheap, LLM-free look at which pages a diff would touch.
- **Build the embeddings index** — run `docsync index` to enable the recall-net for pages the manifest doesn't anchor.
- **Author a docs site from scratch** — use `docsync bootstrap` for a greenfield docs repo instead of `run`.