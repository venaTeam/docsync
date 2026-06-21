---
name: docsync
description: >-
  Operate the docsync CLI — the tool that keeps documentation in sync with code across
  separate repos (code in one repo, docs in another). Use this skill whenever the user
  wants to set up, onboard, run, or troubleshoot docsync: bootstrapping a docs site from
  code, syncing docs for a merged PR/diff, inferring or validating manifest anchors, tuning
  .docsync config, or deploying docsync in an air-gapped / on-prem environment with a private
  PyPI mirror and an internal Anthropic gateway. Trigger it even when the user only says
  things like "sync the docs", "update docs from this code change", "set up doc automation
  for these repos", "validate the docs manifest", or "author docs for this service" — if the
  context is docsync or cross-repo code→docs syncing, use this skill rather than editing docs
  by hand.
metadata:
  author: Yarin Shitrit
  version: "0.1.0"
---

# docsync

docsync keeps documentation in sync with code. It ingests a merged PR's diff from a **service
repo**, maps it to the doc pages it affects (which usually live in a **different** repo), uses
an LLM to make **surgical** find/replace edits to the existing `.mdx`, validates them, and opens
a **reviewable PR** against the docs repo. The cross-repo shape (code and docs in separate repos)
is the point — it also handles **poly-repo** (many code repos → one docs repo).

This skill is the operating manual: which command to reach for, the safe order to run them in,
how the config/manifest work, and how to run it offline. **Source of truth lives in the repo**:
`CLAUDE.md` and `docs/` in the docsync checkout, and the live `docsync explain` command. When a
detail here looks stale, prefer `docsync explain` and the repo over memory.

## Golden rules (read these first)

1. **Dry-run first, always.** `run`, `bootstrap`, and `infer` default to `--dry-run` for a
   reason: LLM stages cost money and edits touch real docs. Inspect the report before writing.
   Move to `--no-dry-run` only after the dry-run looks right.
2. **Edits are surgical, never rewrites.** docsync emits find/replace ops with a strict
   single-occurrence check and hard validation gates (frontmatter freeze, structural-signature
   integrity, diff-size). Don't fight the gates — if an edit is rejected, the page or anchor is
   usually the problem, not the gate.
3. **Cheap before expensive.** Use `map` (LLM-free impact preview) and `doctor` (manifest
   validation, no LLM) before spending tokens on `run`/`bootstrap`.
4. **Never auto-merge the self-docs PR.** docsync documents itself via a CI loop that opens
   `docs: sync …` PRs — those are always human-reviewed. Same norm for any docs PR docsync opens.
5. **Before any code PR to docsync itself:** `poetry run ruff check src/ tests/` and
   `poetry run pytest -q` must be green. Tests fake the LLM client (no network).

## Choosing the command

| Situation | Command | Notes |
|-----------|---------|-------|
| Brand-new docs site, author from a code snapshot (greenfield) | `bootstrap` | Plans the IA, authors every page, writes nav + seeds manifest anchors. |
| Docs already exist but have no manifest anchors (brownfield) | `infer` | Proposes anchors via embeddings + Haiku judge; `--write` to merge them. |
| A PR/diff merged; update affected docs (the live loop) | `run` | diff → impact → edit → validate → PR. The everyday command. |
| Just preview which pages a diff would touch | `map` | LLM-free, cheap. Great smoke test for wiring. |
| Check the manifest is honest against real checkouts | `doctor` | No LLM. Confirms globs/symbols resolve. Run after editing the manifest. |
| Scaffold `.docsync/` in a docs repo | `init` | `--minimal --detect` for a clean start. |
| See every config field / manifest schema | `explain` | `docsync explain` or `docsync explain <field>` / `docsync explain manifest`. |
| Build the optional embeddings recall-net | `index` | Only if embeddings are enabled. |
| Score docsync against a golden set | `eval` | Precision/recall harness for tuning. |

## The onboarding recipe (poly-repo, greenfield)

This is the "start here" sequence. Assume code repos checked out locally and a (possibly empty)
docs repo. Multi-repo input is `--src-repo name=path`, **repeatable**.

```bash
# 1. Scaffold .docsync/ (config.yml, manifest.yml, state/)
docsync init --minimal --detect --docs-repo /path/docs

# 2. Learn the surface (no cost)
docsync explain
docsync explain manifest

# 3. Author the site — DRY RUN FIRST. Review the planned IA + cost report.
docsync bootstrap --docs-repo /path/docs \
  --src-repo api=/path/api --src-repo worker=/path/worker --src-repo ui=/path/ui \
  --dry-run
#   (add --plan-only to see just the information architecture without authoring)

# 4. Apply once the plan looks right (no --open-pr yet if git hosting isn't wired)
docsync bootstrap --docs-repo /path/docs \
  --src-repo api=/path/api --src-repo worker=/path/worker --src-repo ui=/path/ui \
  --no-dry-run

# 5. Validate the seeded anchors against the real code
docsync doctor --docs-repo /path/docs \
  --checkout api=/path/api --checkout worker=/path/worker --checkout ui=/path/ui

# 6. Prove the live loop on a real change in one repo
docsync map --docs-repo /path/docs --src-repo api=/path/api --base <sha1> --head <sha2>
docsync run --docs-repo /path/docs --src-repo api=/path/api --base <sha1> --head <sha2> --dry-run
#   drop --dry-run to write edits; add --open-pr to open the PR/MR (GitHub via gh, GitLab via
#   glab — set config.forge; see "Running in CI"). Needs the host CLI on the runner.
```

For **brownfield** (docs already exist), swap step 3–4 for:
`docsync infer --docs-repo /path/docs --src-repo … --dry-run` then `… --write`, then `doctor`.

**Before `run` will map anything**, the docs repo's `manifest.yml` must have pages anchored to the
target source repo — otherwise `map`/`run` find nothing. If a repo has no anchors yet, that's an
onboarding gap: run `bootstrap` (greenfield) or `infer` (brownfield) for it first.

## Running in CI (the live loop, event-driven)

### GitHub Actions — bare auto-detect works

```bash
# $GITHUB_EVENT_PATH is auto-detected; --from-event is the explicit equivalent.
docsync run --docs-repo /path/docs --from-event "$GITHUB_EVENT_PATH"
```

The diff is fetched from the GitHub API (`gh api compare`), which is fine on GitHub.

### GitLab — pass an explicit LOCAL checkout (important for self-managed / air-gapped)

⚠️ **Gotcha:** bare auto-detect on GitLab (`docsync run --docs-repo …` with no diff flags) parses
the `CI_*` vars correctly **but then fetches the diff via `gh api` against github.com** — its
default diff runner is `diff_github`. On self-managed or air-gapped GitLab that has no github.com
access, that **fails**. The robust pattern is to give `run` an explicit **local** `--src-repo`
(a path with a `.git`), which routes through `diff_local` (a local `git diff`, fully offline):

```bash
# GitLab MR pipeline: $CI_PROJECT_DIR is the cloned source repo. Use the MR base/head SHAs.
docsync run --docs-repo /path/docs \
  --src-repo "$CI_PROJECT_NAME=$CI_PROJECT_DIR" \
  --base "$CI_MERGE_REQUEST_DIFF_BASE_SHA" \
  --head "$CI_MERGE_REQUEST_SOURCE_BRANCH_SHA"
```

(`--src-repo` resolves to `diff_local` only when the path exists and has a `.git`; an `owner/name`
or a non-git path falls back to `diff_github`. So always point it at the local checkout on GitLab.)

- `run` **preflights by default** (`--preflight`): it runs `doctor` and aborts *before any LLM
  spend* if the manifest references doc pages that don't exist. `--no-preflight` bypasses it.
- `--report-path out.md` saves the PR/MR-body markdown for inspection/artifacts.
- `--from-event` is the **GitHub** event-JSON path — it does nothing for GitLab.

### Opening the docs change — `--open-pr` opens a PR (GitHub) or MR (GitLab)

`--open-pr` opens a **GitHub PR** (via the `gh` CLI) or a **GitLab MR** (via the `glab` CLI),
chosen by the **`forge`** config field: `auto` (default — detects the host from the docs repo's
`origin` remote), `github`, or `gitlab`. Self-managed GitLab on an opaque hostname should set
`forge: gitlab` explicitly. The picked CLI (`gh` or `glab`) must be installed and authenticated on
the runner; on air-gapped GitLab that means `glab` and a token that can push + open an MR in the
docs repo.

If you'd rather not let docsync open the MR (e.g. `glab` isn't available), run without `--open-pr`:
`run` writes the page edits + a patch (`--report-path`), and the pipeline opens the MR itself via
GitLab **push options** (`git push -o merge_request.create -o merge_request.target=main …`) or the
GitLab MR API with `$CI_JOB_TOKEN`.

Either way the resulting `docs: sync …` PR/MR is **always human-reviewed — never auto-merged.**

## Backends and models

- `--backend api` (default) → `anthropic.Anthropic()`, reads `ANTHROPIC_API_KEY` from the env.
  The Anthropic SDK also honors **`ANTHROPIC_BASE_URL`**, so pointing at an internal gateway is a
  pure env-var change — no code edit. This is the air-gapped path.
- `--backend claude-code` → shells to the local `claude` CLI (reuses Claude Code auth, no key).
  Dev/dogfood only: per-call overhead and subscription-auth ToS limits make it unfit for batch CI.
- Models come from `.docsync/config.yml` → `models`: edit/author = `claude-opus-4-8`,
  judge/critique/infer = `claude-haiku-4-5`. If a gateway exposes different IDs, override
  `models.edit_model` / `models.judge_model` to match.

## `.docsync/` config and manifest — the essentials

- `config.yml` → `DocsyncConfig`: models, thresholds, `docs_root`, `adapter` (mintlify),
  `repo_mode` (auto/mono/single/poly), `max_pages_per_run`, embeddings settings. **All optional**,
  sane defaults. Inspect with `docsync explain`.
- `manifest.yml` → each page maps to one or more **sources** = `repo` + `globs` + `symbols`.
  - `reference` pages use **precise** anchors and **autopass** into the editor.
  - `concept`/`guide` pages use **broad** anchors and set `judge_required: true` so a Haiku judge
    confirms relevance instead of autopassing.
  - In poly-repo, every source needs an explicit `repo`; empty `repo` is a wildcard (mono/single).
- `state/cursors.json` → idempotency cursor (last processed head per repo).

After editing the manifest, **run `docsync doctor`** to keep anchors honest. Full anchor schema,
page-kind guidance, and config field reference: **see `references/manifest-and-config.md`**.

## Air-gapped / on-prem deployment

Running docsync with no public internet — private PyPI mirror + internal Anthropic gateway — has
its own runbook: mirroring the dependency closure (including the heavy `embeddings` extra:
`sentence-transformers` → `torch`/`transformers`/`huggingface-hub`), building docsync as a wheel,
staging the embedding model to a local path, and forcing offline mode. **See
`references/air-gapped-setup.md`** before attempting an offline install. The short version:

- Set `ANTHROPIC_BASE_URL` + `ANTHROPIC_API_KEY` to the internal gateway; keep `--backend api`.
- Point `embedding_model` in `config.yml` at a **local model directory**, and set
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so no HuggingFace Hub call can fire.
- Verify zero egress: `run --dry-run` should reach only the gateway — never `api.anthropic.com`
  or `huggingface.co`.

## When you're stuck

- An edit got rejected by validation → read the report; check the page's structural signature and
  the anchor's `max_diff_lines`. Tighten the anchor or split the page; don't loosen the gate.
- `doctor` reports a glob/symbol miss → the manifest drifted from the code. Fix the anchor.
- No pages impacted when you expected some → run `map` to see the matching; the anchor probably
  doesn't cover the changed paths/symbols. Broaden the glob or add a symbol.
- Offline run tries to hit the network → an env var is missing; recheck the air-gapped runbook.
