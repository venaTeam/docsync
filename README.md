# docsync

**Keep documentation in sync with code changes.** docsync ingests a merged PR's
diff from a service repo, maps it to the documentation pages it affects — which
may live in a **different** repo — uses an LLM to make *surgical* edits to the
existing `.mdx`, validates them, and opens a reviewable PR against the docs repo.
It runs as part of CI.

MVP target: the **Keep** platform — 4 service repos → `keep-developer-docs`
(Mintlify). The cross-repo shape (code and docs in separate repos) is the niche
docsync is built for; most existing tools are single-repo and API-reference only.

## How it works

```
[service repo: PR merged]                 [docs repo: keep-developer-docs]
 notify-docsync job  ───────────────────►  Action: docsync run
   repository_dispatch (code-merged)
                                            1  event capture + idempotency cursor
                                            2  diff extract   (git/gh + ast symbols)
                                            3  impact map      (anchors → judge)
                                            4  edit gen        (Opus 4.8, str-replace)
                                            5  validate        (frontmatter/MDX/size)
                                            6  open docs PR     (per-page rationale)
```

| Stage | Module | What it does |
|---|---|---|
| 2 | `diff.py` | `git diff` / `gh api compare` → `CodeDiff`; hunks-only changed-symbol extraction. |
| 3 | `impact.py` | Anchor match (manifest globs/symbols) → Haiku judge confirms; optional embeddings recall-net. |
| 4 | `edits.py` | Opus 4.8 → `find`/`replace` edit ops (structured output); strict, unique-match application. |
| 5 | `validate.py` + `adapters/mintlify.py` | Frontmatter freeze, MDX/mermaid integrity, diff-size guardrail, soft link check. |
| 6 | `pr.py` | Branch, commit, push, open PR (or emit a `.patch` in dry-run). |

The mapping is **anchor-first**: a manifest pins each doc page to source globs +
symbols, so the common high-drift pages map deterministically and for free. A
cheap Haiku judge confirms candidates before any expensive Opus edit, so cost
scales with *impacted* pages, not page count. Edits are **str-replace ops, never
full rewrites** — that, plus the validation gates, keeps diffs small and reviewable.

## Install

```bash
poetry install                 # core
poetry install -E embeddings   # + sentence-transformers recall-net (pulls torch)
```

Requires Python ≥ 3.10, `git`, and `gh` (for the cross-repo path). Set
`ANTHROPIC_API_KEY` for the edit/judge models.

## Use it (CLI — the Phase-0 dogfood)

Start in the docs repo — `init` scaffolds `.docsync/`, auto-detecting `docs_root`, the
adapter, and the repo topology (mono/single/poly):

```bash
# 1. Scaffold + auto-detect (run inside the docs repo):
poetry run docsync init --minimal          # or: --infer --src-repo name=path
poetry run docsync explain                  # every config field, its default + meaning
poetry run docsync doctor                   # check the manifest resolves to real code
```

Then the live loop:

```bash
# Inspect which pages a merge would touch (cheap, no LLM):
poetry run docsync map \
  --src-repo ../keep-api-gateway --base HEAD~1 --head HEAD \
  --docs-repo ../keep-developer-docs

# Full pipeline, dry run (report + patch, no writes):
poetry run docsync run \
  --src-repo ../keep-api-gateway --base <base_sha> --head <head_sha> \
  --docs-repo ../keep-developer-docs \
  --pr-number 1234 --pr-title "add /alerts/bulk route" \
  --report-path /tmp/docsync-report.md

# Apply + open a docs PR:
poetry run docsync run ... --no-dry-run --open-pr
```

`--src-repo` accepts a local checkout (uses `git diff`) or a GitHub `owner/name`
(uses `gh api compare`).

## Configure a docs repo

`docsync init` scaffolds this; `docsync explain` documents every field. docsync reads
`<docs-repo>/.docsync/`:

```
.docsync/
  config.yml          # models, thresholds, repo_mode, thoroughness, reviewers
  manifest.yml        # page ↔ source mapping — the heart of impact mapping
  state/cursors.json  # last processed head_sha per repo (idempotency; committed)
```

Config is all-optional with sane defaults (`repo_mode: auto`, `thoroughness: medium`);
run `docsync explain` for the full schema, or `docsync explain manifest` for the manifest.
A ready-made manifest for Keep is in [`examples/keep/`](examples/keep/) — copy it
into `keep-developer-docs/.docsync/`. Manifest shape (`repo:` is optional in a
single-/mono-repo setup):

```yaml
pages:
  - path: services/api-gateway.mdx
    sources:
      - repo: keephq/keep-api-gateway
        globs: ["src/routes/router_setup.py", "src/config/config.py"]
        symbols: ["setup_routers", "KEEP_*"]   # trailing * = prefix match
    max_diff_lines: 60        # diff-size guardrail
    allow_frontmatter_edit: false
```

## Wire it into CI

1. Add [`examples/workflows/docsync.yml`](examples/workflows/docsync.yml) to the
   **docs repo** (`keep-developer-docs/.github/workflows/`).
2. Add [`examples/workflows/notify-docsync.yml`](examples/workflows/notify-docsync.yml)
   to **each service repo** — it fires a `repository_dispatch` on merge.
3. Secrets: `ANTHROPIC_API_KEY` + a `DOCSYNC_TOKEN` (read service repos / write
   docs repo) on the docs repo; `DOCSYNC_DISPATCH_TOKEN` on each service repo.

## Scope (MVP)

**In:** Mintlify; edits to existing pages; the 5 validation gates; one docs PR per
source PR with per-page rationale; file-based state + optional embeddings recall-net.
**Deferred:** GitHub App / webhook hosting; auto nav editing + new pages (flagged,
not applied); Docusaurus/other frameworks (the `adapters/` interface is the seam);
multi-org, billing, dashboards.

## Develop

```bash
poetry run pytest -q      # unit + integration (fake Anthropic client; no network)
poetry run ruff check src tests
```
