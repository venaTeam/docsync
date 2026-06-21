# Manifest & config reference

Authoritative source is `docsync explain` (config) and `docsync explain manifest`, plus
`DocsyncConfig` / `ManifestPage` / `ManifestSource` in `src/docsync/models.py`. This file is a
quick map; when in doubt, run `docsync explain`.

## `.docsync/` layout

```
.docsync/
  config.yml          # DocsyncConfig — all fields optional, unknown keys REJECTED (extra="forbid")
  manifest.yml        # pages → sources (anchors). Required for `run`.
  state/
    cursors.json      # {repo: last_processed_head_sha} — idempotency
    embeddings/       # cached vectors (only if embeddings enabled)
```

A typo'd config key is an error, not silently ignored — `extra="forbid"`. Run `docsync explain`
to confirm exact field names before editing `config.yml`.

## Manifest: anchors

Each page maps to one or more **sources**. A source is `repo` + `globs` + `symbols`.

```yaml
pages:
  - path: reference/cli-commands.mdx      # relative to docs_root
    sources:
      - repo: docsync                     # canonical owner/name OR bare name; matched by basename
        globs:                            # fnmatch patterns against changed file paths
          - src/docsync/cli.py
        symbols:                          # symbol names; trailing * = prefix match
          - app
          - run
          - bootstrap
    judge_required: false                 # reference pages autopass (precise anchors)

  - path: concepts/multi-repo-strategies.mdx
    sources:
      - repo: api                         # poly-repo: each source names its repo explicitly
        globs: [src/**/*.py]
      - repo: worker
        globs: [src/tasks/*.py]
        symbols: [process_alert*]
    max_diff_lines: 60                     # per-page guardrail on net changed lines
    judge_required: true                   # broad/narrative pages go through the Haiku judge
```

Key per-page / per-source fields:
- `path` — page path relative to `docs_root`.
- `sources[].repo` — empty string is a **wildcard** (valid only in mono/single mode). In **poly**
  mode every source must name its repo; matching is by repo basename, so `keephq/keep-api` and a
  local `/checkouts/keep-api` both match `keep-api`.
- `sources[].globs` — fnmatch patterns (`src/**/*.py`, `README*`).
- `sources[].symbols` — function/class names; trailing `*` is a prefix match.
- `judge_required` — `true` routes a page's anchor through the Haiku judge instead of autopassing
  into the editor. Use it for broad `concept`/`guide` pages; keep `false` for precise `reference`.
- `max_diff_lines` — guardrail; an edit exceeding the page's allowed diff size is rejected.

### Page kinds
- `reference` — precise anchors (specific files + symbols), `judge_required: false`, autopass.
- `concept` / `guide` — broad anchors (directory globs), `judge_required: true`, judged for
  relevance so unrelated diffs don't trigger noisy edits.

After any manifest edit, run `docsync doctor --docs-repo … --checkout name=path …` to confirm
every glob matches files and every symbol exists in the checkout.

## config.yml — fields that matter most

All optional; defaults shown.

| Field | Default | Why you'd touch it |
|-------|---------|--------------------|
| `models.edit_model` | `claude-opus-4-8` | Authoring + surgical edits. Override to match a gateway's model ID. |
| `models.judge_model` | `claude-haiku-4-5` | Relevance judge, self-critique, infer. |
| `models.edit_effort` | `high` | Opus reasoning effort for author/edit calls. |
| `docs_root` | `.` | Root of the docs tree, relative to the docs repo (e.g. `docs`). |
| `repo_mode` | `auto` | `mono`/`single`/`poly`; auto-detects from checkout + manifest. |
| `adapter` | `mintlify` | `mintlify` (.mdx + docs.json nav) or `markdown` (plain .md). |
| `thoroughness` | `medium` | `light`/`medium`/`high` — how much content to write. |
| `thoroughness_by_kind` | `{}` | Per-kind override (reference/concept/guide). |
| `ingest_exclude_dirs` | `[]` | Extra dir names to prune on ingest (skip non-product noise). |
| `min_edit_confidence` | `0.0` | Ship-safety: skip edits below this impact confidence. Raise (e.g. `0.7`) for a cautious first rollout. |
| `max_pages_per_run` | `0` | Cap pages edited per run (0 = unlimited); highest-confidence first. |
| `max_parallel_requests` | `4` | Concurrent LLM requests across judge + edit. |
| `readability_pass` | `false` | Fact-frozen polish pass (CLI `--polish`). One extra call/page. |
| `self_critique` | `true` | Drops edit ops not justified by the diff (CLI `--no-self-critique`). |
| `anchor_autopass` | `true` | Anchor hits skip the judge entirely. |
| `judge_confidence_threshold` | `0.5` | Min judge confidence to keep a candidate. |
| `reviewers` | `[]` | GitHub handles requested as reviewers on docs PRs. |
| `pr_labels` | `[docsync]` | Labels on opened docs PRs. |
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Recall-net model. **Set to a local path for air-gapped.** |
| `embedding_floor` | `0.2` | Min cosine similarity for an embedding candidate. |
| `embedding_top_k` | `5` | Max embedding candidates per diff before judging. |
| `monthly_budget_usd` | `null` | Advisory spend target on the dashboard. Never blocks a run. |

## CLI flags worth knowing

- `--src-repo name=path` — repeatable; the multi-repo input for `run`/`bootstrap`/`infer`.
- `--checkout name=path` — repeatable; `doctor`'s equivalent for validating against real code.
- `--dry-run` / `--no-dry-run` — `run`/`bootstrap`/`infer` default to dry-run.
- `--plan-only` — `bootstrap`: produce the IA plan without authoring pages.
- `--open-pr` — open a PR (needs `gh` configured for your git host); off until you wire hosting.
- `--backend api|claude-code` — see SKILL.md "Backends and models".
- `--self-critique` / `--no-self-critique`, `--polish`, `--thoroughness light|medium|high`.
- `--min-confidence`, `--max-pages`, `--max-parallel` — per-run overrides of the config dials.
- `--use-embeddings` — toggle the recall-net for impact mapping.
