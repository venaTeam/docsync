# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in **docsync** — the product repo
(`venaTeam/docsync`). docsync is fully editable; it is the thing being built.

## What docsync is

docsync keeps documentation in sync with code. It ingests a merged PR's diff from a
**service repo**, maps it to the doc pages it affects (which usually live in a **different**
repo), uses an LLM to make **surgical** edits to the existing `.mdx`, validates them, and
opens a **reviewable PR** against the docs repo. It runs in CI. The cross-repo shape
(code and docs in separate repos) is the niche — most tools are single-repo, API-reference
only.

Two generation flows, plus onboarding/diagnostic commands:
- **`run`** — diff-driven: edit existing pages from a code change (the live loop).
- **`bootstrap`** — greenfield: author a whole sectioned docs site from a code snapshot.
- **`infer`** — brownfield: propose manifest anchors for an existing docs site.
- **`doctor`** — validate the manifest against real checkouts. **`init`** — scaffold `.docsync/`.
- **`map` / `index` / `eval`** — impact preview / embeddings build / golden-case harness.

## Stack & conventions

- **Python ≥3.10** (the venv runs 3.13), **Poetry** with an in-project `.venv` (`poetry.toml`).
- **Pydantic v2** (≥2.6) — note this differs from the sibling Keep services, which are v1.
  **Typer** CLI, **Anthropic SDK**. Optional `embeddings` extra = `sentence-transformers`.
- **Lint/format is ruff only** (line-length 100, `src=["src"]`). No black/isort.
- Models: edit/author = **Opus** (`claude-opus-4-8`), judge/critique/infer = **Haiku**
  (`claude-haiku-4-5`). Configurable via `ModelConfig`.
- LLM **backends**: `api` (`ANTHROPIC_API_KEY`) or `claude-code` (shells to the local
  `claude` CLI — free via subscription, used for local dogfooding).

## Commands

```bash
poetry install -E embeddings                 # deps + the embeddings recall-net
poetry run docsync run --docs-repo ./docs --src-repo owner/name --base <sha> --head <sha>
poetry run docsync bootstrap --docs-repo ./docs --src-repo name=path [--polish]
poetry run pytest -q                         # full suite (no network — fakes the client)
poetry run pytest tests/test_pipeline.py::test_x -v   # single test
poetry run ruff check src/ tests/            # lint (must be clean before a PR)
```

Tests use **fake clients keyed on `output_format`** (see `test_pipeline.py`/`test_bootstrap.py`)
— never hit the network. Add tests next to the module you change.

## Architecture

The CLI (`cli.py`) wires stages; the core is pure (no git side effects until `pr.py`).

**`run` pipeline** (`pipeline.py`): diff → **impact** → **edit** → **validate** → PR.
- **Impact** (`impact.py`): match the diff's changed paths/symbols against manifest
  **anchors** (deterministic, autopass) + an **embeddings** recall-net (`embeddings.py`),
  then a **Haiku judge** confirms non-autopass candidates. Pages marked `judge_required`
  never autopass.
- **Edit** (`edits.py`): Opus returns surgical find/replace `EditOp`s (a `PageEdit`) —
  **never a full rewrite**; applied with a strict single-occurrence check.
- **Critique** (`critique.py`, opt-in `--self-critique`): Haiku drops ops not faithful to the diff.
- **Polish** (`polish.py`, opt-in `--polish`): fact-frozen readability pass; falls back to
  the surgical edit on any failure.
- **Validate** (`validate.py`): hard gates — frontmatter freeze, structural-signature
  integrity (additive leaf-component growth allowed, decreases/container changes rejected),
  diff-size guardrail, not-truncated; soft gate — broken links.

**`bootstrap`** (`bootstrap.py`): ingest → plan IA (`DocPlan`, Haiku) → author each page
(Opus, kind-specific) → validate → emit pages + ordered nav + manifest anchors → PR.

Shared: **`style.py`** (the documentation-craft rules — inverted pyramid, scannability,
per-kind structure — consumed by author/edit/polish), **`ingest.py`** (read-only repo walk
+ symbol digests), **`cost.py`** (every LLM call goes through an injectable client wrapped
in `MeteredClient`; usage/cost lands on `RunUsage`), **`models.py`** (all Pydantic models),
**`report.py`** (console + PR-body rendering), **`adapters/mintlify.py`** (the only adapter).

## `.docsync/` — config the tool reads

- `config.yml` → `DocsyncConfig` (models, thresholds, `docs_root`, `max_pages_per_run`,
  `readability_pass`, …). All fields optional.
- `manifest.yml` → page → source **anchors** (`globs` + `symbols`, per repo). `judge_required`
  routes a page's broad anchor through the judge instead of autopassing into Opus. **Page
  kinds**: `reference` (precise anchors, autopass) · `concept`/`guide` (broad anchors, judged).
- `state/cursors.json` — idempotency cursor (last processed head per repo).

Keep anchors honest: after editing the manifest, run `docsync doctor`.

## Self-docs dogfood loop (important)

docsync documents **itself**: `docs/` is docsync's own Mintlify site, anchored to
`src/docsync/*.py` via `.docsync/manifest.yml`. `.github/workflows/docsync-self.yml` runs
`docsync run` on pushes to `main` and opens a `docs: sync …` PR. **Never auto-merge** these
docs PRs — they are always human-reviewed. When you change a public surface (a CLI flag, a
config field, a stage), expect the loop to propose a docs update on the next merge.

## Working norms

- Match the surrounding code's style; reuse existing helpers (check siblings before adding code).
- ruff clean + `pytest` green before any PR. Logical commits/branches.
- Commit/push only when asked. Git commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- PR bodies end with the Claude Code generation line.
- Edits stay faithful and surgical; don't loosen a validation gate without a test proving the
  new boundary (see `test_validate.py`).
