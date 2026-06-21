# Air-gapped / on-prem deployment runbook

Goal: run docsync on a host with **no public internet** — only a **private PyPI artifactory** and
an **internal Anthropic-compatible LLM gateway**. The runtime must make **zero** egress to
`api.anthropic.com`, `pypi.org`, or `huggingface.co`.

Why this works with no code change: `get_client("api")` in `src/docsync/llm_backends.py` calls
`anthropic.Anthropic()` with no arguments, and the Anthropic SDK reads `ANTHROPIC_BASE_URL` and
`ANTHROPIC_API_KEY` from the environment. Pointing at an internal gateway is therefore just env
vars. (If the only available endpoint is **AWS Bedrock**, that needs a small new backend using
`anthropic.AnthropicBedrock()` — out of scope for the first milestone.)

## Phase A — Mirror dependencies + publish docsync

Do this on a **connected** build host, then move artifacts inside the air-gap.

1. **Resolve the full dependency closure**, including the `embeddings` extra. The base set is
   light (`anthropic`, `pydantic`, `typer`, `ruamel.yaml`, `python-frontmatter`, `unidiff`,
   `numpy`); the `embeddings` extra is heavy: `sentence-transformers` pulls `torch`,
   `transformers`, `huggingface-hub`, `tokenizers`, `safetensors`.
   ```bash
   poetry export -E embeddings --without-hashes -f requirements.txt -o req.txt
   pip download -r req.txt -d wheelhouse/      # match the on-prem OS/arch
   ```
   `torch` is large and platform-specific — download the wheel matching the air-gapped host's
   CPU/arch. A CPU-only torch build is sufficient for `all-MiniLM-L6-v2`.
2. **Build docsync as a wheel** so the offline host installs it like any package, not from git:
   ```bash
   poetry build            # -> dist/docsync-*.whl
   ```
3. **Upload** `wheelhouse/*` and `dist/docsync-*.whl` to the private artifactory.

## Phase A (offline host) — install

1. Point pip/Poetry at the artifactory and disable the public PyPI fallback — e.g.
   `PIP_INDEX_URL=https://artifactory.internal/api/pypi/pypi/simple` in `~/.config/pip/pip.conf`,
   or a Poetry `[[tool.poetry.source]]` with `priority = "primary"`.
2. Install:
   ```bash
   pip install "docsync[embeddings]"          # or: poetry install -E embeddings (vendored source)
   docsync --help && docsync explain          # verify the CLI resolves
   ```

## Phase B — embedding model + wire the gateway

**Preferred (no HuggingFace mirror needed): bundle the model in the wheel.** Build the wheel with
the model vendored in — on a connected host run `python scripts/vendor_model.py` before
`poetry build` (or `poetry build` once the model is committed via git-lfs). The model ships as
package data, and `embeddings.resolve_model_source` loads it locally and automatically — **no HF
download, no `embedding_model` config.** This is the cleanest fit when you have torch but no HF
mirror. Belt-and-suspenders, still export `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so nothing
can reach the Hub.

**Alternative (model not bundled): stage it on the host.**
1. On a connected host download `sentence-transformers/all-MiniLM-L6-v2` and copy the directory
   inside the air-gap, e.g. `/opt/docsync/models/all-MiniLM-L6-v2`.
2. Point config at it + force offline. In `.docsync/config.yml`:
   ```yaml
   embedding_model: /opt/docsync/models/all-MiniLM-L6-v2
   ```
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   export HF_HOME=/opt/docsync/hf-cache       # local cache, never the network
   ```
   `SentenceTransformer()` accepts a local directory, so this avoids HuggingFace Hub entirely.
3. **Wire the LLM gateway (no code change):**
   ```bash
   export ANTHROPIC_BASE_URL=https://llm-gateway.internal/anthropic
   export ANTHROPIC_API_KEY=<gateway-token>
   # keep --backend api (the default)
   ```
   The gateway must serve the models named in `config.yml` → `models`: `claude-opus-4-8`
   (edit/author) and `claude-haiku-4-5` (judge/critique/infer). If the gateway uses different
   model IDs, override `models.edit_model` / `models.judge_model` to match.

## Verification (prove zero egress)

Run these and watch the host's outbound connections (e.g. with `lsof -i` / a firewall log):

```bash
# LLM-free — should make NO network calls at all
docsync map --docs-repo /path/docs --src-repo api=/path/api --base <sha1> --head <sha2>

# Embeddings offline — builds vectors using the local model, no huggingface.co
docsync index --docs-repo /path/docs

# Full impact+edit dry run — only egress should be ANTHROPIC_BASE_URL
docsync run --docs-repo /path/docs --src-repo api=/path/api --base <sha1> --head <sha2> --dry-run
```

Pass criteria: the only outbound connection is to the gateway host. No `api.anthropic.com`, no
`pypi.org`, no `huggingface.co`. `pip show docsync sentence-transformers torch` resolve from the
artifactory.

## Running it under air-gapped GitLab CI (when you get there)

Once the offline CLI loop is proven, wiring it into the internal **GitLab** CI has a few specifics:

- **Fetch the diff locally — do NOT rely on bare auto-detect.** docsync parses GitLab's `CI_*`
  vars, but its default diff runner for the CI path is `diff_github` (`gh api compare` against
  github.com) — which is unreachable in an air-gapped GitLab. Instead pass an explicit **local**
  `--src-repo` (a checkout with `.git`), so docsync uses `diff_local` (offline `git diff`):
  ```bash
  docsync run --docs-repo /path/docs \
    --src-repo "$CI_PROJECT_NAME=$CI_PROJECT_DIR" \
    --base "$CI_MERGE_REQUEST_DIFF_BASE_SHA" \
    --head "$CI_MERGE_REQUEST_SOURCE_BRANCH_SHA"
  ```
  (`$CI_PROJECT_DIR` is the source repo GitLab already cloned for the job.) `--from-event` is the
  GitHub event-JSON path and does nothing for GitLab.
- **`--open-pr` opens a GitLab MR natively** (set `forge: gitlab`, or rely on `auto` detecting it
  from the origin remote). It shells to the **`glab`** CLI, so the runner needs `glab` installed and
  a token that can push + open an MR in the docs repo. If you'd rather not ship `glab`, run without
  `--open-pr`: `run` writes the edits + a patch (`--report-path`) and the job opens the MR itself via
  `git push -o merge_request.create -o merge_request.target=<docs-default-branch>` or the MR API with
  `$CI_JOB_TOKEN`. Either way the `docs: sync …` MR is always human-reviewed.
- **Make the CI job hermetic.** Bake into the runner image: the Artifactory `pip.conf`, the
  pre-staged embedding model directory, and the offline/gateway env vars
  (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HOME`)
  via the GitLab CI/CD variables — so every run is reproducible and can't reach out.
- **Swap the stock install.** The shipped `action.yml` is a GitHub Action that installs docsync
  with `pipx install "git+https://github.com/…"`. For GitLab, ignore it: in `.gitlab-ci.yml` run
  `pip install "docsync[embeddings]"` from Artifactory on a self-hosted runner.

## Deferred to a follow-up milestone

- **MR automation** wiring (`git push -o merge_request.create` / `glab` / API) to the internal
  GitLab docs remote. The first CLI milestone uses `--dry-run` and skips MRs.
- **`.gitlab-ci.yml`** running docsync on merges inside the air-gapped CI.
- **A Bedrock backend** in `llm_backends.py:get_client`, only if the LLM endpoint turns out to be
  Bedrock rather than an Anthropic-compatible gateway. Add it test-first (the suite fakes the
  client, so a new backend gets a focused unit test).
