# docsync air-gapped operator checklist

A one-page runbook for standing docsync up in an on-prem **air-gapped** network: a private
**Artifactory** PyPI mirror, an internal **Anthropic-compatible gateway**, and self-managed
**GitLab**. Hand this to whoever owns the environment. Companion files in this folder:
`build-airgap-bundle.sh` (builds the offline bundle) and `gitlab-ci.yml` (the pipeline).

> Why this works with no code change: docsync's `api` backend is a bare `anthropic.Anthropic()`,
> which honors `ANTHROPIC_BASE_URL`. Embeddings accept a local model path. GitLab MRs open
> natively via `glab` (`forge: gitlab`). See `references/air-gapped-setup.md` in the skill for detail.

## Phase A — Build & publish the offline bundle (on a CONNECTED host)

- [ ] Use a build host whose **OS + CPU arch + Python version match the air-gapped target**
      (binary wheels like torch are platform-specific). Confirm: `python3.11 --version`.
- [ ] Run `./build-airgap-bundle.sh` (set `PYTHON=` / `WITH_EMBEDDINGS=` as needed). It emits
      `docsync-airgap-bundle.tgz` = `wheelhouse/` (docsync wheel + full closure) + the staged
      embedding model + `INSTALL.md`.
- [ ] Upload `wheelhouse/*` to the Artifactory **PyPI** repo. Move `models/` inside the air-gap.
- [ ] Build a **runner image** for CI: base Python that matches the target, then
      `pip install "docsync[embeddings]"` from Artifactory, plus `git` and **`glab`**. Push it to
      the internal registry; reference it as `$DOCSYNC_RUNNER_IMAGE`.
- [ ] **Docs site rendering (Docusaurus + mermaid):** docsync authors ```` ```mermaid ````
      diagrams on architecture/concept pages. Mirror the docs site's npm dependencies into the
      Artifactory **npm** repo — including `@docusaurus/theme-mermaid` — and enable it in
      `docusaurus.config.js`: `markdown: { mermaid: true }` + `themes: ['@docusaurus/theme-mermaid']`.
      Mermaid is compiled into the static bundle at **build time** and renders client-side, so the
      built site needs no CDN or any other egress at runtime. On the air-gapped side, point npm at
      the mirror (`npm config set registry https://artifactory.internal.example/api/npm/npm/`).

## Phase B — Configure offline + the gateway (on the air-gapped host)

- [ ] Point pip/Poetry at Artifactory (`PIP_INDEX_URL=…`), with **no** pypi.org fallback.
- [ ] Install & verify: `pip install "docsync[embeddings]"` → `docsync --help && docsync explain`.
- [ ] Stage the model and force offline in the docs repo `.docsync/config.yml`:
      `embedding_model: /opt/docsync/models/all-MiniLM-L6-v2`, and export
      `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME=/opt/docsync/hf-cache`.
- [ ] Wire the gateway: `ANTHROPIC_BASE_URL=<internal>`, `ANTHROPIC_API_KEY=<token>`; keep
      `--backend api`. If the gateway renames models, override `models.edit_model` /
      `models.judge_model` in config.yml to the IDs it serves.
- [ ] Set `forge: gitlab` in the **docs repo** `.docsync/config.yml` (or rely on `auto` if the
      docs origin URL contains "gitlab").

## Phase C — First poly-repo onboarding (CLI, then CI)

- [ ] Scaffold: `docsync init --minimal --detect --docs-repo /path/docs`.
- [ ] Author the site — **dry-run first**:
      `docsync bootstrap --docs-repo /path/docs --src-repo api=/path/api --src-repo worker=/path/worker --src-repo ui=/path/ui --dry-run`,
      review the IA + cost, then re-run `--no-dry-run`.
- [ ] Validate anchors: `docsync doctor --docs-repo /path/docs --checkout api=/path/api …` (fix drift, repeat).
- [ ] Prove the live loop offline (watch egress — only the gateway should be hit):
      `docsync map …` then `docsync run … --dry-run` on a real diff.
- [ ] Wire CI: copy `gitlab-ci.yml` into each **service** repo; set the masked CI/CD variables
      (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `DOCS_BOT_TOKEN`, `DOCS_REPO_URL`,
      `DOCS_DEFAULT_BRANCH`, `DOCSYNC_RUNNER_IMAGE`). The docs MR is **always human-reviewed —
      never auto-merged.**

## Zero-egress acceptance test

- [ ] While running `docsync index` and `docsync run … --dry-run`, watch connections
      (`lsof -i` / firewall log). The **only** outbound host is the gateway. No `api.anthropic.com`,
      no `pypi.org`, no `huggingface.co`. `pip show docsync sentence-transformers torch` all
      resolve from Artifactory.
- [ ] Docs site: `npm install && npm run build` hits **only** the Artifactory npm mirror (no
      `registry.npmjs.org`, no CDN hosts), and mermaid diagrams render in the built site served
      offline (`npm run serve` with networking blocked).
