# Air-gapped docsync kit

Operator assets for running docsync in an on-prem **air-gapped** network (private Artifactory
PyPI mirror + internal Anthropic-compatible gateway + self-managed GitLab).

| File | What it's for |
|------|---------------|
| [`operator-checklist.md`](operator-checklist.md) | One-page runbook, Phases A–C, with a zero-egress acceptance test. **Start here.** |
| [`build-airgap-bundle.sh`](build-airgap-bundle.sh) | Builds the offline bundle: docsync wheel + full dependency closure (wheelhouse) + the staged embedding model. Run it on a host matching the air-gapped target's OS/arch/Python. |
| [`build-wheelhouses.sh`](build-wheelhouses.sh) | Builds dependency wheelhouses for **linux+windows × py3.10/3.11** (CPU torch) via Docker — Linux resolved on-platform in slim containers, Windows cross-downloaded. Uses [`direct.txt`](direct.txt) / [`direct-win.txt`](direct-win.txt). |
| [`gitlab-ci.yml`](gitlab-ci.yml) | Drop-in pipeline for a **service** repo: on each MR it diffs locally (offline) and opens a native docs **MR** via `glab`. |

Background and the *why* behind each step live in the docsync skill at
`.claude/skills/docsync/references/air-gapped-setup.md`.

These are templates — swap the `*.internal.example` hosts, tokens, and image names for yours.
