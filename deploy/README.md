# Deploying docsync against the Keep repos

This folder stages the **configuration** docsync needs to run automatically against the
Keep platform. It adds only CI workflows + `.docsync/` config — **never application
code**. Everything here is also applied to each repo on a dedicated `docsync-wiring`
branch for review (nothing is pushed; nothing on a default branch is changed).

## Topology

| Repo | Branch base | What gets added |
|------|-------------|-----------------|
| `venaTeam/keep-api-gateway` | `dev` | `.github/workflows/notify-docsync.yml` |
| `venaTeam/keep-event-handler` | `dev` | `.github/workflows/notify-docsync.yml` |
| `venaTeam/keep-workflows` | `dev` | `.github/workflows/notify-docsync.yml` |
| `venaTeam/keep-ui` | `dev` | `.github/workflows/notify-docsync.yml` |
| `venaTeam/keep-developer-docs` | `main` | `.github/workflows/docsync.yml` + `.docsync/{config,manifest}.yml` + `.docsync/.gitignore` |

Flow: a push to a service repo's `dev` → `notify-docsync.yml` sends a `repository_dispatch`
(`event_type: docsync`) to the docs repo → `docsync.yml` runs `docsync run --from-event`,
fetches the diff via the GitHub API (no clone), and opens a docs PR.

## Secrets to configure (in GitHub → repo → Settings → Secrets)

**Each service repo:**
- `DOCSYNC_DISPATCH_TOKEN` — a PAT (or fine-grained token) allowed to send a
  `repository_dispatch` to `venaTeam/keep-developer-docs`.

**`keep-developer-docs`:**
- `ANTHROPIC_API_KEY` — for the `api` backend (Opus edits + Haiku judge).
- `DOCSYNC_GH_TOKEN` — token with `contents:read` on the `venaTeam/*` service repos so
  docsync can read the diff via the API. (The default `GITHUB_TOKEN` opens the docs PR.)

## Manifest

`keep-developer-docs/.docsync/manifest.yml` maps each doc page to the source globs/symbols
that should trigger an update. Validate it before relying on the automation:

```bash
docsync doctor --docs-repo /path/to/keep-developer-docs \
  --checkout venaTeam/keep-api-gateway=/path/to/keep-api-gateway \
  --checkout venaTeam/keep-event-handler=/path/to/keep-event-handler \
  --checkout venaTeam/keep-workflows=/path/to/keep-workflows \
  --checkout venaTeam/keep-ui=/path/to/keep-ui
```

The embeddings cache (`.docsync/state/`) is regenerable and **git-ignored** (see
`.docsync/.gitignore`) — never commit `vectors.npy`.

## Publishing docsync

`docsync.yml` installs docsync with `pip install "git+https://github.com/<org>/docsync.git"`.
Replace `<org>/docsync` with the real remote once docsync is published (or switch to
`pip install docsync`).

## Run a real test WITHOUT any CI wiring (read-only)

`deploy/local-real-run.sh` exercises docsync against the real **local** checkouts using
`git diff` over recent history, writing **only** to a throwaway shadow copy of the docs
repo. It touches nothing in the Keep repos:

```bash
BACKEND=claude-code ./deploy/local-real-run.sh
```

## Applying / reviewing the branches

Each repo has a local `docsync-wiring` branch with just these additions. Review, then push
when ready, e.g.:

```bash
git -C /path/to/keep-api-gateway push -u origin docsync-wiring   # opens a PR against dev
```
