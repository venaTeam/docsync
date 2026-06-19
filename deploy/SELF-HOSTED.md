# docsync — self-hosted dogfood loop

docsync documents **itself**: the `docs/` Mintlify site is generated and kept live by
docsync running against its own source. This is the live-docs differentiator proven on
the tool that implements it, in a single repo (no cross-repo dispatch).

## Pieces

| Piece | Path | Role |
|-------|------|------|
| Docs site | `docs/` | Mintlify site (`docs.json` + `*.mdx`), the `docs_root` |
| Config | `.docsync/config.yml` | `docs_root: docs`, models, guardrails |
| Manifest | `.docsync/manifest.yml` | anchors each page → `src/docsync/*.py` (drives impact mapping) |
| Cursor | `.docsync/state/cursors.json` | last processed head_sha (idempotency) |
| CI | `.github/workflows/docsync-self.yml` | on push to `main` touching `src/docsync/**` → opens a docs PR |

## Flow

```
push to main (src/docsync/** changed)
  └─ docsync-self workflow
       ├─ pip install -e ".[embeddings]"        # same-repo, editable
       └─ docsync run --from-event $GITHUB_EVENT_PATH --docs-repo . --backend api --open-pr
            ├─ diff (before..after) via GitHub API
            ├─ impact map (anchors + embeddings; narrative pages judge-gated)
            ├─ Opus edits → validate
            └─ open PR updating docs/  ← human reviews + merges (NEVER auto-merged)
```

The event repo `venaTeam/docsync` and the manifest's source repo `docsync` reconcile
through `impact._repo_key` (last path segment), so anchors match.

## Tokens / secrets to generate

For the self-hosted loop, the **only** secret needed is the model key:

| Secret | Where | Why |
|--------|-------|-----|
| `ANTHROPIC_API_KEY` | repo secret on `venaTeam/docsync` | Opus edits + Haiku judge (`--backend api`) |

`GITHUB_TOKEN` is provided automatically by Actions — it reads the push diff via the
API **and** opens the docs PR in this same repo. No PAT, no `DOCSYNC_GH_TOKEN`, no
cross-repo install token (the Keep cross-repo setup needs those; the self loop does not).

### One-time repo settings

1. **Settings → Secrets and variables → Actions** → add `ANTHROPIC_API_KEY`:
   ```
   gh secret set ANTHROPIC_API_KEY --repo venaTeam/docsync
   ```
2. **Settings → Actions → General → Workflow permissions** →
   enable **"Allow GitHub Actions to create and approve pull requests"** (required for
   `--open-pr` with the default `GITHUB_TOKEN`).
3. **Branch protection** on `main` requiring a review — so docsync's PRs are always
   approved by a human before merge.

## Local dry run (no API key, no writes)

The `claude-code` backend reuses the local Claude CLI auth, and `--dry-run` only writes
a patch + report:

```
poetry run docsync run \
  --src-repo . --base <old-sha> --head <new-sha> \
  --docs-repo . --backend claude-code   # dry-run is the default
```
