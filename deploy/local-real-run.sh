#!/usr/bin/env bash
# Read-only "real run" of docsync against the local Keep checkouts.
#
# Exercises the full pipeline (diff -> impact -> edits -> validate) on REAL code using
# `git diff` over recent history. It is strictly read-only on the Keep service repos and
# writes ONLY to a throwaway shadow copy of the docs repo. Nothing in any Keep repo is
# modified, and no PR is opened.
#
# Usage:   BACKEND=claude-code ./deploy/local-real-run.sh        # local CLI auth
#          BACKEND=api ANTHROPIC_API_KEY=... ./deploy/local-real-run.sh
#          DEPTH=30 ./deploy/local-real-run.sh                   # diff over last N commits
set -euo pipefail

KEEP="${KEEP_DIR:-/Users/yarin/keep-namespace}"
DOCS="$KEEP/keep-developer-docs"
BACKEND="${BACKEND:-claude-code}"
DEPTH="${DEPTH:-20}"

# Shadow copy of the docs repo so docsync can write/patch without touching the original.
SHADOW="$(mktemp -d)/shadow-docs"
cp -R "$DOCS" "$SHADOW"
rm -rf "$SHADOW/node_modules" "$SHADOW/.docsync/state"
echo "shadow docs repo: $SHADOW"
echo "originals are READ-ONLY; writes go only to the shadow."
echo

for repo in keep-api-gateway keep-event-handler keep-workflows keep-ui; do
  src="$KEEP/$repo"
  [ -d "$src/.git" ] || { echo "skip $repo (not a git repo)"; continue; }
  head="$(git -C "$src" rev-parse HEAD)"
  base="$(git -C "$src" rev-parse "HEAD~${DEPTH}" 2>/dev/null \
          || git -C "$src" rev-list --max-parents=0 HEAD | head -1)"
  echo "=================================================================="
  echo "== $repo   ${base:0:8}..${head:0:8}"
  echo "=================================================================="
  docsync run \
    --src-repo "$src" \
    --base "$base" \
    --head "$head" \
    --docs-repo "$SHADOW" \
    --dry-run \
    --no-preflight \
    --backend "$BACKEND" || echo "(docsync exited non-zero for $repo — see above)"
  echo
done

echo "Done. Inspect the shadow for any written patch: $SHADOW/docsync.patch"
