#!/usr/bin/env bash
# build-airgap-bundle.sh — produce a self-contained docsync bundle for an air-gapped host.
#
# A Python *wheel* never bundles its dependencies, so "everything in one wheel" isn't a
# thing. What this builds instead is a **wheelhouse**: the docsync wheel plus the full
# transitive dependency closure (optionally including the heavy `embeddings` extra), plus
# the pre-staged embedding model. Upload the result to your Artifactory, then the offline
# host installs with a single `pip install` that resolves entirely from the mirror.
#
# IMPORTANT — platform matching:
#   Binary wheels (torch, tokenizers, numpy, …) are specific to OS + CPU arch + Python
#   version. Run this on a CONNECTED host whose platform MATCHES the air-gapped target
#   (e.g. linux/x86_64, CPython 3.11). Building on macOS for a Linux target produces the
#   wrong wheels. If you cannot match the host, use the cross-download flags noted below.
#
# Usage:
#   ./build-airgap-bundle.sh                       # embeddings included, python3.11
#   PYTHON=python3.13 WITH_EMBEDDINGS=0 ./build-airgap-bundle.sh
#
# Output: docsync-airgap-bundle.tgz  (wheelhouse/ + requirements.txt + models/ + INSTALL.md)
set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
OUT="${OUT:-docsync-airgap-bundle}"
WITH_EMBEDDINGS="${WITH_EMBEDDINGS:-1}"          # 1 = include the embeddings recall-net
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Cross-platform download (only if you can't build on a matching host). Uncomment and tune
# to your target; note torch may not publish a wheel for every (platform, py) combination:
#   PIP_DL_ARGS="--platform manylinux2014_x86_64 --only-binary=:all: --python-version 311 --abi cp311"
PIP_DL_ARGS="${PIP_DL_ARGS:-}"

echo ">> python: $($PYTHON --version 2>&1) | embeddings: $WITH_EMBEDDINGS"
rm -rf "$OUT" "$OUT.tgz"
mkdir -p "$OUT/wheelhouse" "$OUT/models"

# 1. Build the docsync wheel from this repo.
echo ">> building docsync wheel"
poetry build -f wheel >/dev/null
cp dist/docsync-*.whl "$OUT/wheelhouse/"

# 2. Resolve + download the full dependency closure.
EXTRAS=""
[ "$WITH_EMBEDDINGS" = "1" ] && EXTRAS="-E embeddings"
echo ">> exporting dependency closure ${EXTRAS:-(base only)}"
poetry export $EXTRAS --without-hashes -f requirements.txt -o "$OUT/requirements.txt"
echo ">> downloading wheels (this pulls torch et al. when embeddings are on — can be large)"
# shellcheck disable=SC2086
"$PYTHON" -m pip download -r "$OUT/requirements.txt" -d "$OUT/wheelhouse" $PIP_DL_ARGS

# 3. Pre-stage the embedding model (a runtime HF download otherwise — never in a wheel).
if [ "$WITH_EMBEDDINGS" = "1" ]; then
  echo ">> staging embedding model: $EMBED_MODEL"
  "$PYTHON" -m pip install -q --upgrade huggingface_hub
  "$PYTHON" - "$EMBED_MODEL" "$OUT/models" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
model, out = sys.argv[1], sys.argv[2]
dest = Path(out) / model.split("/")[-1]
snapshot_download(model, local_dir=str(dest))
print(f"   staged -> {dest}")
PY
fi

# 4. Drop an offline-install note next to the artifacts.
cat > "$OUT/INSTALL.md" <<EOF
# docsync air-gapped bundle

Built with: $($PYTHON --version 2>&1) · embeddings=$WITH_EMBEDDINGS

## Option A — upload wheels to Artifactory (recommended)
Upload everything in \`wheelhouse/\` to your internal PyPI repo, then on the offline host:

    pip install $( [ "$WITH_EMBEDDINGS" = "1" ] && echo '"docsync[embeddings]"' || echo "docsync" )

(point \`PIP_INDEX_URL\` at the Artifactory PyPI index, with no pypi.org fallback.)

## Option B — install straight from the wheelhouse (no index)
Copy \`wheelhouse/\` to the host and:

    pip install --no-index --find-links ./wheelhouse $( [ "$WITH_EMBEDDINGS" = "1" ] && echo '"docsync[embeddings]"' || echo "docsync" )

## Embedding model
Copy \`models/$(basename "$EMBED_MODEL")\` to e.g. /opt/docsync/models/, then in
\`.docsync/config.yml\`: \`embedding_model: /opt/docsync/models/$(basename "$EMBED_MODEL")\`
and export HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1.

## Verify
    docsync --help && docsync explain
EOF

tar czf "$OUT.tgz" "$OUT"
echo ">> done: $OUT.tgz ($(du -h "$OUT.tgz" | cut -f1))"
echo "   upload wheelhouse/* to Artifactory and copy models/ to the host (see $OUT/INSTALL.md)"
