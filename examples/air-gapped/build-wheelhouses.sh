#!/usr/bin/env bash
# Build docsync dependency wheelhouses for linux+windows × py3.10/3.11 (CPU torch).
# Linux closures resolve correctly inside official slim containers; windows are
# cross-downloaded (--platform win_amd64) with colorama added (a marker-gated tqdm dep
# that a non-Windows resolve drops). Run on a host with Docker. Requires direct.txt.
set -euo pipefail
REQ="${REQ:-direct.txt}"; REQ_WIN="${REQ_WIN:-direct-win.txt}"; OUTDIR="${OUTDIR:-.}"
CPU="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"
linux () { local py=$1; local out="$OUTDIR/wheelhouse-linux-cp${py/./}"; rm -rf "$out"; mkdir -p "$out"
  docker run --rm --platform linux/amd64 -v "$PWD/$REQ:/req.txt:ro" -v "$out:/out" "python:$py-slim" \
    bash -c "pip install -q --upgrade pip >/dev/null 2>&1; pip download -r /req.txt -d /out $CPU"; }
windows () { local py=$1; local out="$OUTDIR/wheelhouse-win-cp${py/./}"; rm -rf "$out"; mkdir -p "$out"
  docker run --rm --platform linux/amd64 -v "$PWD/$REQ_WIN:/req.txt:ro" -v "$out:/out" "python:$py-slim" \
    bash -c "pip install -q --upgrade pip >/dev/null 2>&1; pip download -r /req.txt -d /out \
      --platform win_amd64 --python-version $py --implementation cp --abi cp${py/./} --only-binary=:all: $CPU"; }
for py in 3.10 3.11; do linux "$py"; windows "$py"; done
echo "done -> $OUTDIR/wheelhouse-*"
