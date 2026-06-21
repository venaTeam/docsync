#!/usr/bin/env python3
"""Vendor the embeddings model into the package for an offline (air-gapped) wheel.

Downloads a sentence-transformers model into `src/docsync/_models/<name>/` on a CONNECTED
host. The wheel built afterwards ships it as package data, and `embeddings.resolve_model_source`
loads it locally — so an air-gapped install needs no HuggingFace mirror and no `embedding_model`
config.

Keeps only what `SentenceTransformer` needs to load offline, and drops the duplicate
`pytorch_model.bin` when `model.safetensors` is present (halves the size).

Usage (on a host with internet + `pip install huggingface_hub`):
    python scripts/vendor_model.py
    python scripts/vendor_model.py --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Only the files needed to load + encode offline (skip README, ONNX, openvino, etc.).
ALLOW = [
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "model.safetensors",
    "1_Pooling/*",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id to vendor.")
    ap.add_argument(
        "--dest",
        default=str(Path(__file__).resolve().parent.parent / "src" / "docsync" / "_models"),
        help="Target _models dir (defaults to the package's).",
    )
    args = ap.parse_args()

    from huggingface_hub import snapshot_download  # connected host only

    out = Path(args.dest) / args.model.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model, local_dir=str(out), allow_patterns=ALLOW)

    # Prefer safetensors; drop the torch .bin duplicate if both came down.
    if (out / "model.safetensors").exists():
        legacy = out / "pytorch_model.bin"
        if legacy.exists():
            legacy.unlink()

    if not (out / "config.json").exists():
        raise SystemExit(f"vendor failed: no config.json under {out}")
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"vendored {args.model} -> {out}  ({size_mb:.0f} MB)")
    print("next: `poetry build` (the wheel now bundles the model)")


if __name__ == "__main__":
    main()
