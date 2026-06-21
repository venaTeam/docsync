# Vendored embedding models

This directory holds an **offline copy of the embeddings recall-net model**, so docsync
can run in an air-gapped environment with **no HuggingFace mirror**. When a model is
vendored here, `embeddings.resolve_model_source` loads it locally (no network) and the
wheel built from this repo ships it as package data.

It is **empty by default** to keep the dev checkout lean. Populate it on a connected host:

```bash
python scripts/vendor_model.py            # -> src/docsync/_models/all-MiniLM-L6-v2/
poetry build                              # the wheel now bundles the model
```

The model files are large and tracked with **git-lfs** (see `.gitattributes`). Committing
them is optional: if you only need the model *in the wheel* (not the git history), run
`vendor_model.py` in CI right before `poetry build` and don't commit the bytes.

Layout once vendored: `all-MiniLM-L6-v2/` with `config.json`, the tokenizer files,
`model.safetensors`, `modules.json`, and `1_Pooling/`.
