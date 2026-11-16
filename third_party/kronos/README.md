# Kronos (vendored, unmodified)

`model/` is copied verbatim from https://github.com/shiyu-coder/Kronos at commit
`67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13), MIT licence (see `LICENSE`).
Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", arXiv 2508.02739.

Only the three model files are vendored (pure PyTorch, reviewed 2026-11-16: no file, network,
pickle or subprocess calls). Weights are NOT in git; they are fetched from the Hugging Face Hub
(`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`, safetensors) on first use.

Used by `scripts/v5_fm_features.py` (V5_FINDINGS §3ew). Needs `einops`.
