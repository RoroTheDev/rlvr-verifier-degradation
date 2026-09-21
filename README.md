# rlvr-verifier-degradation

Overview
This repository contains the early setup and structure for our RLVR verifier degradation project.
The goal is to study how verifier error — including rate, asymmetry, and persistence — affects RLVR/GRPO training stability across different regimes.

This repo will hold:

the noise model implementation

persistence masks

GRPO training scripts

experiment configs

logs and analysis outputs

More components will be added as the project progresses.

## Qwen decontamination pipeline

The contamination pipeline can use `Qwen/Qwen2.5-0.5B-Instruct` as a semantic
near-duplicate judge after exact and MinHash filtering:

```bash
python contamination_pipeline/decontamination_pipeline.py
```

For a bounded smoke test, use `--qwen-limit 10`. To run only the deterministic
filters, use `--disable-qwen`. This is Qwen-assisted filtering against
MATH-500; it is not evidence about Qwen's private pretraining corpus and must
not be described as proof that Qwen training data is contamination-free.

Work Log
2026‑08‑30
Created the GitHub repository

Added initial project structure
