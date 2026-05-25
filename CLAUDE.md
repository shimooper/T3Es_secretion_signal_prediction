# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bioinformatics ML project for predicting T3E (Type III Effector) secretion signals in bacterial proteins using protein language models (ESM, ProtT5, ProteinBERT). The final production model uses frozen ProtT5 embeddings + an MLP classifier head.

## Environment Setup

Two separate conda environments are required:

```bash
# Primary environment (PyTorch, Transformers, ESM, ProtT5)
conda env create -f env.yml
conda activate secretion_signal

# ProteinBERT only (TensorFlow-based, incompatible with primary env)
conda env create -f env_proteinbert.yml
conda activate secretion_signal_proteinbert
```

Key versions: Python 3.9.19, PyTorch 2.3.1 + CUDA 11.8, Transformers 4.43.3, scikit-learn 1.5.1.

## Running Inference

The primary entry point for new protein sequences:

```bash
python src/inference/predict_secretion_signal.py \
    --input_fasta_file <path_to_fasta> \
    [--batch_size 16] \
    [--load_llm_from_disk]   # use local model files instead of HuggingFace hub
```

Output: CSV with T3E secretion signal probabilities. Input should be FASTA files with N-terminal 100 amino acids per protein.

## Running Training Scripts

```bash
# Calculate embeddings (must run before training classifiers)
python src/pretrained_embeddings/calc_pt5_embeddings.py
python src/pretrained_embeddings/calc_esm_embeddings.py

# Train classical ML classifiers on precomputed embeddings
python src/classic_ml_classifiers/train_classifiers_on_embeddings.py

# Finetune a full model
python src/finetune_pretrained_models/pt5/main_train.py
python src/finetune_pretrained_models/esm/main_train.py

# Train the final production model on all data
python src/train_final_model_on_all_data/train_final_model_on_all_data.py

# Analyze and compare all results
python src/analyze_results/analyze_results.py
```

## Architecture Overview

**Two training paradigms:**
1. **Finetuning** (`src/finetune_pretrained_models/`) — full model or LoRA-based finetuning tracked via Weights & Biases
2. **Embedding + classifier** (`src/classic_ml_classifiers/`) — freeze the LLM, generate `.npy` embedding files, train lightweight sklearn classifiers (SVM, RF, MLP, XGBoost, LightGBM, etc.)

**Production model** (`final_results/trained_pt5_head/model.pkl`): joblib-serialized MLP with architecture Dense→Dropout→Dense→Tanh→Dropout→Output on top of mean-pooled ProtT5 last hidden states.

**Inference pipeline** (`src/inference/predict_secretion_signal.py`): standalone script that loads the ProtT5 model + trained MLP head, computes embeddings on the fly, and returns predictions.

## Key Constants and Paths

All shared constants (model paths, HuggingFace IDs, dataset paths, hyperparameters) live in `src/utils/consts.py`. `PROJECT_ROOT_DIR` is defined there and used throughout to resolve paths relative to the repo root.

Important defaults from `consts.py`:
- Default batch size: 8
- Random state: 42
- Finetuning epochs: 10
- Two dataset versions: "original" and "fixed" (`data/datasets_fixed/` is the current one)
- Primary metric: Matthews Correlation Coefficient (MCC)

## Data Format

- Input: FASTA files in `data/` with N-terminal 100 AA sequences
- Positive examples (T3E, label=1) and negative examples (non-T3E, label=0) are separate FASTA files
- Precomputed embeddings cached as `.npy` files
- Results stored as CSVs in `final_results/`

## Experiment Tracking

Finetuning runs log to Weights & Biases:
- ESM project: `t3e_secretion_signal_esm_new_data`
- ProtT5 project: `t3e_secretion_signal_pt5_new_data`

## Models Directory

Local model weights are stored in `models/` (ESM2 variants from 6M to 3B params, ProtT5 3B, ProteinBERT 16M). Pass `--load_llm_from_disk` in inference to use local weights instead of downloading from HuggingFace.

## ProteinBERT Submodule

`protein_bert/` is a custom fork of ProteinBERT. It must use the separate TensorFlow conda environment and cannot be mixed with the PyTorch-based models. Clone with `--recursive` to include it.

## Inference Runtime Optimization (Cascade / Mixed Classifier)

This work-in-progress explores a two-stage cascade to reduce inference cost while preserving accuracy. The idea: run the fast ESM-6 (6M param) model first, and only escalate to the expensive ProtT5 model for sequences where ESM-6 is uncertain.

**How the cascade works:**
- ESM-6 classifier produces a probability for each sequence.
- Sequences where the probability falls in `[accuracy_threshold, 1 - accuracy_threshold]` are considered uncertain and escalated to ProtT5.
- `accuracy_threshold=1.0` → ESM-6 decides everything (fastest); `accuracy_threshold=0.0` → ProtT5 decides everything (most accurate, slowest).
- Intermediate thresholds (0.1–0.4) trade speed against accuracy.

**Key scripts:**

| Script | Purpose |
|--------|---------|
| `src/classic_ml_classifiers/test_mixed_classifiers_on_embeddings.py` | Benchmarks the cascade on the labeled test set; reports MCC, AUPRC, and elapsed time at a given threshold |
| `src/inference/effectidor_samples/infer_mixed_classifiers_on_embeddings.py` | Runs the cascade on real (unlabeled) effectidor sample FASTA files; records inference time only |
| `src/inference/effectidor_samples/combine_results.py` | Aggregates per-sample timing CSVs into `results/aggregated_elapsed_times.csv` and a bar plot |

**SLURM scripts** (`src/inference/effectidor_samples/infer_mixed_1.sh` – `infer_mixed_4.sh`): submit jobs for effectidor sample sets 1–4, each sweeping all threshold values (0, 0.1, 0.2, 0.3, 0.4, 1.0).

**Output directories:**
- `outputs_optimization/embeddings_classifiers/{esm_6,mixed_0.1,mixed_0.2,mixed_0.3,mixed_0.4,pt5}/` — results from the labeled test benchmark
- `src/inference/effectidor_samples/results/<sample_id>/mixed_<threshold>/` — per-sample timing results from real effectidor runs
