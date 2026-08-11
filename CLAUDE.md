# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bioinformatics ML project for predicting T3E (Type III Effector) secretion signals in bacterial proteins using protein language models (ESM, ProtT5, ProteinBERT). The final production model uses frozen ProtT5 embeddings + an MLP classifier head.

## Repository Layout

The repo is split into three top-level areas:

- **`common/`** — code, models, and raw data shared by both tracks below. Nothing in `common/` imports from `effectidor2_paper/` or `runtime_optimization/`.
- **`effectidor2_paper/`** — the analysis behind the Effectidor2 paper: dataset processing output, embedding calculation, classic-ML and finetuning pipelines, the production inference script, and the paper's results.
- **`runtime_optimization/`** — work-in-progress on a two-stage cascade (ESM-6 → ProtT5) to reduce inference cost. May import code from `effectidor2_paper/` (it reuses the paper's embedding/classifier code against its own data), but never the reverse.

- `common/consts.py` — the only Python file in `common/`; holds universal constants (model registry, batch size, random state, `PRETRAINED_MODELS_DIR`).
- `effectidor2_paper/src/utils/consts_paths.py` — paper-specific data/output paths (`DATASETS_DIR`, `CLASSIFIERS_OUTPUT_DIR`, `FINAL_RESULTS`, `FIXED_*_FILE`, `PROTEIN_BERT_DIR`, `FINETUNE_NUMBER_OF_EPOCHS`, `PROTEIN_BERT_MODEL_NAME`) plus `read_fasta_utils.py`/`dataset_readers.py` (only the paper's code reads FASTA files today).
- `runtime_optimization/src/utils/consts_paths.py` — optimization-specific data/output paths, same constant names as the paper's, pointing at `runtime_optimization/data` and `runtime_optimization/outputs`.

Imports are absolute from the repo root, e.g. `from common.consts import BATCH_SIZE` or `from effectidor2_paper.src.utils.consts_paths import FINAL_RESULTS`.

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
python effectidor2_paper/src/inference/predict_secretion_signal.py \
    --input_fasta_file <path_to_fasta> \
    [--batch_size 16] \
    [--load_llm_from_disk]   # use local model files instead of HuggingFace hub
```

Output: CSV with T3E secretion signal probabilities. Input should be FASTA files with N-terminal 100 amino acids per protein.

## Running Training Scripts

```bash
# Calculate embeddings (must run before training classifiers)
python effectidor2_paper/src/pretrained_embeddings/calc_pt5_embeddings.py
python effectidor2_paper/src/pretrained_embeddings/calc_esm_embeddings.py

# Train classical ML classifiers on precomputed embeddings
python effectidor2_paper/src/classic_ml_classifiers/train_classifiers_on_embeddings.py

# Finetune a full model
python effectidor2_paper/src/finetune_pretrained_models/pt5/main_train.py
python effectidor2_paper/src/finetune_pretrained_models/esm/main_train.py

# Train the final production model on all data
python effectidor2_paper/src/train_final_model_on_all_data/train_final_model_on_all_data.py

# Analyze and compare all results
python effectidor2_paper/src/analyze_results/analyze_results.py
```

## Architecture Overview

**Two training paradigms:**
1. **Finetuning** (`effectidor2_paper/src/finetune_pretrained_models/`) — full model or LoRA-based finetuning tracked via Weights & Biases
2. **Embedding + classifier** (`effectidor2_paper/src/classic_ml_classifiers/`) — freeze the LLM, generate `.npy` embedding files, train lightweight sklearn classifiers (SVM, RF, MLP, XGBoost, LightGBM, etc.)

**Production model** (`effectidor2_paper/results/trained_pt5_head/model.pkl`): joblib-serialized MLP with architecture Dense→Dropout→Dense→Tanh→Dropout→Output on top of mean-pooled ProtT5 last hidden states.

**Inference pipeline** (`effectidor2_paper/src/inference/predict_secretion_signal.py`): standalone script that loads the ProtT5 model + trained MLP head, computes embeddings on the fly, and returns predictions.

## Key Constants and Paths

Constants are split across three files (see Repository Layout above): `common/consts.py` for universal constants, and a `consts_paths.py` in each of `effectidor2_paper/src/utils/` and `runtime_optimization/src/utils/` for that track's data/output locations. `PROJECT_BASE_DIR` (in `common/consts.py`) resolves to the repo root.

Important defaults:
- Default batch size: 8
- Random state: 42
- Finetuning epochs: 10
- Paper's dataset: `effectidor2_paper/data/new_data_processed/`
- Optimization track's dataset: `runtime_optimization/data/data_processing_29_12_25/4__Final_Datasets/`
- Primary metric: Matthews Correlation Coefficient (MCC)

## Data Format

- Positive examples (T3E, label=1) and negative examples (non-T3E, label=0) are separate FASTA files, N-terminal 100 AA sequences
- Raw source data (shared by both tracks) lives in `common/data/raw_data/`
- Precomputed embeddings cached as `.npy` files
- Results stored as CSVs under each track's own results/outputs directory

## Experiment Tracking

Finetuning runs log to Weights & Biases:
- ESM project: `t3e_secretion_signal_esm_new_data`
- ProtT5 project: `t3e_secretion_signal_pt5_new_data`

## Models Directory

Local model weights are stored directly in `common/models/` (ESM2 variants from 6M to 3B params, ProtT5 3B, ProteinBERT 16M). Pass `--load_llm_from_disk` in inference to use local weights instead of downloading from HuggingFace.

## ProteinBERT Submodule

`effectidor2_paper/protein_bert/` is a custom fork of ProteinBERT (git submodule). It must use the separate TensorFlow conda environment and cannot be mixed with the PyTorch-based models. Clone with `--recursive` to include it.

## Inference Runtime Optimization (Cascade / Mixed Classifier)

This work-in-progress, under `runtime_optimization/`, explores a two-stage cascade to reduce inference cost while preserving accuracy. The idea: run the fast ESM-6 (6M param) model first, and only escalate to the expensive ProtT5 model for sequences where ESM-6 is uncertain.

**How the cascade works:**
- ESM-6 classifier produces a probability for each sequence.
- Sequences where the probability falls in `[accuracy_threshold, 1 - accuracy_threshold]` are considered uncertain and escalated to ProtT5.
- `accuracy_threshold=1.0` → ESM-6 decides everything (fastest); `accuracy_threshold=0.0` → ProtT5 decides everything (most accurate, slowest).
- Intermediate thresholds (0.1–0.4) trade speed against accuracy.

**Key scripts:**

| Script | Purpose |
|--------|---------|
| `runtime_optimization/src/classic_ml_classifiers/test_mixed_classifiers/test_mixed_classifiers_on_embeddings.py` | Benchmarks the cascade on the labeled test set; reports MCC, AUPRC, and elapsed time at a given threshold |
| `runtime_optimization/src/inference/effectidor_samples/infer_mixed_classifiers_on_embeddings.py` | Runs the cascade on real (unlabeled) effectidor sample FASTA files; records inference time only |
| `runtime_optimization/src/inference/effectidor_samples/combine_results.py` | Aggregates per-sample timing CSVs into `results/aggregated_elapsed_times.csv` and a bar plot |

**SLURM scripts** (`runtime_optimization/src/inference/effectidor_samples/infer_mixed_1.sh` – `infer_mixed_4.sh`): submit jobs for effectidor sample sets 1–4, each sweeping all threshold values (0, 0.1, 0.2, 0.3, 0.4, 1.0).

**Output directories:**
- `runtime_optimization/outputs/embeddings_classifiers/{esm_6,mixed_models,pt5,pt5_layers_expirement}/` — results from the labeled test benchmark
- `runtime_optimization/src/inference/effectidor_samples/results/<sample_id>/mixed_<threshold>/` — per-sample timing results from real effectidor runs
