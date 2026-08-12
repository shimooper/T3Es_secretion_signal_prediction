# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bioinformatics ML project for predicting T3E (Type III Effector) secretion signals in bacterial proteins using protein language models (ESM, ProtT5, ProteinBERT). The final production model uses frozen ProtT5 embeddings + an MLP classifier head.

## Repository Layout

The repo is split into three top-level areas:

- **`common/`** — code, models, and raw data shared by both tracks below. Nothing in `common/` imports from `effectidor2_paper/` or `runtime_optimization/` — any track-specific default (a data path, an output dir) is passed in by the caller rather than baked in here.
- **`effectidor2_paper/`** — the analysis behind the Effectidor2 paper: dataset processing output, embedding calculation, classic-ML and finetuning pipelines, the production inference script, and the paper's results.
- **`runtime_optimization/`** — work-in-progress on a two-stage cascade (ESM-6 → ProtT5) to reduce inference cost. May import code from `common/` (and, for finetuning-adjacent utilities, from `effectidor2_paper/`), but never the reverse.

- `common/consts.py` — universal constants (model registry, batch size, random state, `PRETRAINED_MODELS_DIR`).
- `common/read_fasta_utils.py` — generic FASTA reading helpers, no track-specific paths.
- `common/pretrained_embeddings/{calc_esm_embeddings,calc_pt5_embeddings}.py` — ESM/ProtT5 embedding calculation, shared by both tracks. The low-level `calc_embeddings_of_fasta_file_with_huggingface_model_{esm,pt5}(model_id, fasta_file_path, embeddings_file_path)` computes embeddings for one FASTA file; the higher-level `calc_{esm,pt5}_embeddings(model_id, split, positive_fasta_file, negative_fasta_file, embeddings_dir, ...)` adds `.npy` caching keyed by `split`. All path arguments are required — this module has no notion of "the paper's" or "the optimization track's" default paths. ProteinBERT embedding calculation stays in `effectidor2_paper/src/pretrained_embeddings/calc_proteinbert_embeddings.py` since it needs the `protein_bert` submodule and its own TensorFlow env; `runtime_optimization` never uses it.
- `common/classic_ml_classifiers/` — `classifiers_params_grids.py` (sklearn/xgboost/lightgbm param grids), `utils.py` (`prepare_Xs_and_Ys`, which takes an injected `calc_embeddings_fn` callable rather than knowing about specific model families), and `{train,test}_classifiers_on_embeddings.py` (`main()` takes `embeddings_dir`/`classifiers_output_dir`/`positive_fasta_file`/`negative_fasta_file` as required arguments, plus an optional `calc_embeddings_fn` override — omit it and it auto-picks the ESM or ProtT5 calculator based on `model_id`, via a lazy import so the PyTorch stack is never pulled in unless actually needed).
- `effectidor2_paper/src/classic_ml_classifiers/{train,test}_classifiers_on_embeddings.py` and `runtime_optimization/src/classic_ml_classifiers/{train,test}_classifiers_on_embeddings.py` — thin per-track wrapper scripts: same CLI as before, calling `common`'s `main()` with that track's own paths from its `consts_paths.py`. The paper's wrapper additionally passes an explicit `calc_embeddings_fn` for `model_id='protein_bert'` (lazily imported, so it's only ever touched when that model is requested).
- `effectidor2_paper/src/utils/consts_paths.py` — paper-specific data/output paths (`DATASETS_DIR`, `EMBEDDINGS_DIR`, `CLASSIFIERS_OUTPUT_DIR`, `FINAL_RESULTS`, `FIXED_*_FILE`, `PROTEIN_BERT_DIR`, `FINETUNE_NUMBER_OF_EPOCHS`, `PROTEIN_BERT_MODEL_NAME`) plus `dataset_readers.py` (reads the paper's train/test FASTA files, using `common/read_fasta_utils.py`).
- `runtime_optimization/src/utils/consts_paths.py` — optimization-specific data/output paths, same constant names as the paper's (`EMBEDDINGS_DIR`, `CLASSIFIERS_OUTPUT_DIR`, `FIXED_*_FILE`, ...), pointing at `runtime_optimization/data` and `runtime_optimization/outputs` instead.

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
# Embeddings are calculated on demand by train/test_classifiers_on_embeddings.py (below) via
# common/pretrained_embeddings/{calc_pt5_embeddings,calc_esm_embeddings}.py — no separate step needed.
# Those modules are also runnable standalone if you want to precompute/cache embeddings directly:
python common/pretrained_embeddings/calc_pt5_embeddings.py --model_id pt5 --split train \
    --positive_fasta_file <path> --negative_fasta_file <path> --embeddings_dir <path>

# Train classical ML classifiers on precomputed embeddings (paper's own paths)
python effectidor2_paper/src/classic_ml_classifiers/train_classifiers_on_embeddings.py --model_id pt5

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
2. **Embedding + classifier** (`effectidor2_paper/src/classic_ml_classifiers/`, backed by the shared `common/classic_ml_classifiers/` implementation) — freeze the LLM, generate `.npy` embedding files, train lightweight sklearn classifiers (SVM, RF, MLP, XGBoost, LightGBM, etc.)

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
| `runtime_optimization/src/classic_ml_classifiers/train_classifiers_on_embeddings.py` | Thin wrapper that calls `common.classic_ml_classifiers.train_classifiers_on_embeddings.main()` with this track's own `EMBEDDINGS_DIR`/`CLASSIFIERS_OUTPUT_DIR`/FASTA files (single model_id, e.g. `esm_6` or `pt5` — the base for the cascade) |
| `runtime_optimization/src/classic_ml_classifiers/test_classifiers_on_embeddings.py` | Same, wrapping `common.classic_ml_classifiers.test_classifiers_on_embeddings.main()` |
| `runtime_optimization/src/test_mixed_classifiers/test_mixed_classifiers_on_embeddings.py` | Benchmarks the cascade on the labeled test set; reports MCC, AUPRC, and elapsed time at a given threshold |
| `runtime_optimization/src/inference/effectidor_samples/infer_mixed_classifiers_on_embeddings.py` | Runs the cascade on real (unlabeled) effectidor sample FASTA files; records inference time only |
| `runtime_optimization/src/inference/effectidor_samples/combine_results.py` | Aggregates per-sample timing CSVs into `results/aggregated_elapsed_times.csv` and a bar plot |
| `runtime_optimization/src/pt5_layers_expirement/compare_results.py` | Compares classifier results across truncated-ProtT5 encoder layers |

`runtime_optimization` imports its ESM/ProtT5 embedding calculation and classic-ML classifier training/testing from `common/` (see Repository Layout above) rather than from `effectidor2_paper/` — this is what lets it reuse that code against its own data/output dirs without forking the logic or creating a reverse dependency on the paper's track.

**SLURM scripts** (`runtime_optimization/src/inference/effectidor_samples/infer_mixed_1.sh` – `infer_mixed_4.sh`): submit jobs for effectidor sample sets 1–4, each sweeping all threshold values (0, 0.1, 0.2, 0.3, 0.4, 1.0).

**Output directories:**
- `runtime_optimization/outputs/pretrained_embeddings/{esm_6,pt5}/` — this track's own cached embeddings (`EMBEDDINGS_DIR` in `runtime_optimization/src/utils/consts_paths.py`)
- `runtime_optimization/outputs/embeddings_classifiers/{esm_6,pt5,pt5_layers_expirement}/` — this track's own per-model classifier results (`CLASSIFIERS_OUTPUT_DIR`), kept separate from the paper's `effectidor2_paper/results/embeddings_classifiers/`
- `runtime_optimization/outputs/mixed_models/` — cascade sweep results
- `runtime_optimization/src/inference/effectidor_samples/results/<sample_id>/mixed_<threshold>/` — per-sample timing results from real effectidor runs
