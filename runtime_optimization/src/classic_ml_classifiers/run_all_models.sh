#!/bin/bash
#SBATCH --job-name=all_models_classifiers      # Job name
#SBATCH --account=pupko-users_v2               # Account name for billing
#SBATCH --partition=pupko-pool                 # Partition name
#SBATCH --qos=owner
#SBATCH --time=48:00:00                        # Time allotted per model job (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (processes)
#SBATCH --cpus-per-task=10                     # CPU cores for embeddings + classifier grid search
#SBATCH --mem=64G                              # Memory
#SBATCH --array=0-5                            # One job per model (see MODEL_IDS below)
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/outputs/embeddings_classifiers/%A_%a.out
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/outputs/embeddings_classifiers/%A_%a.err

# Trains and tests the classic-ML classifiers on embeddings of each ESM model and ProtT5 (CPU only).
# The --output/--error directory must exist before submitting:
#   mkdir -p ~/secretion_signal_prediction/runtime_optimization/outputs/embeddings_classifiers
# Always recomputes everything: the cached (last-layer) embeddings of this model are removed below before running.

export HOME=/groups/pupko/yairshimony

MODEL_IDS=(esm_6 esm_12 esm_30 esm_33 esm_36 pt5)
MODEL_ID=${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting classifiers on embeddings for model: $MODEL_ID"
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction/runtime_optimization/src/classic_ml_classifiers

# Remove cached last-layer embeddings so they are recomputed (files with a _layerN suffix are left alone)
EMBEDDINGS_CACHE=~/secretion_signal_prediction/runtime_optimization/outputs/pretrained_embeddings/$MODEL_ID
rm -f "$EMBEDDINGS_CACHE"/{train,test}_{positive,negative}_embeddings.npy

echo "--- Training $MODEL_ID ---"
python train_classifiers_on_embeddings.py \
    --model_id $MODEL_ID \
    --n_jobs $SLURM_JOB_CPUS_PER_NODE \
    --device cpu

echo "--- Testing $MODEL_ID ---"
python test_classifiers_on_embeddings.py \
    --model_id $MODEL_ID \
    --device cpu

echo "Done with $MODEL_ID"
