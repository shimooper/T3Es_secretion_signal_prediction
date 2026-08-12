#!/bin/bash
#SBATCH --job-name=pt5_layer_exp               # Job name
#SBATCH --account=pupko-users_v2               # Account name for billing
#SBATCH --partition=pupko-pool                 # Partition name
#SBATCH --qos=owner
#SBATCH --time=12:00:00                        # Time allotted per layer job (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (processes)
#SBATCH --cpus-per-task=10                     # CPU cores for classifier grid search
#SBATCH --mem=64G                              # Memory
#SBATCH --array=1-24                           # One job per encoder layer (ProtT5 has 24 transformer blocks)
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/outputs/embeddings_classifiers/pt5_layers_expirement/%A_%a.out
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/outputs/embeddings_classifiers/pt5_layers_expirement/%A_%a.err

export HOME=/groups/pupko/yairshimony

echo "Starting pt5 layer experiment"
echo "Job ID: $SLURM_JOB_ID  Array task (layer): $SLURM_ARRAY_TASK_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

LAYER=$SLURM_ARRAY_TASK_ID

cd ~/secretion_signal_prediction/effectidor2_paper/src/classic_ml_classifiers

echo "--- Training layer $LAYER ---"
python train_classifiers_on_embeddings.py \
    --model_id pt5 \
    --hidden_layer_number $LAYER \
    --n_jobs $SLURM_JOB_CPUS_PER_NODE

echo "--- Testing layer $LAYER ---"
python test_classifiers_on_embeddings.py \
    --model_id pt5 \
    --hidden_layer_number $LAYER

echo "Done with layer $LAYER"
