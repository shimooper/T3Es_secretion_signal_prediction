#!/bin/bash
#SBATCH --job-name=pt5_layer_compare           # Job name
#SBATCH --account=pupko-users_v2               # Account name for billing
#SBATCH --partition=pupko-pool                 # Partition name
#SBATCH --qos=owner
#SBATCH --time=00:30:00                        # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                             # Number of tasks (processes)
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task
#SBATCH --mem=8G                               # Memory
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_optimization/embeddings_classifiers/pt5_layers_expirement/%j.out
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_optimization/embeddings_classifiers/pt5_layers_expirement/%j.err

export HOME=/groups/pupko/yairshimony

echo "Starting pt5 layers comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction/src/classic_ml_classifiers/pt5_layers_expirement
python compare_results.py

echo "Done"
