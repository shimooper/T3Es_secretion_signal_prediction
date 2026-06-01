#!/bin/bash
#SBATCH --job-name=combine_results
#SBATCH --account=pupko-users_v2
#SBATCH --partition=pupko-pool
#SBATCH --qos=owner
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/src/classic_ml_classifiers/test_mixed_classifiers/%j.out
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/src/classic_ml_classifiers/test_mixed_classifiers/%j.err

export HOME=/groups/pupko/yairshimony

echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction/src/classic_ml_classifiers/test_mixed_classifiers
python combine_results.py
