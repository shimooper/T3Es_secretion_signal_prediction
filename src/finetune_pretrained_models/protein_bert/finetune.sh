#!/bin/bash

#SBATCH --job-name=finetune             # Job name
#SBATCH --account=gpu-research          # Account name for billing
#SBATCH --partition=gpu-ai              # Partition name
#SBATCH --time=12:00:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --gres=gpu:1     # number of GPU's to use in the job
#SBATCH --mem=20G              # Memory per CPU core
#SBATCH --output=/home/ai_center/ai_users/yairshimony/secretion_signal_prediction/outputs_new_data_after_revision/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/home/ai_center/ai_users/yairshimony/secretion_signal_prediction/outputs_new_data_after_revision/%j.err         # Separate file for standard error

export HOME=/home/ai_center/ai_users/yairshimony

# Print some information about the job
echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

# Load modules or software if required
source ~/miniconda/etc/profile.d/conda.sh
conda activate secretion_signal_proteinbert
export PATH=$CONDA_PREFIX/bin:$PATH

# Run your application, this could be anything from a custom script to standard applications
python ~/secretion_signal_prediction/src/finetune_pretrained_models/protein_bert/train_and_test.py
