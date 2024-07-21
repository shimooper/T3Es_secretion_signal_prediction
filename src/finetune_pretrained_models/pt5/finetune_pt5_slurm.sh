#!/bin/bash

#SBATCH --job-name=finetune_pt5             # Job name
#SBATCH --account=gpu-talp-users          # Account name for billing
#SBATCH --partition=gpu-talp              # Partition name
#SBATCH --time=01:00:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --gres=gpu:1     # number of GPU's to use in the job
#SBATCH --mem-per-cpu=4G              # Memory per CPU core
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/outputs/prott5_finetune/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/outputs/prott5_finetune/%j.err         # Separate file for standard error

# Print some information about the job
echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"

# Load modules or software if required
source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

# Run your application, this could be anything from a custom script to standard applications
python ~/python_test/test_gpu/check_cuda_available.py
python ~/secretion_signal_prediction/src/finetune_pretrained_models/pt5/main.py
