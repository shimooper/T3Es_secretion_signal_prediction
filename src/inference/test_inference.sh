#!/bin/bash
#SBATCH --job-name=test_classifiers             # Job name
#SBATCH --account=gpu-research          # Account name for billing
#SBATCH --partition=killable              # Partition name
#SBATCH --time=06:40:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=32G              # Memory per CPU core
#SBATCH --output=/home/ai_center/ai_users/yairshimony/secretion_signal_prediction/infer_on_all_db/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/home/ai_center/ai_users/yairshimony/secretion_signal_prediction/infer_on_all_db/%j.err         # Separate file for standard error

export HOME=/home/ai_center/ai_users/yairshimony

# Print some information about the job
echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

source ~/miniconda/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

python ~/python_test/test_gpu/check_cuda_available.py

cd ~/secretion_signal_prediction/src/inference
python infer_db.py --output_dir /home/ai_center/ai_users/yairshimony/secretion_signal_prediction/infer_on_all_db

