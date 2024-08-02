#!/bin/bash

#SBATCH --job-name=esm_embeddings             # Job name
#SBATCH --account=gpu-talp-users          # Account name for billing
#SBATCH --partition=gpu-talp              # Partition name
#SBATCH --time=10:00:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --gres=gpu:1             # Number of GPU cores per task
#SBATCH --mem-per-cpu=4G              # Memory per CPU core
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_new/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_new/%j.err         # Separate file for standard error

# Print some information about the job
echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

# Load modules or software if required
source /groups/pupko/yairshimony/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

# Run your application, this could be anything from a custom script to standard applications
python /groups/pupko/yairshimony/python_test/test_gpu/check_cuda_available.py
python /groups/pupko/yairshimony/secretion_signal_prediction/src/pretrained_embeddings/calc_esm_embeddings.py --model_id esm_6 --split train
python /groups/pupko/yairshimony/secretion_signal_prediction/src/pretrained_embeddings/calc_esm_embeddings.py --model_id esm_12 --split train
python /groups/pupko/yairshimony/secretion_signal_prediction/src/pretrained_embeddings/calc_esm_embeddings.py --model_id esm_30 --split train
python /groups/pupko/yairshimony/secretion_signal_prediction/src/pretrained_embeddings/calc_esm_embeddings.py --model_id esm_33 --split train
python /groups/pupko/yairshimony/secretion_signal_prediction/src/pretrained_embeddings/calc_esm_embeddings.py --model_id esm_36 --split train
