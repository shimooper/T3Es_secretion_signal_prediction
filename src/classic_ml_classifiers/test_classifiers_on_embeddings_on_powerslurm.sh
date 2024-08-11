#!/bin/bash
#SBATCH --job-name=test_classifiers             # Job name
#SBATCH --account=power-general-users          # Account name for billing
#SBATCH --partition=power-general              # Partition name
#SBATCH --time=12:00:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-cpu=20G              # Memory per CPU core
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_new/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/outputs_new/%j.err         # Separate file for standard error

export HOME=/groups/pupko/yairshimony

# Print some information about the job
echo "Starting my SLURM job"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Allocated CPUs: $SLURM_JOB_CPUS_PER_NODE"
echo "Cuda visible devices: $CUDA_VISIBLE_DEVICES"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

python ~/python_test/test_gpu/check_cuda_available.py

cd ~/secretion_signal_prediction/src/classic_ml_classifiers
python test_classifiers_on_embeddings.py --model_id esm_6
python test_classifiers_on_embeddings.py --model_id esm_12
python test_classifiers_on_embeddings.py --model_id esm_30
python test_classifiers_on_embeddings.py --model_id esm_33
python test_classifiers_on_embeddings.py --model_id esm_36
python test_classifiers_on_embeddings.py --model_id pt5