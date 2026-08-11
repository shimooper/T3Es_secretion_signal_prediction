#!/bin/bash
#SBATCH --job-name=test_classifiers             # Job name
#SBATCH --account=pupko-users_v2          # Account name for billing
#SBATCH --partition=pupko-pool              # Partition name
#SBATCH --qos=owner
#SBATCH --time=06:40:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=16G              # Memory per CPU core
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/src/inference/effectidor_samples/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/runtime_optimization/src/inference/effectidor_samples/%j.err         # Separate file for standard error

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

#python ~/python_test/test_gpu/check_cuda_available.py

cd ~/secretion_signal_prediction/runtime_optimization/src/inference/effectidor_samples

python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 1
python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 0.4
python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 0.3
python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 0.2
python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 0.1
python infer_mixed_classifiers_on_embeddings.py --fasta_path effectidor_samples/4/N_terminals_4.faa --accuracy_threshold 0
