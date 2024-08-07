#!/bin/bash
#SBATCH --job-name=predict             # Job name
#SBATCH --account=power-general-users          # Account name for billing
#SBATCH --partition=power-general              # Partition name
#SBATCH --time=01:40:00               # Time allotted for the job (hh:mm:ss)
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem-per-cpu=20G              # Memory per CPU core
#SBATCH --output=/groups/pupko/yairshimony/secretion_signal_prediction/src/tests/test_inference/%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=/groups/pupko/yairshimony/secretion_signal_prediction/src/tests/test_inference/%j.err         # Separate file for standard error

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

#python ~/secretion_signal_prediction/src/inference/predict_secretion_signal.py --input_fasta_file ~/secretion_signal_prediction/src/tests/test_inference/negative_Xanthomonas_data.fasta --output_file ~/secretion_signal_prediction/src/tests/test_inference/predictions_10_cpus.csv --cpus 10
python ~/secretion_signal_prediction/src/inference/predict_secretion_signal.py --input_fasta_file ~/secretion_signal_prediction/src/tests/test_inference/negative_Xanthomonas_data.fasta