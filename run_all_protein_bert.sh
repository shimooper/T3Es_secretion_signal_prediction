#!/bin/bash
#PBS -q power-pupko
#PBS -l select=1:ncpus=10
#PBS -o /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -e /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -N secretion_signal
#PBS -r y

hostname
echo $PBS_JOBID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate python38_tensorflow
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction
python run_all_protein_bert.py --cpus 10