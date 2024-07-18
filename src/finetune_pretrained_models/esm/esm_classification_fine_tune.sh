#!/bin/bash
#PBS -S /bin/bash
#PBS -q power-pupko
#PBS -l select=1:ncpus=20
#PBS -o /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -e /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -N esm
#PBS -r y

hostname
echo $PBS_JOBID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction
python esm_classification_fine_tune.py