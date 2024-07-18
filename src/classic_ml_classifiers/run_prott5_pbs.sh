#!/bin/bash
#PBS -q power-pupko
#PBS -l select=1:ncpus=100
#PBS -o /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -e /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -N secretion_signal
#PBS -r y

hostname
echo $PBS_JOBID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction
python classification_with_classic_ML.py --model_id Rostlab/prot_t5_xl_uniref50 --output_dir prot_t5 --n_jobs 50
python classification_with_classic_ML.py --model_id Rostlab/prot_t5_xl_half_uniref50-enc --output_dir prot_t5_half --n_jobs 50