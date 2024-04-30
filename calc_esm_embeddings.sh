#!/bin/bash
#PBS -S /bin/bash
#PBS -q power-pupko
#PBS -l select=1:ncpus=20
#PBS -o /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -e /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -N esm_embeddings
#PBS -r y

hostname
echo $PBS_JOBID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate secretion_signal
export PATH=$CONDA_PREFIX/bin:$PATH

cd ~/secretion_signal_prediction
python calc_esm_embeddings.py --model_name facebook/esm2_t6_8M_UR50D --output_dir /groups/pupko/yairshimony/secretion_signal_prediction/outputs/esm_embeddings_trainer_api --split train --esm_embeddings_calculation_mode trainer_api --measure_time
python calc_esm_embeddings.py --model_name facebook/esm2_t6_8M_UR50D --output_dir /groups/pupko/yairshimony/secretion_signal_prediction/outputs/esm_embeddings_huggingface_model --split train --esm_embeddings_calculation_mode huggingface_model --measure_time
python calc_esm_embeddings.py --model_name facebook/esm2_t6_8M_UR50D --output_dir /groups/pupko/yairshimony/secretion_signal_prediction/outputs/esm_embeddings_native_script --split train --esm_embeddings_calculation_mode native_script --measure_time
