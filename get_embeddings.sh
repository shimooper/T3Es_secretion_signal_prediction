#!/bin/bash
#PBS -l select=1:ncpus=1
##choose any gpu queue: gpu/gpu2
#PBS -q power-pupko
#PBS -o /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -e /groups/pupko/yairshimony/secretion_signal_prediction/outputs
#PBS -N esm

hostname
echo $PBS_JOBID

source ~/miniconda3/etc/profile.d/conda.sh
conda activate esm

python /groups/pupko/yairshimony/python_test/check_cuda_available.py

cd ~/secretion_signal_prediction

POSITIVE_TRAIN_FILE=$(python -c "from consts import FIXED_POSITIVE_TRAIN_FILE; print(FIXED_POSITIVE_TRAIN_FILE)")
echo $POSITIVE_TRAIN_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $POSITIVE_TRAIN_FILE ~/secretion_signal_prediction/outputs/train_positive/ --repr_layers 33 --include mean

POSITIVE_TEST_FILE=$(python -c "from consts import FIXED_POSITIVE_TEST_FILE; print(FIXED_POSITIVE_TEST_FILE)")
echo $POSITIVE_TEST_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $POSITIVE_TEST_FILE ~/secretion_signal_prediction/outputs/test_positive/ --repr_layers 33 --include mean

POSITIVE_XANTOMONAS_FILE=$(python -c "from consts import FIXED_POSITIVE_XANTOMONAS_FILE; print(FIXED_POSITIVE_XANTOMONAS_FILE)")
echo POSITIVE_XANTOMONAS_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $POSITIVE_XANTOMONAS_FILE ~/secretion_signal_prediction/outputs/xantomonas_positive/ --repr_layers 33 --include mean

NEGATIVE_TRAIN_FILE=$(python -c "from consts import FIXED_NEGATIVE_TRAIN_FILE; print(FIXED_NEGATIVE_TRAIN_FILE)")
echo $NEGATIVE_TRAIN_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $NEGATIVE_TRAIN_FILE ~/secretion_signal_prediction/outputs/train_negative/ --repr_layers 33 --include mean

NEGATIVE_TEST_FILE=$(python -c "from consts import FIXED_NEGATIVE_TEST_FILE; print(FIXED_NEGATIVE_TEST_FILE)")
echo $NEGATIVE_TEST_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $NEGATIVE_TEST_FILE ~/secretion_signal_prediction/outputs/test_negative/ --repr_layers 33 --include mean

NEGATIVE_XANTOMONAS_FILE=$(python -c "from consts import FIXED_NEGATIVE_XANTOMONAS_FILE; print(FIXED_NEGATIVE_XANTOMONAS_FILE)")
echo $NEGATIVE_XANTOMONAS_FILE
python ~/esm/scripts/extract.py esm2_t33_650M_UR50D $NEGATIVE_XANTOMONAS_FILE ~/secretion_signal_prediction/outputs/xantomonas_negative/ --repr_layers 33 --include mean
