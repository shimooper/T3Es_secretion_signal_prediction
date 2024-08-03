import os
from pathlib import Path

PROJECT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_ORIGINAL_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'datasets_original')
DATASETS_FIXED_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'datasets_fixed')
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_BASE_DIR, 'models')

OUTPUTS_DIR = os.path.join(PROJECT_BASE_DIR, 'outputs_new')
EMBEDDINGS_BASE_DIR = os.path.join(OUTPUTS_DIR, 'pretrained_embeddings')
CLASSIFIERS_OUTPUT_BASE_DIR = os.path.join(OUTPUTS_DIR, 'embeddings_classifiers')
CLASSIFIERS_TEST_OUTPUT_BASE_DIR = os.path.join(OUTPUTS_DIR, 'embeddings_classifiers_test')
USE_LOCAL_MODELS = True

MODEL_ID_TO_MODEL_NAME = {
    'esm_6': Path(PRETRAINED_MODELS_DIR) / 'esm2_t6_8M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t6_8M_UR50D',
    'esm_12': Path(PRETRAINED_MODELS_DIR) / 'esm2_t12_35M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t12_35M_UR50D',
    'esm_30': Path(PRETRAINED_MODELS_DIR) / 'esm2_t30_150M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t30_150M_UR50D',
    'esm_33': Path(PRETRAINED_MODELS_DIR) / 'esm2_t33_650M_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t33_650M_UR50D',
    'esm_36': Path(PRETRAINED_MODELS_DIR) / 'esm2_t36_3B_UR50D_01_08_2024' if USE_LOCAL_MODELS else 'facebook/esm2_t36_3B_UR50D',
    'protein_bert': Path(PRETRAINED_MODELS_DIR) / 'protein_bert',
    'pt5': Path(PRETRAINED_MODELS_DIR) / 'prot_t5_xl_uniref50_01_08_2024' if USE_LOCAL_MODELS else 'facebook/prot_t5_xl_uniref50',
}

BATCH_SIZE = 8
RANDOM_STATE = 42



ORIGINAL_POSITIVE_TRAIN_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "positive_train_data.fasta")
ORIGINAL_NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "negative_train_data.fasta")

ORIGINAL_POSITIVE_TEST_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "positive_test_data.fasta")
ORIGINAL_NEGATIVE_TEST_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "negative_test_data.fasta")

ORIGINAL_POSITIVE_XANTOMONAS_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "positive_Xantomonas_data.fasta")
ORIGINAL_NEGATIVE_XANTOMONAS_FILE = os.path.join(DATASETS_ORIGINAL_DIR, "negative_Xanthomonas_data.fasta")

FIXED_POSITIVE_TRAIN_FILE = os.path.join(DATASETS_FIXED_DIR, "positive_train_data.fasta")
FIXED_NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_FIXED_DIR, "negative_train_data.fasta")

FIXED_POSITIVE_TEST_FILE = os.path.join(DATASETS_FIXED_DIR, "positive_test_data.fasta")
FIXED_NEGATIVE_TEST_FILE = os.path.join(DATASETS_FIXED_DIR, "negative_test_data.fasta")

FIXED_POSITIVE_XANTOMONAS_FILE = os.path.join(DATASETS_FIXED_DIR, "positive_Xantomonas_data.fasta")
FIXED_NEGATIVE_XANTOMONAS_FILE = os.path.join(DATASETS_FIXED_DIR, "negative_Xanthomonas_data.fasta")
