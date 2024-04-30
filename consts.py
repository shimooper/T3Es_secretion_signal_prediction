import os

LOCAL_RUN = True if os.name == 'nt' else False
PROJECT_BASE_DIR = r"/groups/pupko/yairshimony/secretion_signal_prediction" if not LOCAL_RUN else r"C:\repos\T3Es_secretion_signal_prediction"

ESM_MODEL_NUMBER_OF_LAYERS_TO_MODEL_NAME = {
    '6': 'facebook/esm2_t6_8M_UR50D',
    '12': 'facebook/esm2_t12_35M_UR50D',
    '30': 'facebook/esm2_t30_150M_UR50D',
    '33': 'facebook/esm2_t33_650M_UR50D',
    '36': 'facebook/esm2_t36_3B_UR50D',
    '48': 'facebook/esm2_t48_15B_UR50D'
}

DATASETS_ORIGINAL_DIR = os.path.join(PROJECT_BASE_DIR, 'datasets_original')
DATASETS_FIXED_DIR = os.path.join(PROJECT_BASE_DIR, 'datasets_fixed')
OUTPUTS_DIR = os.path.join(PROJECT_BASE_DIR, 'outputs')

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
