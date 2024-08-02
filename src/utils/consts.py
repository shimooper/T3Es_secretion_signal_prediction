import os

LOCAL_RUN = True if os.name == 'nt' else False
PROJECT_BASE_DIR = r"/groups/pupko/yairshimony/secretion_signal_prediction" if not LOCAL_RUN else r"C:\repos\T3Es_secretion_signal_prediction"

MODEL_ID_TO_MODEL_NAME = {
    'esm_6': 'esm2_t6_8M_UR50D_01_08_2024',
    'esm_12': 'esm2_t12_35M_UR50D_01_08_2024',
    'esm_30': 'esm2_t30_150M_UR50D_01_08_2024',
    'esm_33': 'esm2_t33_650M_UR50D_01_08_2024',
    'esm_36': 'esm2_t36_3B_UR50D_01_08_2024',
    'protein_bert': 'protein_bert',
    'pt5': 'prot_t5_xl_uniref50_01_08_2024',
}

BATCH_SIZE = 8
RANDOM_STATE = 42

DATASETS_ORIGINAL_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'datasets_original')
DATASETS_FIXED_DIR = os.path.join(PROJECT_BASE_DIR, 'data', 'datasets_fixed')
OUTPUTS_DIR = os.path.join(PROJECT_BASE_DIR, 'outputs_new')
PRETRAINED_MODELS_DIR = os.path.join(PROJECT_BASE_DIR, 'models')

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
