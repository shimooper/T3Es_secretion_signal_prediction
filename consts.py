import os

LOCAL_RUN = True
PROJECT_BASE_DIR = r"/groups/pupko/yairshimony/secretion_signal_prediction" if not LOCAL_RUN else r"C:\repos\T3Es_secretion_signal_prediction"

DATASETS_ORIGINAL_DIR = os.path.join(PROJECT_BASE_DIR, 'datasets_original')
DATASETS_FIXED_DIR = os.path.join(PROJECT_BASE_DIR, 'datasets_fixed')
DATASETS_DIR = DATASETS_ORIGINAL_DIR

POSITIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "positive_train_data.fasta")
NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "negative_train_data.fasta")

POSITIVE_TEST_FILE = os.path.join(DATASETS_DIR, "positive_test_data.fasta")
NEGATIVE_TEST_FILE = os.path.join(DATASETS_DIR, "negative_test_data.fasta")

POSITIVE_XANTOMONAS_FILE = os.path.join(DATASETS_DIR, "positive_Xantomonas_data.fasta")
NEGATIVE_XANTOMONAS_FILE = os.path.join(DATASETS_DIR, "negative_Xanthomonas_data.fasta")

