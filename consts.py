import os

LOCAL_RUN = True if os.name == 'nt' else False
PROJECT_BASE_DIR = r"/groups/pupko/yairshimony/secretion_signal_prediction" if not LOCAL_RUN else r"C:\repos\T3Es_secretion_signal_prediction"

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

EMBEDDINGS_POSITIVE_TRAIN_DIR = os.path.join(OUTPUTS_DIR, "train_positive")
EMBEDDINGS_NEGATIVE_TRAIN_DIR = os.path.join(OUTPUTS_DIR, "train_negative")

EMBEDDINGS_POSITIVE_TEST_DIR = os.path.join(OUTPUTS_DIR, "test_positive")
EMBEDDINGS_NEGATIVE_TEST_DIR = os.path.join(OUTPUTS_DIR, "test_negative")

EMBEDDINGS_POSITIVE_XANTOMONAS_DIR = os.path.join(OUTPUTS_DIR, "xantomonas_positive")
EMBEDDINGS_NEGATIVE_XANTOMONAS_DIR = os.path.join(OUTPUTS_DIR, "xantomonas_negative")
