from pathlib import Path

OPTIMIZATION_DIR = Path(__file__).resolve().parents[2]

DATASETS_DIR = OPTIMIZATION_DIR / 'data' / 'data_processing_29_12_25' / '4__Final_Datasets'
OUTPUTS_DIR = OPTIMIZATION_DIR / 'outputs'
EMBEDDINGS_DIR = OUTPUTS_DIR / 'pretrained_embeddings'
MIXED_MODELS_DIR = OUTPUTS_DIR / 'mixed_models' / 'two_thresholds'

POSITIVE_TRAIN_FILE = DATASETS_DIR / "positive_train_data.fasta"
NEGATIVE_TRAIN_FILE = DATASETS_DIR / "negative_train_data.fasta"

POSITIVE_TEST_FILE = DATASETS_DIR / "positive_test_data.fasta"
NEGATIVE_TEST_FILE = DATASETS_DIR / "negative_test_data.fasta"
