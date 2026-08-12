from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[2]

PROTEIN_BERT_DIR = PAPER_DIR / 'protein_bert'
PROTEIN_BERT_MODEL_NAME = 'epoch_92400_sample_23500000.pkl'

FINETUNE_NUMBER_OF_EPOCHS = 10

DATASETS_DIR = PAPER_DIR / 'data'

RESULTS_DIR = PAPER_DIR / 'results'
EMBEDDINGS_DIR = RESULTS_DIR / 'pretrained_embeddings'
CLASSIFIERS_OUTPUT_DIR = RESULTS_DIR / 'embeddings_classifiers'
FINETUNED_MODELS_OUTPUT_DIR = RESULTS_DIR / 'finetuned_models'
FINAL_RESULTS = RESULTS_DIR

FIXED_POSITIVE_TRAIN_FILE = DATASETS_DIR / "positive_train_data.fasta"
FIXED_NEGATIVE_TRAIN_FILE = DATASETS_DIR / "negative_train_data.fasta"

FIXED_POSITIVE_TEST_FILE = DATASETS_DIR / "positive_test_data.fasta"
FIXED_NEGATIVE_TEST_FILE = DATASETS_DIR / "negative_test_data.fasta"
