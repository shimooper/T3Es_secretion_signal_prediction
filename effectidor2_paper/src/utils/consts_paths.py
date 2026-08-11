import os

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROTEIN_BERT_DIR = os.path.join(PAPER_DIR, 'protein_bert')
PROTEIN_BERT_MODEL_NAME = 'epoch_92400_sample_23500000.pkl'

FINETUNE_NUMBER_OF_EPOCHS = 10

DATASETS_DIR = os.path.join(PAPER_DIR, 'data', 'new_data_processed')

RESULTS_DIR = os.path.join(PAPER_DIR, 'results')
EMBEDDINGS_DIR = os.path.join(RESULTS_DIR, 'pretrained_embeddings')
CLASSIFIERS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'embeddings_classifiers')
FINETUNED_MODELS_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'finetuned_models')
FINAL_RESULTS = RESULTS_DIR

FIXED_POSITIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "positive_train_data.fasta")
FIXED_NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "negative_train_data.fasta")

FIXED_POSITIVE_TEST_FILE = os.path.join(DATASETS_DIR, "positive_test_data.fasta")
FIXED_NEGATIVE_TEST_FILE = os.path.join(DATASETS_DIR, "negative_test_data.fasta")
