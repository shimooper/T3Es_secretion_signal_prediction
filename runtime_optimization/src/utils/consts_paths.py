import os

OPTIMIZATION_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_DIR = os.path.join(OPTIMIZATION_DIR, 'data', 'data_processing_29_12_25', '4__Final_Datasets')

OUTPUTS_DIR = os.path.join(OPTIMIZATION_DIR, 'outputs')
EMBEDDINGS_DIR = os.path.join(OUTPUTS_DIR, 'pretrained_embeddings')
CLASSIFIERS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'embeddings_classifiers')
FINETUNED_MODELS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'finetuned_models')
FINAL_RESULTS = os.path.join(OUTPUTS_DIR, 'final_results')

FIXED_POSITIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "positive_train_data.fasta")
FIXED_NEGATIVE_TRAIN_FILE = os.path.join(DATASETS_DIR, "negative_train_data.fasta")

FIXED_POSITIVE_TEST_FILE = os.path.join(DATASETS_DIR, "positive_test_data.fasta")
FIXED_NEGATIVE_TEST_FILE = os.path.join(DATASETS_DIR, "negative_test_data.fasta")
