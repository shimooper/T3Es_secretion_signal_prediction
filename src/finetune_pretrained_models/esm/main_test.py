import torch  # Important to import torch first - without this I get weird error when trying to import torch later
import os
import argparse
import time
import sys

import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.consts import MODEL_ID_TO_MODEL_NAME, BATCH_SIZE, FINETUNED_MODELS_OUTPUT_DIR, MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION
from src.utils.read_fasta_utils import read_train_data, read_test_data
from src.finetune_pretrained_models.huggingface_utils import compute_metrics


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id (number of layers)', type=str, required=True)
    return parser.parse_args()


def create_dataset(tokenizer, sequences, labels=None):
    tokenized = tokenizer(sequences)
    dataset = Dataset.from_dict(tokenized)

    if labels:
        dataset = dataset.add_column("labels", labels)

    return dataset


def evaluate_model_on_dataset(model, tokenizer, sequences, labels, device):
    dataset = create_dataset(tokenizer, sequences, labels)
    # make compatible with torch DataLoader
    dataset = dataset.with_format("torch", device=device)

    # Create a dataloader for the test dataset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Make predictions on the test dataset
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # add batch results (logits) to predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities += F.softmax(outputs.logits, dim=-1)

    scores = compute_metrics((torch.stack(probabilities), labels))
    return scores


def main(model_id):
    model_name = MODEL_ID_TO_MODEL_NAME[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(Path(FINETUNED_MODELS_OUTPUT_DIR) / model_id / "best_model", num_labels=2)

    # Set the device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()
    print("Evaluating model on train data...")
    train_scores = evaluate_model_on_dataset(model, tokenizer, train_sequences, train_labels, device)
    print("Evaluating model on validation data...")
    validation_scores = evaluate_model_on_dataset(model, tokenizer, validation_sequences, validation_labels, device)

    start_test_time = time.time()
    test_sequences, test_labels = read_test_data('test')
    print("Evaluating model on test data...")
    test_scores = evaluate_model_on_dataset(model, tokenizer, test_sequences, test_labels, device)
    test_elapsed_time = time.time() - start_test_time

    start_xantomonas_time = time.time()
    xantomonas_sequences, xantomonas_labels = read_test_data('xantomonas')
    print("Evaluating model on xantomonas data...")
    xantomonas_scores = evaluate_model_on_dataset(model, tokenizer, xantomonas_sequences, xantomonas_labels, device)
    xantomonas_elapsed_time = time.time() - start_xantomonas_time

    # Save the results
    all_scores = {
        'train_mcc': [train_scores['matthews_correlation']],
        'train_auprc': [train_scores['auprc']],
        'validation_mcc': [validation_scores['matthews_correlation']],
        'validation_auprc': [validation_scores['auprc']],
        'test_mcc': [test_scores['matthews_correlation']],
        'test_auprc': [test_scores['auprc']],
        'test_elapsed_time': [test_elapsed_time],
        'xantomonas_mcc': [xantomonas_scores['matthews_correlation']],
        'xantomonas_auprc': [xantomonas_scores['auprc']],
        'xantomonas_elapsed_time': [xantomonas_elapsed_time],
        'model_id': [model_id],
        'training_mode': ['finetune'],
        'number_of_parameters (millions)': MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[model_id]
    }

    results_df = pd.DataFrame(all_scores)
    output_path = Path(FINETUNED_MODELS_OUTPUT_DIR) / model_id / 'best_model_results.csv'
    print(f"Writing scores to {output_path}")
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id)
