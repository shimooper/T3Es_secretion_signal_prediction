import time
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .classifier import PT5_classification_model
from .data_utils import prepare_dataset
from ..huggingface_utils import compute_metrics
from src.utils.read_fasta_utils import read_test_data, read_train_data
from src.utils.consts import OUTPUTS_DIR


def load_model(filepath, num_labels=1, half_precision=False):
    # Creates a new PT5 model and loads the finetuned weights from a file

    # load a new model
    checkpoint, model, tokenizer = PT5_classification_model(num_labels=num_labels, half_precision=half_precision)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath)

    # Assign the non-frozen parameters to the corresponding parameters of the model
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data

    return tokenizer, model


def evaluate_model_on_dataset(model, tokenizer, sequences, labels, device):
    dataset = prepare_dataset(pd.DataFrame({'sequence': sequences, 'label': labels}), tokenizer)
    # make compatible with torch DataLoader
    dataset = dataset.with_format("torch", device=device)

    # Create a dataloader for the test dataset
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Make predictions on the test dataset
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # add batch results (logits) to predictions
            predictions += model(input_ids, attention_mask=attention_mask).logits.tolist()

    scores = compute_metrics((predictions, labels))
    return scores


def main():
    tokenizer, model = load_model("./PT5_GB1_finetuned.pth", num_labels=1, half_precision=False)

    # Set the device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()
    train_scores = evaluate_model_on_dataset(model, tokenizer, train_sequences, train_labels, device)
    validation_scores = evaluate_model_on_dataset(model, tokenizer, validation_sequences, validation_labels, device)

    start_test_time = time.time()
    test_sequences, test_labels = read_test_data('test')
    test_scores = evaluate_model_on_dataset(model, tokenizer, test_sequences, test_labels, device)
    test_elapsed_time = time.time() - start_test_time

    start_xantomonas_time = time.time()
    xantomonas_sequences, xantomonas_labels = read_test_data('xantomonas')
    xantomonas_scores = evaluate_model_on_dataset(model, tokenizer, xantomonas_sequences, xantomonas_labels, device)
    xantomonas_elapsed_time = time.time() - start_xantomonas_time

    all_scores = {
        'train_mcc': train_scores['matthews_correlation'],
        'train_auprc': train_scores['auprc'],
        'validation_mcc': validation_scores['matthews_correlation'],
        'validation_auprc': validation_scores['auprc'],
        'test_mcc': test_scores['matthews_correlation'],
        'test_auprc': test_scores['auprc'],
        'test_elapsed_time': test_elapsed_time,
        'xantomonas_mcc': xantomonas_scores['matthews_correlation'],
        'xantomonas_auprc': xantomonas_scores['auprc'],
        'xantomonas_elapsed_time': xantomonas_elapsed_time,
        'backbone': 'PT5',
    }

    results_df = pd.DataFrame(all_scores)
    results_df.to_csv(f"{OUTPUTS_DIR}/pt5_finetune/pt5_results.csv", index=False)


if __name__ == "__main__":
    main()
