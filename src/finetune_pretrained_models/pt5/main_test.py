import torch  # Important to import torch first - without this I get weird error when trying to import torch later
import time
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.finetune_pretrained_models.pt5.classifier import PT5_classification_model
from src.finetune_pretrained_models.pt5.data_utils import prepare_dataset
from src.finetune_pretrained_models.huggingface_utils import compute_metrics
from src.finetune_pretrained_models.pt5.training_utils import FINETUNED_WEIGHTS_FILE
from src.utils.read_fasta_utils import read_test_data, read_train_data
from src.utils.consts import FINETUNED_MODELS_OUTPUT_DIR, MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION


def load_model(model_id, filepath, half_precision=False):
    # Creates a new PT5 model and loads the finetuned weights from a file

    # load a new model
    model, tokenizer = PT5_classification_model(model_id, num_labels=2, half_precision=half_precision)

    # Load the non-frozen parameters from the saved file
    non_frozen_params = torch.load(filepath, map_location=torch.device('cpu'))

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
    probabilities = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # add batch results (logits) to predictions
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities += F.softmax(outputs.logits, dim=-1)

    scores = compute_metrics((torch.stack(probabilities), labels))
    return scores


def main():
    model_id = 'pt5'
    tokenizer, model = load_model(model_id, os.path.join(FINETUNED_MODELS_OUTPUT_DIR, model_id, FINETUNED_WEIGHTS_FILE), half_precision=False)

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

    all_scores = {
        'train_mcc': [train_scores['matthews_correlation']],
        'train_auprc': [train_scores['auprc']],
        'validation_mcc': [validation_scores['matthews_correlation']],
        'validation_auprc': [validation_scores['auprc']],
        'test_mcc': [test_scores['matthews_correlation']],
        'test_auprc': [test_scores['auprc']],
        'test_elapsed_time': [test_elapsed_time],
        'model_id': [model_id],
        'training_mode': ['finetune'],
        'number_of_parameters (millions)': MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[model_id]
    }

    results_df = pd.DataFrame(all_scores)
    output_path = Path(FINETUNED_MODELS_OUTPUT_DIR) / model_id / 'best_model_results.csv'
    print(f"Writing scores to {output_path}")
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
