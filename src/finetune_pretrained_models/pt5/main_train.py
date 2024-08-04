import torch
import os.path
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.read_fasta_utils import read_train_data
from src.finetune_pretrained_models.pt5.training_utils import train_per_protein, save_model, FINETUNED_WEIGHTS_FILE
from src.utils.consts import BATCH_SIZE, FINETUNED_MODELS_OUTPUT_DIR


def main():
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()
    train_df = pd.DataFrame({'sequence': train_sequences, 'label': train_labels})
    valid_df = pd.DataFrame({'sequence': validation_sequences, 'label': validation_labels})

    model_id = 'pt5'
    model = train_per_protein(model_id, train_df, valid_df, num_labels=2, batch=1, accum=BATCH_SIZE, epochs=10, seed=42, deepspeed=False)

    # Save model (only the finetuned parameters)
    save_model(model, os.path.join(FINETUNED_MODELS_OUTPUT_DIR, model_id, FINETUNED_WEIGHTS_FILE))


if __name__ == "__main__":
    main()
