import os.path
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.utils.read_fasta_utils import read_train_data
from src.finetune_pretrained_models.pt5.training_utils import train_per_protein


def main():
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()
    train_df = pd.DataFrame({'sequence': train_sequences, 'label': train_labels})
    valid_df = pd.DataFrame({'sequence': validation_sequences, 'label': validation_labels})

    train_per_protein(train_df, valid_df, num_labels=2, batch=1, accum=8, epochs=10, seed=42, deepspeed=False)


if __name__ == "__main__":
    main()
