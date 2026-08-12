import argparse
from timeit import default_timer as timer
import re
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from effectidor2_paper.src.utils.consts_paths import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                                                       FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                                                       EMBEDDINGS_DIR)
from common.consts import BATCH_SIZE, MODEL_ID_TO_MODEL_NAME
from effectidor2_paper.src.utils.read_fasta_utils import read_sequences_from_fasta_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The model id to use for the embeddings calculation', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings', action='store_true')
    parser.add_argument('--hidden_layer_number', help='Index of the encoder hidden layer to use (0=embedding layer, 1=first transformer block, etc.). Defaults to the last layer.', type=int, default=None)
    return parser.parse_args()


def calc_embeddings_of_fasta_file_with_huggingface_model_pt5(model_id, fasta_file_path, embeddings_file_path, hidden_layer_number=None):
    sequences = read_sequences_from_fasta_file(fasta_file_path)

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    model_name = MODEL_ID_TO_MODEL_NAME[model_id]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)

    if hidden_layer_number is not None:
        # Truncate the encoder so computation stops at the target layer.
        # hidden_layer_number=0 → embedding output only (no transformer blocks),
        # hidden_layer_number=k → output after the k-th transformer block.
        model.encoder.block = torch.nn.ModuleList(list(model.encoder.block)[:hidden_layer_number])
        # Skip the final layer norm: it is trained for the last layer and is wrong for intermediate ones.
        model.encoder.final_layer_norm = torch.nn.Identity()

    tokenized_sequences = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tokenized_sequences['input_ids']).to(device)
    attention_mask = torch.tensor(tokenized_sequences['attention_mask']).to(device)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_input_ids = input_ids[i:i+BATCH_SIZE]
            batch_attention_mask = attention_mask[i:i+BATCH_SIZE]
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            sequences_representation = outputs.last_hidden_state[:, :-1, :].mean(1)
            embeddings.extend(sequences_representation)

    Xs = torch.stack(embeddings).cpu().numpy()
    np.save(embeddings_file_path, Xs)
    return Xs


def calc_pt5_embeddings(model_id, split, always_calc_embeddings=False, hidden_layer_number=None):
    if split == 'train':
        positive_fasta_file = FIXED_POSITIVE_TRAIN_FILE
        negative_fasta_file = FIXED_NEGATIVE_TRAIN_FILE
    elif split == 'test':
        positive_fasta_file = FIXED_POSITIVE_TEST_FILE
        negative_fasta_file = FIXED_NEGATIVE_TEST_FILE
    else:
        raise ValueError(f"split must be one of ['train', 'test'], got {split}")

    output_dir = EMBEDDINGS_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_suffix = f'_layer{hidden_layer_number}' if hidden_layer_number is not None else ''
    positive_embeddings_output_file_path = output_dir / f'{split}_positive_embeddings{layer_suffix}.npy'
    negative_embeddings_output_file_path = output_dir / f'{split}_negative_embeddings{layer_suffix}.npy'

    if not positive_embeddings_output_file_path.exists() or always_calc_embeddings:
        print(f"Calculating embeddings of {positive_fasta_file} into {positive_embeddings_output_file_path}")
        positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model_pt5(
            model_id, positive_fasta_file, positive_embeddings_output_file_path, hidden_layer_number)
    else:
        print(f"Found embeddings of {positive_fasta_file} in {positive_embeddings_output_file_path}")
        positive_embeddings = np.load(positive_embeddings_output_file_path)

    if not negative_embeddings_output_file_path.exists() or always_calc_embeddings:
        print(f"Calculating embeddings of {negative_fasta_file} into {negative_embeddings_output_file_path}")
        negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model_pt5(
            model_id, negative_fasta_file, negative_embeddings_output_file_path, hidden_layer_number)
    else:
        print(f"Found embeddings of {negative_fasta_file} in {negative_embeddings_output_file_path}")
        negative_embeddings = np.load(negative_embeddings_output_file_path)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_pt5_embeddings(args.model_id, args.split, args.always_calc_embeddings, args.hidden_layer_number)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation: {elapsed_time} seconds.")
