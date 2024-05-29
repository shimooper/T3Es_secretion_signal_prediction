import subprocess
import os
import argparse
from timeit import default_timer as timer
import re

import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel

from consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                    FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                    FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE,
                    BATCH_SIZE)
from utils import read_sequences_from_fasta_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='The model name to use for the embeddings calculation', type=str, required=True)
    parser.add_argument('--output_dir', help='The output directory to save the embeddings', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings', action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file_with_huggingface_model(model, tokenizer, device, fasta_file_path,
                                                         embeddings_file_path, always_calc_embeddings=False):
    if not os.path.exists(embeddings_file_path) or always_calc_embeddings:
        sequences = read_sequences_from_fasta_file(fasta_file_path)

        # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

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

        Xs = torch.stack(embeddings).numpy()
        np.save(embeddings_file_path, Xs)

    Xs = np.load(embeddings_file_path)
    return Xs


def calc_embeddings(model_name, output_dir, split, always_calc_embeddings):
    if split == 'train':
        positive_fasta_file = FIXED_POSITIVE_TRAIN_FILE
        negative_fasta_file = FIXED_NEGATIVE_TRAIN_FILE
    elif split == 'test':
        positive_fasta_file = FIXED_POSITIVE_TEST_FILE
        negative_fasta_file = FIXED_NEGATIVE_TEST_FILE
    elif split == 'xantomonas':
        positive_fasta_file = FIXED_POSITIVE_XANTOMONAS_FILE
        negative_fasta_file = FIXED_NEGATIVE_XANTOMONAS_FILE
    else:
        raise ValueError(f"split must be one of ['train', 'test', 'xantomonas'], got {split}")

    os.makedirs(output_dir, exist_ok=True)
    positive_embeddings_output_file_path = os.path.join(output_dir, f'{split}_positive_embeddings.npy')
    negative_embeddings_output_file_path = os.path.join(output_dir, f'{split}_negative_embeddings.npy')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)

    # Load the model
    model = T5EncoderModel.from_pretrained(model_name).to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device == torch.device("cpu"):
        model.to(torch.float32)

    positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
        model, tokenizer, device, positive_fasta_file, positive_embeddings_output_file_path, always_calc_embeddings)
    negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
        model, tokenizer, device, negative_fasta_file, negative_embeddings_output_file_path, always_calc_embeddings)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_embeddings(args.model_name, args.output_dir, args.split, args.always_calc_embeddings)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation: {elapsed_time} seconds.")
