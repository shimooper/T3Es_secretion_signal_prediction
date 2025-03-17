import os
import argparse
from timeit import default_timer as timer
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                              FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                              BATCH_SIZE, EMBEDDINGS_DIR, MODEL_ID_TO_MODEL_NAME)
from src.utils.read_fasta_utils import read_sequences_from_fasta_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The model id to use for the embeddings calculation', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings', action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file_with_huggingface_model(model_name, fasta_file_path, embeddings_file_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sequences = read_sequences_from_fasta_file(fasta_file_path)

    model.eval()
    tokenized_sequences = tokenizer(sequences, return_tensors="pt")

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_inputs = {key: value[i:i+BATCH_SIZE] for key, value in tokenized_sequences.items()}
            outputs = model(**batch_inputs)
            sequence_representation = outputs.last_hidden_state[:, 1: -1, :].mean(1)
            embeddings.append(sequence_representation)

    Xs = torch.cat(embeddings, dim=0).numpy()

    np.save(embeddings_file_path, Xs)
    return Xs


def calc_embeddings(model_id, split, always_calc_embeddings=False):
    if split == 'train':
        positive_fasta_file = FIXED_POSITIVE_TRAIN_FILE
        negative_fasta_file = FIXED_NEGATIVE_TRAIN_FILE
    elif split == 'test':
        positive_fasta_file = FIXED_POSITIVE_TEST_FILE
        negative_fasta_file = FIXED_NEGATIVE_TEST_FILE
    else:
        raise ValueError(f"split must be one of ['train', 'test'], got {split}")

    output_dir = Path(EMBEDDINGS_DIR) / model_id
    os.makedirs(output_dir, exist_ok=True)
    positive_embeddings_output_file_path = os.path.join(output_dir, f'{split}_positive_embeddings.npy')
    negative_embeddings_output_file_path = os.path.join(output_dir, f'{split}_negative_embeddings.npy')

    model_name = MODEL_ID_TO_MODEL_NAME[model_id]

    if not os.path.exists(positive_embeddings_output_file_path) or always_calc_embeddings:
        print(f"Calculating embeddings of {positive_fasta_file} into {positive_embeddings_output_file_path}")
        positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
            model_name, positive_fasta_file, positive_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {positive_fasta_file} in {positive_embeddings_output_file_path}")
        positive_embeddings = np.load(positive_embeddings_output_file_path)

    if not os.path.exists(negative_embeddings_output_file_path) or always_calc_embeddings:
        print(f"Calculating embeddings of {negative_fasta_file} into {negative_embeddings_output_file_path}")
        negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
            model_name, negative_fasta_file, negative_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {negative_fasta_file} in {negative_embeddings_output_file_path}")
        negative_embeddings = np.load(negative_embeddings_output_file_path)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_embeddings(args.model_id, args.split, args.always_calc_embeddings)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation using {args.esm_embeddings_calculation_mode}: {elapsed_time} seconds.")
