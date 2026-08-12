import argparse
from timeit import default_timer as timer
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.consts import BATCH_SIZE, MODEL_ID_TO_MODEL_NAME
from common.read_fasta_utils import read_sequences_from_fasta_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The model id to use for the embeddings calculation', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--positive_fasta_file', help='Path to the positive-class FASTA file', type=Path, required=True)
    parser.add_argument('--negative_fasta_file', help='Path to the negative-class FASTA file', type=Path, required=True)
    parser.add_argument('--embeddings_dir', help='Base directory to cache computed embeddings under', type=Path, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings', action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file_with_huggingface_model_esm(model_id, fasta_file_path, embeddings_file_path):
    model_name = MODEL_ID_TO_MODEL_NAME[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sequences = read_sequences_from_fasta_file(fasta_file_path)

    model.eval()
    tokenized_sequences = tokenizer(sequences, return_tensors="pt", padding=True)

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


def calc_esm_embeddings(model_id, split, positive_fasta_file, negative_fasta_file, embeddings_dir,
                        always_calc_embeddings=False):
    output_dir = embeddings_dir / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_embeddings_output_file_path = output_dir / f'{split}_positive_embeddings.npy'
    negative_embeddings_output_file_path = output_dir / f'{split}_negative_embeddings.npy'

    if not positive_embeddings_output_file_path.exists() or always_calc_embeddings:
        print(f"Calculating embeddings of {positive_fasta_file} into {positive_embeddings_output_file_path}")
        positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
            model_id, positive_fasta_file, positive_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {positive_fasta_file} in {positive_embeddings_output_file_path}")
        positive_embeddings = np.load(positive_embeddings_output_file_path)

    if not negative_embeddings_output_file_path.exists() or always_calc_embeddings:
        print(f"Calculating embeddings of {negative_fasta_file} into {negative_embeddings_output_file_path}")
        negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
            model_id, negative_fasta_file, negative_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {negative_fasta_file} in {negative_embeddings_output_file_path}")
        negative_embeddings = np.load(negative_embeddings_output_file_path)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_esm_embeddings(args.model_id, args.split, args.positive_fasta_file, args.negative_fasta_file,
                        args.embeddings_dir, args.always_calc_embeddings)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation: {elapsed_time} seconds.")
