import numpy as np
import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../protein_bert'))

from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from src.utils.consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                              FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                              FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE)
from src.utils.read_fasta_utils import read_sequences_from_fasta_file


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', help='The output directory to save the embeddings', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file(fasta_file_path, embeddings_output_file_path, always_calc_embeddings=False):
    if not os.path.exists(embeddings_output_file_path) or always_calc_embeddings:
        sequences = read_sequences_from_fasta_file(fasta_file_path)
        sequence_length = len(sequences[0]) + 2

        pretrained_model_generator, input_encoder = load_pretrained_model(validate_downloading=False)
        model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(sequence_length))
        encoded_x = input_encoder.encode_X(sequences, sequence_length)
        local_representations, global_representations = model.predict(encoded_x, batch_size=8)
        sequences_representation = local_representations[:, 1: -1, :].mean(1)
        np.save(embeddings_output_file_path, sequences_representation)

    Xs = np.load(embeddings_output_file_path)
    return Xs


def calc_embeddings(output_dir, split, always_calc_embeddings):
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

    positive_embeddings = calc_embeddings_of_fasta_file(positive_fasta_file, os.path.join(output_dir, f'{split}_positive_embeddings.npy'), always_calc_embeddings)
    negative_embeddings = calc_embeddings_of_fasta_file(negative_fasta_file, os.path.join(output_dir, f'{split}_negative_embeddings.npy'), always_calc_embeddings)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()
    calc_embeddings(args.output_dir, args.split, args.always_calc_embeddings)
