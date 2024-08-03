import numpy as np
import os
import argparse
import sys
from timeit import default_timer as timer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                              FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                              FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE,
                              PROJECT_BASE_DIR, EMBEDDINGS_DIR, BATCH_SIZE, MODEL_ID_TO_MODEL_NAME, USE_LOCAL_MODELS)
from src.utils.read_fasta_utils import read_sequences_from_fasta_file

sys.path.append(os.path.join(PROJECT_BASE_DIR, 'protein_bert'))

from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

PROTEIN_BERT_MODEL_NAME = 'epoch_92400_sample_23500000.pkl'

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings',
                        action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file(fasta_file_path, embeddings_output_file_path):
    sequences = read_sequences_from_fasta_file(fasta_file_path)
    sequence_length = len(sequences[0]) + 2

    pretrained_model_dir = MODEL_ID_TO_MODEL_NAME['protein_bert']
    if USE_LOCAL_MODELS:
        pretrained_model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=pretrained_model_dir, local_model_dump_file_name=PROTEIN_BERT_MODEL_NAME)
    else:
        pretrained_model_generator, input_encoder = load_pretrained_model(validate_downloading=False)
    model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(sequence_length))
    encoded_x = input_encoder.encode_X(sequences, sequence_length)
    local_representations, global_representations = model.predict(encoded_x, batch_size=BATCH_SIZE)
    sequences_representation = local_representations[:, 1: -1, :].mean(1)
    np.save(embeddings_output_file_path, sequences_representation)
    return sequences_representation


def calc_embeddings(split, always_calc_embeddings):
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

    output_dir = Path(EMBEDDINGS_DIR) / 'protein_bert'
    os.makedirs(output_dir, exist_ok=True)
    positive_embeddings_output_file_path = os.path.join(output_dir, f'{split}_positive_embeddings.npy')
    negative_embeddings_output_file_path = os.path.join(output_dir, f'{split}_negative_embeddings.npy')

    if not os.path.exists(positive_embeddings_output_file_path) or always_calc_embeddings:
        print(f"Calculating embeddings of {positive_fasta_file} into {positive_embeddings_output_file_path}")
        positive_embeddings = calc_embeddings_of_fasta_file(positive_fasta_file, positive_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {positive_fasta_file} in {positive_embeddings_output_file_path}")
        positive_embeddings = np.load(positive_embeddings_output_file_path)

    if not os.path.exists(negative_embeddings_output_file_path) or always_calc_embeddings:
        print(f"Calculating embeddings of {negative_fasta_file} into {negative_embeddings_output_file_path}")
        negative_embeddings = calc_embeddings_of_fasta_file(negative_fasta_file, negative_embeddings_output_file_path)
    else:
        print(f"Found embeddings of {negative_fasta_file} in {negative_embeddings_output_file_path}")
        negative_embeddings = np.load(negative_embeddings_output_file_path)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_embeddings(args.split, args.always_calc_embeddings)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation: {elapsed_time} seconds.")
