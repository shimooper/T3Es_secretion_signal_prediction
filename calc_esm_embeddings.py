import subprocess
import os
import argparse
from timeit import default_timer as timer

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset

from consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                    FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                    FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE,
                    BATCH_SIZE)
from utils import read_fasta_file, read_sequences_from_fasta_file

ESM_SCRIPT_PATH = "/groups/pupko/yairshimony/esm/scripts/extract.py"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='The model name to use for the embeddings calculation', type=str, required=True)
    parser.add_argument('--output_dir', help='The output directory to save the embeddings', type=str, required=True)
    parser.add_argument('--split', help='The split to calc the embeddings for', type=str, required=True)
    parser.add_argument('--esm_embeddings_calculation_mode', help='The mode to use for the embeddings calculation', type=str,
                        choices=['native_script', 'huggingface_model', 'trainer_api'], required=True)
    parser.add_argument('--always_calc_embeddings', help='Whether to always calc the embeddings even if they were already calculated', action='store_true')
    parser.add_argument('--measure_time', help='Whether to measure the time it takes to calc the embeddings', action='store_true')
    return parser.parse_args()


def calc_embeddings_of_fasta_file_with_script(model_name, fasta_file_path, embeddings_dir, embeddings_file_path, always_calc_embeddings=False):
    if not os.path.exists(embeddings_file_path) or always_calc_embeddings:
        # command is taken from: https://github.com/facebookresearch/esm
        cmd = f"python {ESM_SCRIPT_PATH} {model_name.split('/')[1]} {fasta_file_path} {embeddings_dir} --include mean"
        subprocess.run(cmd, shell=True, check=True)

        id_to_sequence = read_fasta_file(fasta_file_path)
        embeddings = []
        for record_id in id_to_sequence:
            sequence_embedding_file_path = os.path.join(embeddings_dir, f"{record_id}.pt")
            if not os.path.exists(sequence_embedding_file_path):
                raise ValueError(f"Couldn't find the embedding file for the sequence with id {record_id}")
            embeddings_object = torch.load(sequence_embedding_file_path)
            sequence_representation = next(iter(embeddings_object['mean_representations'].values()))  # take the first and only layer representation which is the last model layer
            embeddings.append(sequence_representation)
        Xs = torch.stack(embeddings, dim=0).numpy()
        np.save(embeddings_file_path, Xs)

    Xs = np.load(embeddings_file_path)
    return Xs


def calc_embeddings_of_fasta_file_with_huggingface_model(model, tokenizer, fasta_file_path, embeddings_dir,
                                                         embeddings_file_path, esm_embeddings_calculation_mode, always_calc_embeddings=False):
    if not os.path.exists(embeddings_file_path) or always_calc_embeddings:
        sequences = read_sequences_from_fasta_file(fasta_file_path)

        if esm_embeddings_calculation_mode == 'huggingface_model':
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

        # This way to run inference and get sequence embeddings isn't intuitive, but I saw this somewhere as a fast way to get embeddings.
        elif esm_embeddings_calculation_mode == 'trainer_api':
            tokenized_sequences = tokenizer(sequences)
            dataset = Dataset.from_dict(tokenized_sequences)
            training_args = TrainingArguments(
                embeddings_dir,  # In this context this argument doesn't have any meaning, but it's a required argument
                per_device_eval_batch_size=BATCH_SIZE)
            trainer = Trainer(model=model, args=training_args)
            outputs = trainer.predict(dataset)
            tokens_representations = outputs.predictions[0]
            Xs = tokens_representations[:, 1: -1, :].mean(1)

        else:
            raise ValueError(f"esm_embeddings_calculation_mode must be one of ['huggingface_model', 'trainer_api'], "
                             f"got {esm_embeddings_calculation_mode}")

        np.save(embeddings_file_path, Xs)

    Xs = np.load(embeddings_file_path)
    return Xs


def calc_embeddings(model_name, output_dir, split, esm_embeddings_calculation_mode, always_calc_embeddings):
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
    positive_embeddings_tmp_dir = os.path.join(output_dir, f'{split}_positive_embeddings')
    negative_embeddings_tmp_dir = os.path.join(output_dir, f'{split}_negative_embeddings')
    positive_embeddings_output_file_path = os.path.join(output_dir, f'{split}_positive_embeddings.npy')
    negative_embeddings_output_file_path = os.path.join(output_dir, f'{split}_negative_embeddings.npy')

    if esm_embeddings_calculation_mode == 'native_script':
        positive_embeddings = calc_embeddings_of_fasta_file_with_script(
            model_name, positive_fasta_file, positive_embeddings_tmp_dir, positive_embeddings_output_file_path,
            always_calc_embeddings)
        negative_embeddings = calc_embeddings_of_fasta_file_with_script(
            model_name, negative_fasta_file, negative_embeddings_tmp_dir, negative_embeddings_output_file_path,
            always_calc_embeddings)

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
            model, tokenizer, positive_fasta_file, positive_embeddings_tmp_dir, positive_embeddings_output_file_path,
            esm_embeddings_calculation_mode, always_calc_embeddings)
        negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(
            model, tokenizer, negative_fasta_file, negative_embeddings_tmp_dir, negative_embeddings_output_file_path,
            esm_embeddings_calculation_mode,  always_calc_embeddings)

    return positive_embeddings, negative_embeddings


if __name__ == "__main__":
    args = get_arguments()

    start_test_time = timer()
    calc_embeddings(args.model_name, args.output_dir, args.split, args.esm_embeddings_calculation_mode,
                    args.always_calc_embeddings)

    if args.measure_time:
        end_test_time = timer()
        elapsed_time = end_test_time - start_test_time
        print(f"Time took for embedding calculation using {args.esm_embeddings_calculation_mode}: {elapsed_time} seconds.")
