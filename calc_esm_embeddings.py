import subprocess
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from datasets import Dataset

from consts import (FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE,
                    FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE,
                    FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE)
from utils import read_fasta_file

ESM_SCRIPT_PATH = "/groups/pupko/yairshimony/esm/scripts/extract.py"


def calc_embeddings_of_fasta_file_with_script(model_name, fasta_file_path, embeddings_dir, always_calc_embeddings=False):
    if not os.path.exists(embeddings_dir) or len(os.listdir(embeddings_dir)) == 0 or always_calc_embeddings:
        cmd = f"python {ESM_SCRIPT_PATH} {model_name} {fasta_file_path} {embeddings_dir} --include mean"
        subprocess.run(cmd, shell=True, check=True)

    embeddings = []
    for file in os.listdir(embeddings_dir):
        embeddings_object = torch.load(os.path.join(embeddings_dir, file))
        embeddings.append(embeddings_object['mean_representations'][0])
    Xs = torch.stack(embeddings, dim=0).numpy()
    return Xs


def calc_embeddings_of_fasta_file_with_huggingface_model(model, tokenizer, fasta_file_path, embeddings_dir,
                                                         embeddings_file_path, esm_embeddings_calculation_mode, always_calc_embeddings=False):
    if not os.path.exists(embeddings_file_path) or always_calc_embeddings:
        sequences = read_fasta_file(fasta_file_path)

        if esm_embeddings_calculation_mode == 'huggingface_model':
            model.eval()
            embeddings = []
            with torch.no_grad():
                for sequence in sequences:
                    tokenized_sequence = tokenizer(sequence, return_tensors="pt")
                    outputs = model(**tokenized_sequence)
                    sequence_representation = outputs.last_hidden_state.squeeze()[1: -1, :].mean(0)
                    embeddings.append(sequence_representation)

            Xs = torch.stack(embeddings, dim=0).numpy()

        elif esm_embeddings_calculation_mode == 'trainer_api':
            tokenized = tokenizer(sequences)
            dataset = Dataset.from_dict(tokenized)
            training_args = TrainingArguments(embeddings_dir, per_device_eval_batch_size=8)
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

    if esm_embeddings_calculation_mode == 'native_script':
        positive_embeddings = calc_embeddings_of_fasta_file_with_script(model_name, positive_fasta_file,
            os.path.join(output_dir, f'{split}_positive_embeddings'), always_calc_embeddings)
        negative_embeddings = calc_embeddings_of_fasta_file_with_script(model_name, negative_fasta_file,
            os.path.join(output_dir, f'{split}_negative_embeddings'), always_calc_embeddings)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        positive_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(model, tokenizer,
            positive_fasta_file, output_dir, os.path.join(output_dir, f'{split}_positive_embeddings.npy'), esm_embeddings_calculation_mode, always_calc_embeddings)
        negative_embeddings = calc_embeddings_of_fasta_file_with_huggingface_model(model, tokenizer,
            negative_fasta_file, output_dir, os.path.join(output_dir, f'{split}_negative_embeddings.npy'), esm_embeddings_calculation_mode,  always_calc_embeddings)

    return positive_embeddings, negative_embeddings
