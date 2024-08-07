import numpy as np
import torch
import os
import re
import pandas as pd
import argparse
from Bio import SeqIO
import joblib
import time
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
import torch.multiprocessing as mp

BATCH_SIZE = 32
PRETRAINED_MODEL_DIR = r"/groups/pupko/yairshimony/secretion_signal_prediction/models/prot_t5_xl_uniref50_01_08_2024/"
TRAINED_CLASSIFIER_FILE = r"/groups/pupko/yairshimony/secretion_signal_prediction/outputs_classifier/model.pkl"


def get_embeddings_internal(model, tokenizer, sequences):
    tokenized_sequences = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tokenized_sequences['input_ids'])
    attention_mask = torch.tensor(tokenized_sequences['attention_mask'])

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_input_ids = input_ids[i:i + BATCH_SIZE]
            batch_attention_mask = attention_mask[i:i + BATCH_SIZE]
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            sequences_representation = outputs.last_hidden_state[:, :-1, :].mean(1)
            embeddings.append(sequences_representation)

    Xs = torch.cat(embeddings)
    return Xs


def get_embeddings(sequences, cpus=1):
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_DIR, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PRETRAINED_MODEL_DIR)
    model.eval()

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    if cpus == 1:
         Xs = get_embeddings_internal(model, tokenizer, sequences)
    else:
        sequences_chunks = np.array_split(sequences, cpus)

        # Multiprocessing requires the model to be shared among workers
        mp.set_start_method('spawn', force=True)  # 'spawn' is safe for PyTorch multiprocessing
        model.share_memory()  # Share model's memory among processes

        with mp.Pool(processes=cpus) as pool:
            embeddings = pool.starmap(get_embeddings_internal, [(model, tokenizer, chunk) for chunk in sequences_chunks])

        Xs = torch.cat(embeddings)

    return Xs


def read_fasta_file(file_path):
    sequences = {}
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences[seq_record.id] = str(seq_record.seq)
    df = pd.DataFrame(sequences.items(), columns=['id', 'sequence'])
    return df


def main(args):
    start_time = time.time()

    input_fasta_file = args.input_fasta_file
    output_file = args.output_file
    input_fasta_file_name = Path(input_fasta_file).stem

    if output_file is None:
        output_file = os.path.join(os.path.dirname(input_fasta_file), f'{input_fasta_file_name}_predictions.csv')

    print(f'Reading sequences from {input_fasta_file}...')
    df = read_fasta_file(input_fasta_file)
    print(f'Read {len(df)} sequences, time elapsed: {time.time() - start_time:.2f} seconds')

    print(f'Calculating embeddings for {len(df)} sequences using {args.cpus} CPUs...')
    embeddings = get_embeddings(df['sequence'], args.cpus)
    print(f'Embeddings calculated, time elapsed: {time.time() - start_time:.2f} seconds')

    print(f'Predicting probabilities using the trained classifier...')
    model = joblib.load(TRAINED_CLASSIFIER_FILE)
    predictions = model.predict_proba(embeddings)
    print(f'Predictions done, time elapsed: {time.time() - start_time:.2f} seconds')

    df['probability'] = predictions[:, 1]
    df.drop(columns=['sequence'], inplace=True)

    df.to_csv(output_file, index=False)

    print(f'Predictions saved to {output_file}')
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta_file', help='path to a fasta file', required=True)
    parser.add_argument('--output_file', help='path to the output file with predictions. By default, the output file '
                                              'will be saved in the same directory as the input file.', required=False)
    parser.add_argument('--cpus', help='The number of CPUs to use for the prediction', type=int, default=1)

    args = parser.parse_args()
    main(args)
