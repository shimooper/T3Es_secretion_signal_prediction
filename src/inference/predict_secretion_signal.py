import torch
import os
import re
import pandas as pd
import argparse
from Bio import SeqIO
import joblib
import logging
import time
from tqdm import tqdm
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel


HOME_DIR = os.path.expanduser("~")

PRETRAINED_MODEL_DIR = {
    'pt5': f"{HOME_DIR}/secretion_signal_prediction/models/prot_t5_xl_uniref50_01_08_2024/",
    'esm33': f"{HOME_DIR}/secretion_signal_prediction/models/esm2_t33_650M_UR50D_01_08_2024/"
}

TRAINED_HEAD_DIR = {
    'pt5': f"{HOME_DIR}/secretion_signal_prediction/outputs_new/final_results/trained_pt5_head/model.pkl",
    'esm33': f"{HOME_DIR}/secretion_signal_prediction/outputs_new/final_results/trained_esm33_head/model.pkl"
}


def get_pt5_embeddings(sequences, batch_size=4):
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_DIR['pt5'], do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PRETRAINED_MODEL_DIR['pt5'])
    model.eval()

    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]

    tokenized_sequences = tokenizer(sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(tokenized_sequences['input_ids'])
    attention_mask = torch.tensor(tokenized_sequences['attention_mask'])

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            sequences_representation = outputs.last_hidden_state[:, :-1, :].mean(1)
            embeddings.append(sequences_representation)

    Xs = torch.cat(embeddings)

    return Xs


def get_esm33_embeddings(sequences, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_DIR['esm33'])
    model = AutoModel.from_pretrained(PRETRAINED_MODEL_DIR['esm33'])
    model.eval()

    tokenized_sequences = tokenizer(list(sequences), return_tensors="pt")

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch_inputs = {key: value[i:i + batch_size] for key, value in tokenized_sequences.items()}
            outputs = model(**batch_inputs)
            sequence_representation = outputs.last_hidden_state[:, 1: -1, :].mean(1)
            embeddings.append(sequence_representation)

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
    model_name = 'pt5' if args.use_large_model else 'esm33'

    if output_file is None:
        output_file = os.path.join(os.path.dirname(input_fasta_file), f'{input_fasta_file_name}_predictions.csv')

    log_path = os.path.join(os.path.dirname(output_file), f'{input_fasta_file_name}_model_{model_name}_batch_{args.batch_size}_log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path)])
    logger = logging.getLogger(__name__)

    logger.info(f'Reading sequences from {input_fasta_file}...')
    df = read_fasta_file(input_fasta_file)
    logger.info(f'Read {len(df)} sequences, time elapsed: {time.time() - start_time:.2f} seconds')

    logger.info(f'Calculating embeddings for {len(df)} sequences using model {model_name} and batch size {args.batch_size}...')

    if model_name == 'pt5':
        embeddings = get_pt5_embeddings(df['sequence'], args.batch_size)
    else:
        embeddings = get_esm33_embeddings(df['sequence'], args.batch_size)
    logger.info(f'Embeddings calculated, time elapsed: {time.time() - start_time:.2f} seconds')

    logger.info(f'Predicting probabilities using the trained classifier...')
    model = joblib.load(TRAINED_HEAD_DIR[model_name])
    predictions = model.predict_proba(embeddings)
    logger.info(f'Predictions done, time elapsed: {time.time() - start_time:.2f} seconds')

    df['probability'] = predictions[:, 1]
    df.drop(columns=['sequence'], inplace=True)

    df.to_csv(output_file, index=False)

    logger.info(f'Predictions saved to {output_file}')
    logger.info(f'Elapsed time: {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta_file', help='path to a fasta file', required=True)
    parser.add_argument('--output_file', help='path to the output file with predictions. By default, the output file '
                                              'will be saved in the same directory as the input file.', required=False)
    parser.add_argument('--batch_size', help='The batch size to use for the prediction', type=int, default=16)
    parser.add_argument('--use_large_model', help='Whether to use the large model (prot_t5_xl_uniref50_01_08_2024) or '
                                                  'the small model (esm2_t33_650M_UR50D_01_08_2024). By default, use'
                                                  'the small model.', action='store_true')

    args = parser.parse_args()
    main(args)
