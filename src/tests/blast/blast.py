import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from sklearn.metrics import matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.read_fasta_utils import read_fasta_file
from src.utils.consts import FIXED_POSITIVE_TRAIN_FILE, FIXED_NEGATIVE_TRAIN_FILE, FIXED_POSITIVE_TEST_FILE, \
    FIXED_NEGATIVE_TEST_FILE, FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE

BLAST_OUTPUT_HEADER = ['query', 'subject', 'identity_percent', 'alignment_length', 'mismatches', 'gap_openings',
                        'query_start', 'query_end', 'subject_start', 'subject_end', 'evalue', 'bit_score']
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / 'outputs'


def evaluate_on_test_data(positive_test_fasta, negative_test_fasta, split, train_sequence_id_to_label):
    # Blast test sequences against the train sequences db
    subprocess.run(['blastp', '-query', positive_test_fasta, '-db', OUTPUT_DIR / 'all_train_sequences.fasta',
                    '-out', OUTPUT_DIR / f'positive_{split}_blast_results.tsv', '-outfmt', '6',
                    '-max_target_seqs', '1', '-evalue', '1000'])
    subprocess.run(['blastp', '-query', negative_test_fasta, '-db', OUTPUT_DIR / 'all_train_sequences.fasta',
                    '-out', OUTPUT_DIR / f'negative_{split}_blast_results.tsv', '-outfmt', '6',
                    '-max_target_seqs', '1', '-evalue', '1000'])

    positive_test_blast_df = pd.read_csv(OUTPUT_DIR / f'positive_{split}_blast_results.tsv', sep='\t',
                                         names=BLAST_OUTPUT_HEADER)
    negative_test_blast_df = pd.read_csv(OUTPUT_DIR / f'negative_{split}_blast_results.tsv', sep='\t',
                                         names=BLAST_OUTPUT_HEADER)
    positive_test_blast_df.to_csv(OUTPUT_DIR / f'positive_{split}_blast_results.csv', index=False)
    negative_test_blast_df.to_csv(OUTPUT_DIR / f'negative_{split}_blast_results.csv', index=False)

    # Get the best hit for each query
    positive_test_blast_df = positive_test_blast_df.groupby('query')['evalue'].min().reset_index()
    negative_test_blast_df = negative_test_blast_df.groupby('query')['evalue'].min().reset_index()

    # Get the predictions for the test sequences
    positive_test_blast_df['prediction'] = positive_test_blast_df['subject'].apply(
        lambda x: train_sequence_id_to_label[x])
    negative_test_blast_df['prediction'] = negative_test_blast_df['subject'].apply(
        lambda x: train_sequence_id_to_label[x])
    positive_test_blast_df['true_label'] = 1
    negative_test_blast_df['true_label'] = 0
    all_predictions = pd.concat([positive_test_blast_df, negative_test_blast_df], ignore_index=True)[
        ['query', 'true_label', 'prediction']]
    all_predictions.to_csv(OUTPUT_DIR / f'all_{split}_predictions.csv', index=False)

    # Calculate metrics
    test_mcc = matthews_corrcoef(all_predictions['true_label'], all_predictions['prediction'])
    print(f'{split} MCC: {test_mcc}')


def main():
    train_sequence_id_to_label = {}
    positive_train_sequences = read_fasta_file(FIXED_POSITIVE_TRAIN_FILE)
    for seq_id in positive_train_sequences:
        train_sequence_id_to_label[seq_id] = 1
    negative_train_sequences = read_fasta_file(FIXED_NEGATIVE_TRAIN_FILE)
    for seq_id in negative_train_sequences:
        train_sequence_id_to_label[seq_id] = 0

    # Write positive and negative train sequences to one file and create a blast database
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DIR / 'all_train_sequences.fasta', 'w') as f:
        for seq_id, seq in positive_train_sequences.items():
            f.write(f'>{seq_id}\n{seq}\n')
        for seq_id, seq in negative_train_sequences.items():
            f.write(f'>{seq_id}\n{seq}\n')

    subprocess.run(['makeblastdb', '-in', OUTPUT_DIR / 'all_train_sequences.fasta', '-dbtype', 'prot'])

    evaluate_on_test_data(FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE, 'test', train_sequence_id_to_label)
    evaluate_on_test_data(FIXED_POSITIVE_XANTOMONAS_FILE, FIXED_NEGATIVE_XANTOMONAS_FILE, 'xantomonas', train_sequence_id_to_label)
    print('Done!')


if __name__ == "__main__":
    main()
