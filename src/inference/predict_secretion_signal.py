import os
import sys
import pandas as pd
import argparse
from Bio import SeqIO


def read_fasta_file(file_path):
    sequences = {}
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences[seq_record.id] = str(seq_record.seq)
    df = pd.DataFrame(sequences.items(), columns=['id', 'sequence'])
    return df


def main(args):
    input_fasta_file = args.input_fasta_file
    output_path = args.output_path
    model_dir = args.model_dir

    if output_path is None:
        output_path = os.path.join(os.path.dirname(input_fasta_file), 'predictions.csv')

    df = read_fasta_file(input_fasta_file)
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fasta_file', help='path to a fasta file', required=True)
    parser.add_argument('output_path', help='path to the output file with predictions. By default, the output file '
                                            'will be saved in the same directory as the input file.', required=False)
    parser.add_argument('model_dir', help='path to the directory with the trained model', required=True)

    args = parser.parse_args()
    main(args)
