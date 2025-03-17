from pathlib import Path
from Bio import SeqIO
import subprocess
import logging
import argparse
import pandas as pd
import random
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = SCRIPT_DIR / 'raw_data'

T3E_PATH = RAW_DATA_PATH / 'T3Es10_EDITED_NO_PARTIAL_v2.faa'
ECOLI_PATH = RAW_DATA_PATH / 'corrected_e_coli_k12.faa'

LOG_MESSAGE_FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'

MMSEQS_OUTPUT_FORMAT = 'query,target,fident,qcov,tcov,bits,evalue'
MMSEQS_OUTPUT_HEADER = MMSEQS_OUTPUT_FORMAT.split(',')


def setup_logger(outputs_dir):
    logger = logging.getLogger('main')
    file_handler = logging.FileHandler(outputs_dir / 'log.txt', mode='w')
    formatter = logging.Formatter(LOG_MESSAGE_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


def run_command(logger, command):
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_message = f'Error in command: "{e.cmd}": {e.stderr}'
        logger.exception(error_message)
        raise e


def cut_N_terminal(logger, records, output_path, record_id_prefix_to_add):
    for record in records:
        record.seq = record.seq[:100]
        record.id = f'{record_id_prefix_to_add}:{record.id}'

    SeqIO.write(records, output_path, 'fasta')
    logger.info(f'Wrote {len(records)} records to {output_path}.')


def cluster_records(logger, input_fasta_path, output_dir, output_fasta_path, min_e_value, min_coverage, min_identity):
    number_of_sequences = len(list(SeqIO.parse(input_fasta_path, 'fasta')))

    output_dir.mkdir(exist_ok=True, parents=True)
    cmd = f'mmseqs easy-cluster {input_fasta_path} {output_dir / "result"} {output_dir / "tmp_dir"} -e {min_e_value} ' \
          f'--min-seq-id {min_identity} -c {min_coverage} --threads 1'
    run_command(logger, cmd)
    shutil.copy(output_dir / 'result_rep_seq.fasta', output_fasta_path)

    number_of_representatives = len(list(SeqIO.parse(output_fasta_path, 'fasta')))
    logger.info(f'Wrote representatives of {input_fasta_path} to {output_fasta_path}. '
                f'Reduced from {number_of_sequences} to {number_of_representatives} sequences.')


def find_homologs(logger, query_fasta_path, target_fasta_path, output_fasta_path, tmp_dir_path, min_e_value,
                  min_coverage, min_identity):
    cmd = f'mmseqs easy-search {query_fasta_path} {target_fasta_path} {output_fasta_path} {tmp_dir_path} ' \
          f'-e {min_e_value} --min-seq-id {min_identity} -c {min_coverage} --cov-mode 0 --threads 1 ' \
          f'--format-output {MMSEQS_OUTPUT_FORMAT}'
    run_command(logger, cmd)

    homologs_df = pd.read_csv(output_fasta_path, sep='\t', names=MMSEQS_OUTPUT_HEADER)
    homologs_df.to_csv(output_fasta_path, index=False)


def create_negative_ecoli_dataset(logger, ecoli_fasta_path, t3e_fasta_path, output_dir, min_e_value, min_coverage,
                                  min_identity):
    find_homologs(logger, ecoli_fasta_path, t3e_fasta_path, output_dir / 'ecoli_homologs_te3.m8', output_dir / 'tmp',
                  min_e_value, min_coverage, min_identity)
    ecoli_homologs_t3e_df = pd.read_csv(output_dir / 'ecoli_homologs_te3.m8')
    ecoli_homologs_t3e_ids = set(ecoli_homologs_t3e_df['query'])

    ecoli_records = list(SeqIO.parse(ecoli_fasta_path, 'fasta'))
    negative_ecoli_records = [record for record in ecoli_records if record.id not in ecoli_homologs_t3e_ids]
    SeqIO.write(negative_ecoli_records, output_dir / 'negative_ecoli.faa', 'fasta')

    logger.info(f'Out of {len(ecoli_records)} E. coli sequences, {len(negative_ecoli_records)} were chosen as negative, '
                f'after removing {len(ecoli_homologs_t3e_ids)} homologs to T3Es.')


def split_fasta(logger, input_fasta, output_train_fasta, output_test_fasta, test_ratio):
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    random.shuffle(sequences)

    split_idx = int(len(sequences) * test_ratio)
    test_seqs = sequences[:split_idx]
    train_seqs = sequences[split_idx:]

    # Write training set
    with open(output_train_fasta, "w") as train_out:
        SeqIO.write(train_seqs, train_out, "fasta")

    # Write testing set
    with open(output_test_fasta, "w") as test_out:
        SeqIO.write(test_seqs, test_out, "fasta")

    logger.info(f"Split {len(sequences)} sequences from {input_fasta} into {len(train_seqs)} train and {len(test_seqs)} test sequences.")


def main():
    # Init: Parse arguments, and create outputs directory and logger, and read sequences.
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir_name', default='outputs', help='Directory name to save outputs')
    parser.add_argument('--min_e_value', default=1e-3, type=float, help='Minimum e-value for homologs/clustering')
    parser.add_argument('--min_coverage', default=0.6, type=float, help='Minimum coverage for homologs/clustering')
    parser.add_argument('--min_identity', default=0.5, type=float, help='Minimum identity for homologs/clustering')
    parser.add_argument('--test_ratio', default=0.2, type=float, help='Ratio of test data')
    args = parser.parse_args()

    outputs_dir = SCRIPT_DIR / args.outputs_dir_name
    outputs_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger(outputs_dir)

    logger.info('Reading sequences...')
    t3e_records = list(SeqIO.parse(T3E_PATH, 'fasta'))
    ecoli_records = list(SeqIO.parse(ECOLI_PATH, 'fasta'))
    logger.info(f'Done reading sequences. There are {len(t3e_records)} T3Es and {len(ecoli_records)} E. coli sequences')

    # Step 1: Cut N-terminal.
    logger.info('Step 1: Cut N-terminal...')
    n_terminal_dir = outputs_dir / '1__N_terminal'
    n_terminal_dir.mkdir(exist_ok=True, parents=True)
    cut_N_terminal(logger, t3e_records, n_terminal_dir / 't3e.faa', 't3e')
    cut_N_terminal(logger, ecoli_records, n_terminal_dir / 'ecoli.faa', 'ecoli')

    # Step 2: Create negative-ecoli dataset.
    logger.info('Step 2: Create negative-ecoli dataset...')
    negative_ecoli_dir = outputs_dir / '2__Negative_Ecoli'
    negative_ecoli_dir.mkdir(exist_ok=True, parents=True)
    create_negative_ecoli_dataset(logger, n_terminal_dir / 'ecoli.faa', n_terminal_dir / 't3e.faa', negative_ecoli_dir,
                                  args.min_e_value, args.min_coverage, args.min_identity)

    # Step 3: Cluster sequences and choose representatives.
    logger.info('Step 3: Cluster sequences and choose representatives...')
    representatives_dir = outputs_dir / '3__Representatives'
    representatives_dir.mkdir(exist_ok=True, parents=True)
    cluster_records(logger, n_terminal_dir / 't3e.faa', representatives_dir / 't3e',
                    representatives_dir / 't3e_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)
    cluster_records(logger, negative_ecoli_dir / 'negative_ecoli.faa', representatives_dir / 'ecoli',
                    representatives_dir / 'ecoli_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)

    # Step 4: Split to train and test.
    logger.info('Step 4: Split to train and test...')
    final_datasets_dir = outputs_dir / '4__Final_Datasets'
    final_datasets_dir.mkdir(exist_ok=True, parents=True)
    split_fasta(logger, representatives_dir / 't3e_representatives.faa', final_datasets_dir / 'positive_train_data.fasta',
                final_datasets_dir / 'positive_test_data.fasta', args.test_ratio)
    split_fasta(logger, representatives_dir / 'ecoli_representatives.faa', final_datasets_dir / 'negative_train_data.fasta',
                final_datasets_dir / 'negative_test_data.fasta', args.test_ratio)


if __name__ == '__main__':
    main()
