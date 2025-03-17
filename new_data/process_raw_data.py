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
XANTHOMONAS_PATH = RAW_DATA_PATH / 'Xcc8004.fasta'

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


def create_negative_xanthomonas_dataset(logger, xanthomonas_fasta_path, ecoli_fasta_path, te3_fasta_path, output_dir, min_e_value,
                                        min_coverage, min_identity):
    find_homologs(logger, xanthomonas_fasta_path, ecoli_fasta_path, output_dir / 'xanthomonas_homologs_ecoli.m8',
                  output_dir / 'tmp_homologs_to_ecoli', min_e_value, min_coverage, min_identity)
    xanthomonas_homologs_ecoli_df = pd.read_csv(output_dir / 'xanthomonas_homologs_ecoli.m8')
    xanthomonas_homologs_ecoli_ids = set(xanthomonas_homologs_ecoli_df['query'])

    # Removing homologs from Xanthomonas-negatives to T3Es is just to make sure, since there aren't supposed to be any.
    find_homologs(logger, xanthomonas_fasta_path, te3_fasta_path, output_dir / 'xanthomonas_homologs_te3.m8',
                  output_dir / 'tmp_homologs_to_te3', min_e_value, min_coverage, min_identity)
    xanthomonas_homologs_te3_df = pd.read_csv(output_dir / 'xanthomonas_homologs_te3.m8')
    xanthomonas_homologs_te3_ids = set(xanthomonas_homologs_te3_df['query'])

    xanthomonas_records = list(SeqIO.parse(xanthomonas_fasta_path, 'fasta'))
    negative_xanthomonas_records = [record for record in xanthomonas_records
                                    if record.id in xanthomonas_homologs_ecoli_ids and record.id not in xanthomonas_homologs_te3_ids]
    SeqIO.write(negative_xanthomonas_records, output_dir / 'negative_xanthomonas.faa', 'fasta')

    logger.info(f'Out of {len(xanthomonas_records)} Xanthomonas sequences, {len(negative_xanthomonas_records)} were chosen as negative, '
                f'after choosing {len(xanthomonas_homologs_ecoli_ids)} homologs to ecoli-negatives, and removing {len(xanthomonas_homologs_te3_ids)} homologs to T3Es.')


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
    xanthomonas_records = list(SeqIO.parse(XANTHOMONAS_PATH, 'fasta'))
    logger.info(f'Done reading sequences. There are {len(t3e_records)} T3Es, {len(ecoli_records)} E. coli sequences, '
                f'and {len(xanthomonas_records)} Xanthomonas sequences.')

    # validate_xanthomonas(logger, t3e_records)

    # Step 1: Cut N-terminal.
    logger.info('Step 1: Cut N-terminal...')
    n_terminal_dir = outputs_dir / '1__N_terminal'
    n_terminal_dir.mkdir(exist_ok=True, parents=True)
    cut_N_terminal(logger, t3e_records, n_terminal_dir / 't3e.faa', 't3e')
    cut_N_terminal(logger, ecoli_records, n_terminal_dir / 'ecoli.faa', 'ecoli')
    cut_N_terminal(logger, xanthomonas_records, n_terminal_dir / 'xanthomonas.faa', 'xanthomonas')

    # Step 2: Create negative-ecoli dataset.
    logger.info('Step 2: Create negative-ecoli dataset...')
    negative_ecoli_dir = outputs_dir / '2__Negative_Ecoli'
    negative_ecoli_dir.mkdir(exist_ok=True, parents=True)
    create_negative_ecoli_dataset(logger, n_terminal_dir / 'ecoli.faa', n_terminal_dir / 't3e.faa', negative_ecoli_dir,
                                  args.min_e_value, args.min_coverage, args.min_identity)

    # Step 3: Create negative-xanthomonas dataset.
    logger.info('Step 3: Create negative-xanthomonas dataset...')
    negative_xanthomonas_dir = outputs_dir / '3__Negative_Xanthomonas'
    negative_xanthomonas_dir.mkdir(exist_ok=True, parents=True)
    create_negative_xanthomonas_dataset(logger, n_terminal_dir / 'xanthomonas.faa', negative_ecoli_dir / 'negative_ecoli.faa', n_terminal_dir / 't3e.faa',
                                        negative_xanthomonas_dir, args.min_e_value, args.min_coverage, args.min_identity)

    # Step 4: Cluster sequences and choose representatives.
    logger.info('Step 4: Cluster sequences and choose representatives...')
    representatives_dir = outputs_dir / '4__Representatives'
    representatives_dir.mkdir(exist_ok=True, parents=True)
    cluster_records(logger, n_terminal_dir / 't3e.faa', representatives_dir / 't3e',
                    representatives_dir / 't3e_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)
    cluster_records(logger, negative_ecoli_dir / 'negative_ecoli.faa', representatives_dir / 'ecoli',
                    representatives_dir / 'ecoli_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)
    cluster_records(logger, negative_xanthomonas_dir / 'negative_xanthomonas.faa', representatives_dir / 'xanthomonas',
                    representatives_dir / 'xanthomonas_representatives.faa',
                    args.min_e_value, args.min_coverage, args.min_identity)

    # Step 5: Separate t3e to xanthomonas and non-xanthomonas.
    logger.info('Step 5: Separate t3e to xanthomonas and non-xanthomonas...')
    separate_te3_dir = outputs_dir / '5__Separate_T3E'
    separate_te3_dir.mkdir(exist_ok=True, parents=True)
    t3e_representatives = list(SeqIO.parse(representatives_dir / 't3e_representatives.faa', 'fasta'))
    te3_xanthomonas_representatives = [record for record in t3e_representatives if 'Xanthomonas' in record.description]
    SeqIO.write(te3_xanthomonas_representatives, separate_te3_dir / 't3e_xanthomonas.faa', 'fasta')
    te3_non_xanthomonas_representatives = [record for record in t3e_representatives if 'Xanthomonas' not in record.description]
    SeqIO.write(te3_non_xanthomonas_representatives, separate_te3_dir / 't3e_non_xanthomonas.faa', 'fasta')
    logger.info(f'There are {len(te3_xanthomonas_representatives)} Xanthomonas T3Es and {len(te3_non_xanthomonas_representatives)} '
                f'non-Xanthomonas T3Es.')

    # Step 6: Aggregate files.
    logger.info('Step 6: Aggregate files...')
    all_data_dir = outputs_dir / '6__All_Data'
    all_data_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(negative_ecoli_dir / 'negative_ecoli.faa', all_data_dir / 'negative_ecoli.faa')
    shutil.copy(negative_xanthomonas_dir / 'negative_xanthomonas.faa', all_data_dir / 'negative_xanthomonas.faa')
    shutil.copy(separate_te3_dir / 't3e_xanthomonas.faa', all_data_dir / 't3e_xanthomonas.faa')
    shutil.copy(separate_te3_dir / 't3e_non_xanthomonas.faa', all_data_dir / 't3e_non_xanthomonas.faa')

    # Step 7: Split to train and test.
    logger.info('Step 7: Split to train and test...')
    final_datasets_dir = outputs_dir / '7__Final_Datasets'
    final_datasets_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(all_data_dir / 't3e_xanthomonas.faa', final_datasets_dir / 'positive_Xanthomonas_data.fasta')
    shutil.copy(all_data_dir / 'negative_xanthomonas.faa', final_datasets_dir / 'negative_Xanthomonas_data.fasta')
    split_fasta(logger, all_data_dir / 't3e_non_xanthomonas.faa', final_datasets_dir / 'positive_train_data.fasta',
                final_datasets_dir / 'positive_test_data.fasta', args.test_ratio)
    split_fasta(logger, all_data_dir / 'negative_ecoli.faa', final_datasets_dir / 'negative_train_data.fasta',
                final_datasets_dir / 'negative_test_data.fasta', args.test_ratio)


if __name__ == '__main__':
    main()
