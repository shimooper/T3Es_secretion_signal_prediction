from pathlib import Path
from Bio import SeqIO
import subprocess
import logging
import argparse
import pandas as pd
import re
import shutil

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = SCRIPT_DIR / 'raw_data'

T3E_PATH = RAW_DATA_PATH / 'T3Es10_EDITED_NO_PARTIAL_v2.faa'
ECOLI_PATH = RAW_DATA_PATH / 'corrected_e_coli_k12.faa'
XANTHOMONAS_PATH = RAW_DATA_PATH / 'Xcc8004.fasta'
DB_PATH = RAW_DATA_PATH / 'T3Edb_summary.csv'

LOG_MESSAGE_FORMAT = '%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s'


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


def validate_xanthomonas(logger, t3e_records):
    t3e_xanthomonas_records = [record for record in t3e_records if
                               'Xanthomonas' in record.description or 'Xanthomonas' in record.id]

    db_df = pd.read_csv(DB_PATH, encoding='utf-8', encoding_errors='replace')
    db_xanthomonas_df = db_df[db_df['species'].str.startswith('Xanthomonas')]
    db_xanthomonas_ids = list(db_xanthomonas_df['rec_id'])
    logger.info(
        f'There are {len(t3e_xanthomonas_records)} Xanthomonas in T3Es fasta and {len(db_xanthomonas_df)} in the database.')

    fasta_xanthomonas_ids = []
    for record in t3e_xanthomonas_records:
        match = re.search(r'^(.*?)_Xanthomonas', record.description)
        if match is None:
            logger.error(f'Xanthomonas record {record.id} from fasta does not have a valid record id.')
            continue

        rec_id = match.group(1)
        fasta_xanthomonas_ids.append(rec_id)
        fasta_xanthomonas_ids.append(record.id)
        if rec_id not in db_xanthomonas_ids and record.id not in db_xanthomonas_ids:
            logger.error(f'Xanthomonas record {record.id} from fasta is not in the database.')

    for record_id in db_xanthomonas_ids:
        if record_id not in fasta_xanthomonas_ids:
            logger.error(f'Xanthomonas record {record_id} from db is not in the fasta.')


def cut_N_terminal(logger, records, output_path):
    for record in records:
        record.seq = record.seq[:100]

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


def main():
    # Init: Parse arguments, and create outputs directory and logger, and read sequences.
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir_name', default='outputs', help='Directory name to save outputs')
    parser.add_argument('--cluster_min_e_value', default=1e-3, type=float, help='Minimum e-value for clustering')
    parser.add_argument('--cluster_min_coverage', default=0.6, type=float, help='Minimum coverage for clustering')
    parser.add_argument('--cluster_min_identity', default=0.5, type=float, help='Minimum identity for clustering')
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

    # Step 1: Split T3E to xanthomonas and non-xanthomonas, cut N-terminal.
    te3_xanthomonas_records = [record for record in t3e_records if 'Xanthomonas' in record.description]
    te3_non_xanthomonas_records = [record for record in t3e_records if 'Xanthomonas' not in record.description]
    logger.info(f'There are {len(te3_xanthomonas_records)} Xanthomonas T3Es and {len(te3_non_xanthomonas_records)} '
                f'non-Xanthomonas T3Es.')

    n_terminal_dir = outputs_dir / '1__N_terminal'
    n_terminal_dir.mkdir(exist_ok=True, parents=True)
    cut_N_terminal(logger, te3_non_xanthomonas_records, n_terminal_dir / 't3e_non_xanthomonas.faa')
    cut_N_terminal(logger, te3_xanthomonas_records, n_terminal_dir / 't3e_xanthomonas.faa')
    cut_N_terminal(logger, ecoli_records, n_terminal_dir / 'ecoli.faa')
    cut_N_terminal(logger, xanthomonas_records, n_terminal_dir / 'xanthomonas.faa')

    # Step 2: Cluster sequences and choose representatives.
    representatives_dir = outputs_dir / '2__Representatives'
    representatives_dir.mkdir(exist_ok=True, parents=True)
    cluster_records(logger, n_terminal_dir / 't3e_non_xanthomonas.faa', representatives_dir / 't3e_non_xanthomonas',
                    representatives_dir / 't3e_non_xanthomonas_representatives.faa',
                    args.cluster_min_e_value, args.cluster_min_coverage, args.cluster_min_identity)
    cluster_records(logger, n_terminal_dir / 't3e_xanthomonas.faa', representatives_dir / 't3e_xanthomonas',
                    representatives_dir / 't3e_xanthomonas_representatives.faa',
                    args.cluster_min_e_value, args.cluster_min_coverage, args.cluster_min_identity)
    cluster_records(logger, n_terminal_dir / 'ecoli.faa', representatives_dir / 'ecoli',
                    representatives_dir / 'ecoli_representatives.faa',
                    args.cluster_min_e_value, args.cluster_min_coverage, args.cluster_min_identity)
    cluster_records(logger, n_terminal_dir / 'xanthomonas.faa', representatives_dir / 'xanthomonas',
                    representatives_dir / 'xanthomonas_representatives.faa',
                    args.cluster_min_e_value, args.cluster_min_coverage, args.cluster_min_identity)


if __name__ == '__main__':
    main()
