from Bio import SeqIO
from pathlib import Path
import argparse
import subprocess

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT_DIR = SCRIPT_DIR.parent.parent
DB_PATH = PROJECT_ROOT_DIR / 'new_data' / 'raw_data' / 'T3Es10_EDITED_NO_PARTIAL_v2.faa'
INFERENCE_SCRIPT_PATH = PROJECT_ROOT_DIR / 'src' / 'inference' / 'predict_secretion_signal.py'


def cut_N_terminal(input_path, output_path):
    records = list(SeqIO.parse(input_path, 'fasta'))
    for record in records:
        record.seq = record.seq[:100]

    SeqIO.write(records, output_path, 'fasta')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    cut_N_terminal(DB_PATH, args.output_dir / 'db_N_terminal.faa')
    cmd = f'python {INFERENCE_SCRIPT_PATH} --input_fasta_file {args.output_dir / "db_N_terminal.faa"} --load_llm_from_disk'
    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    main()
