from pathlib import Path
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from consts import (POSITIVE_TRAIN_FILE, NEGATIVE_TRAIN_FILE, POSITIVE_TEST_FILE, NEGATIVE_TEST_FILE,
                    POSITIVE_XANTOMONAS_FILE, NEGATIVE_XANTOMONAS_FILE, DATASETS_FIXED_DIR)


def fix_all_fasta_files():
    fix_fasta_file(POSITIVE_TRAIN_FILE)
    fix_fasta_file(NEGATIVE_TRAIN_FILE)
    fix_fasta_file(POSITIVE_TEST_FILE)
    fix_fasta_file(NEGATIVE_TEST_FILE)
    fix_fasta_file(POSITIVE_XANTOMONAS_FILE)
    fix_fasta_file(NEGATIVE_XANTOMONAS_FILE)


def fix_fasta_file(file_path):
    with open(file_path, "r") as f:
        records = list(SeqIO.parse(f, "fasta"))
    fixed_records = [SeqRecord(seq=record.seq, id=record.id.replace("/", " ").replace("|", "_"), description="") for record in records]

    fixed_file_path = os.path.join(DATASETS_FIXED_DIR, "fixed_" + Path(file_path).name)
    SeqIO.write(fixed_records, fixed_file_path, "fasta")


if __name__ == "__main__":
    fix_all_fasta_files()
