from consts import POSITIVE_TRAIN_FILE, NEGATIVE_TRAIN_FILE, POSITIVE_TEST_FILE, NEGATIVE_TEST_FILE, POSITIVE_XANTOMONAS_FILE, NEGATIVE_XANTOMONAS_FILE
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


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
    fixed_records = [SeqRecord(seq=record.seq, id=record.id, description=record.description.replace("/", " ")) for record in records]
    SeqIO.write(fixed_records, f"{file_path}_fixed", "fasta")


if __name__ == "__main__":
    fix_all_fasta_files()
