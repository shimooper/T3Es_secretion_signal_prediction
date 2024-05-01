from Bio import SeqIO


def get_class_name(obj):
    return obj.__class__.__name__


def read_fasta_file(file_path):
    sequences = {}
    for seq_record in SeqIO.parse(file_path, 'fasta'):
        sequences[seq_record.id] = str(seq_record.seq)
    return sequences


def read_sequences_from_fasta_file(file_path):
    id_to_sequence = read_fasta_file(file_path)
    return list(id_to_sequence.values())
