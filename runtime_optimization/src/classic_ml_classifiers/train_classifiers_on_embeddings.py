import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.classic_ml_classifiers.train_classifiers_on_embeddings import main as train_classifiers_on_embeddings
from runtime_optimization.src.utils.consts_paths import (EMBEDDINGS_DIR, CLASSIFIERS_OUTPUT_DIR,
                                                         POSITIVE_TRAIN_FILE, NEGATIVE_TRAIN_FILE)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--n_jobs', help='The number of jobs to run in parallel', type=int, default=1)
    parser.add_argument('--hidden_layer_number', help='Encoder layer index to use for embeddings (1=first block, None=last hidden state)', type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    train_classifiers_on_embeddings(args.model_id, args.n_jobs, args.hidden_layer_number,
                                    embeddings_dir=EMBEDDINGS_DIR, classifiers_output_dir=CLASSIFIERS_OUTPUT_DIR,
                                    positive_fasta_file=POSITIVE_TRAIN_FILE,
                                    negative_fasta_file=NEGATIVE_TRAIN_FILE)
