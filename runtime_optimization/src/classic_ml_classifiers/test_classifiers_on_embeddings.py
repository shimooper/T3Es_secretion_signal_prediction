import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.classic_ml_classifiers.test_classifiers_on_embeddings import main as test_classifiers_on_embeddings
from runtime_optimization.src.utils.consts_paths import (EMBEDDINGS_DIR, CLASSIFIERS_OUTPUT_DIR,
                                                         POSITIVE_TEST_FILE, NEGATIVE_TEST_FILE)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--hidden_layer_number', help='Encoder layer index used during training (must match)', type=int, default=None)
    parser.add_argument('--device', help='Device to compute embeddings on (esm/pt5 only)', type=str, choices=['cpu', 'cuda'], default='cpu')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    test_classifiers_on_embeddings(args.model_id,
                                   embeddings_dir=EMBEDDINGS_DIR,
                                   classifiers_output_dir=CLASSIFIERS_OUTPUT_DIR,
                                   positive_fasta_file=POSITIVE_TEST_FILE,
                                   negative_fasta_file=NEGATIVE_TEST_FILE,
                                   hidden_layer_number=args.hidden_layer_number,
                                   device=args.device)
