import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from common.classic_ml_classifiers.test_classifiers_on_embeddings import main as test_classifiers_on_embeddings
from effectidor2_paper.src.utils.consts_paths import (EMBEDDINGS_DIR, CLASSIFIERS_OUTPUT_DIR,
                                                       FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--hidden_layer_number', help='Encoder layer index used during training (must match)', type=int, default=None)
    return parser.parse_args()


def _calc_embeddings_fn_for(model_id):
    if model_id == 'protein_bert':
        # Lazily imported: needs the protein_bert submodule + the separate TensorFlow conda env, so this must
        # never be imported for the esm/pt5 codepaths (which run in the PyTorch env instead).
        from effectidor2_paper.src.pretrained_embeddings.calc_proteinbert_embeddings import calc_embeddings
        return calc_embeddings
    return None  # let common auto-pick the esm/pt5 embedding calculator based on model_id


if __name__ == "__main__":
    args = get_arguments()
    test_classifiers_on_embeddings(args.model_id, embeddings_dir=EMBEDDINGS_DIR,
                                   classifiers_output_dir=CLASSIFIERS_OUTPUT_DIR,
                                   positive_fasta_file=FIXED_POSITIVE_TEST_FILE,
                                   negative_fasta_file=FIXED_NEGATIVE_TEST_FILE,
                                   hidden_layer_number=args.hidden_layer_number,
                                   calc_embeddings_fn=_calc_embeddings_fn_for(args.model_id))
