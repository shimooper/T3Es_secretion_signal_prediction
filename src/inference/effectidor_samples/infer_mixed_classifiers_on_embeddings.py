try:
    import torch  # Important - without this I get weird error when trying to import torch in another module imported here
except Exception:
    pass
import argparse
from timeit import default_timer as timer
import joblib

import pandas as pd
import numpy as np
import os
import logging
import sys
from pathlib import Path
from Bio import SeqIO

from sklearn.metrics import matthews_corrcoef, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import CLASSIFIERS_OUTPUT_DIR, FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE
from src.pretrained_embeddings.calc_esm_embeddings import calc_embeddings_of_fasta_file_with_huggingface_model_esm
from src.pretrained_embeddings.calc_pt5_embeddings import calc_embeddings_of_fasta_file_with_huggingface_model_pt5


def filter_fasta_by_indices(input_fasta, output_fasta, indices_set):
    records = list(SeqIO.parse(input_fasta, "fasta"))
    selected = [r for i, r in enumerate(records) if i in indices_set]
    SeqIO.write(selected, output_fasta, "fasta")


def test_on_test_data(logger, fasta_path, accuracy_threshold, output_dir):
    start_test_time = timer()

    # First, generate embeddings of all test sequences using esm6 and predict probabilities.
    esm6_embeddings_output_dir = Path(output_dir) / 'esm_6_embeddings'
    esm6_embeddings_output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_output_file_path = esm6_embeddings_output_dir / f'embeddings.npy'
    Xs = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
        'esm_6', fasta_path, embeddings_output_file_path)
    logger.info(f"ESM-6 embeddings calculated: shape {Xs.shape}")

    model_esm_6 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'esm_6', f'model.pkl'))
    probs_esm6 = model_esm_6.predict_proba(Xs)[:, 1]
    logger.info(f"ESM-6 model probabilities predicted.")

    significant_mask = (probs_esm6 < accuracy_threshold) | (probs_esm6 > (1 - accuracy_threshold))
    nonsignificant_mask = ~significant_mask

    final_probs = np.empty_like(probs_esm6)
    final_probs[significant_mask] = probs_esm6[significant_mask]

    if any(nonsignificant_mask):
        n_pos = Xs.shape[0]

        nonsig_indices = np.where(nonsignificant_mask)[0]

        nonsig_pos_indices = [i for i in nonsig_indices if i < n_pos]
        nonsig_neg_indices = [i - n_pos for i in nonsig_indices if i >= n_pos]

        pt5_dir = Path(output_dir) / "pt5_stage"
        pt5_dir.mkdir(exist_ok=True)

        PT5_FASTA = pt5_dir / "nonsignificant.fasta"

        filter_fasta_by_indices(fasta_path, PT5_FASTA, nonsig_pos_indices)
        logger.info(f"Filtered nonsignificant sequences for PT5 embeddings. Wrote filtered FASTA files to {PT5_FASTA}")

        pt5_embeddings_dir = pt5_dir / "embeddings"
        pt5_embeddings_dir.mkdir(exist_ok=True)

        Xs_pt5 = calc_embeddings_of_fasta_file_with_huggingface_model_pt5(
            'pt5', PT5_FASTA, pt5_embeddings_dir / "embeddings.npy")
        logger.info(f"PT5 embeddings calculated for nonsignificant sequences: shape {Xs_pt5.shape}")

        model_pt5 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5', f'model.pkl'))
        probs_pt5 = model_pt5.predict_proba(Xs_pt5)[:, 1]
        logger.info(f"PT5 model probabilities predicted for nonsignificant sequences.")

        final_probs[nonsignificant_mask] = probs_pt5
        assert np.sum(nonsignificant_mask) == len(probs_pt5)

    assert not np.any(np.isnan(final_probs))

    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    logging.info(f"time took for embedding and prediction: {elapsed_time} seconds.")

    test_results = pd.DataFrame({'test_elapsed_time': [elapsed_time]})
    return test_results


def main(fasta_path, accuracy_threshold):
    output_dir = fasta_path.parent / f'mixed_{accuracy_threshold}'
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(output_dir, 'classification_with_classic_ML_test.log'), mode='w')])
    logger = logging.getLogger(__name__)

    test_results = test_on_test_data(logger, fasta_path, accuracy_threshold, output_dir)
    test_results.to_csv(os.path.join(output_dir, 'mixed_classifier_test_results.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_path', type=Path, help='Path to the input FASTA file', required=True)
    parser.add_argument('--accuracy_threshold', help='The threshold required to calculate the accurate model', type=float, default=0.4)
    args = parser.parse_args()
    main(args.fasta_path, args.accuracy_threshold)
