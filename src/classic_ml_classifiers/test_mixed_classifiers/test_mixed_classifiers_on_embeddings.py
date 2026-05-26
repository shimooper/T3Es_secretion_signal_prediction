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


def test_on_test_data(logger, lower_threshold, upper_threshold, output_dir):
    start_test_time = timer()

    # First, generate embeddings of all test sequences using esm6 and predict probabilities.
    esm6_embeddings_output_dir = Path(output_dir) / 'esm_6_embeddings'
    esm6_embeddings_output_dir.mkdir(parents=True, exist_ok=True)

    positive_embeddings_output_file_path = esm6_embeddings_output_dir / f'positive_embeddings.npy'
    negative_embeddings_output_file_path = esm6_embeddings_output_dir / f'negative_embeddings.npy'
    Xs_positive = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
        'esm_6', FIXED_POSITIVE_TEST_FILE, positive_embeddings_output_file_path)
    Xs_negative = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
        'esm_6', FIXED_NEGATIVE_TEST_FILE, negative_embeddings_output_file_path)
    logger.info(f"ESM-6 embeddings calculated: positive shape {Xs_positive.shape}, negative shape {Xs_negative.shape}")

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    model_esm_6 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'esm_6', f'model.pkl'))
    probs_esm6 = model_esm_6.predict_proba(Xs)[:, 1]
    logger.info(f"ESM-6 model probabilities predicted.")

    significant_mask = (probs_esm6 < lower_threshold) | (probs_esm6 > upper_threshold)
    nonsignificant_mask = ~significant_mask

    final_probs = np.empty_like(probs_esm6)
    final_probs[significant_mask] = probs_esm6[significant_mask]

    if any(nonsignificant_mask):
        n_pos = Xs_positive.shape[0]

        nonsig_indices = np.where(nonsignificant_mask)[0]

        nonsig_pos_indices = [i for i in nonsig_indices if i < n_pos]
        nonsig_neg_indices = [i - n_pos for i in nonsig_indices if i >= n_pos]

        pt5_dir = Path(output_dir) / "pt5_stage"
        pt5_dir.mkdir(exist_ok=True)

        POS_PT5_FASTA = pt5_dir / "positive_nonsignificant.fasta"
        NEG_PT5_FASTA = pt5_dir / "negative_nonsignificant.fasta"

        filter_fasta_by_indices(FIXED_POSITIVE_TEST_FILE, POS_PT5_FASTA, nonsig_pos_indices)
        filter_fasta_by_indices(FIXED_NEGATIVE_TEST_FILE, NEG_PT5_FASTA, nonsig_neg_indices)
        logger.info(f"Filtered nonsignificant sequences for PT5 embeddings. Wrote filtered FASTA files to {POS_PT5_FASTA} and {NEG_PT5_FASTA}")

        pt5_embeddings_dir = pt5_dir / "embeddings"
        pt5_embeddings_dir.mkdir(exist_ok=True)

        Xs_positive_pt5 = calc_embeddings_of_fasta_file_with_huggingface_model_pt5(
            'pt5', POS_PT5_FASTA, pt5_embeddings_dir / "positive.npy")
        Xs_negative_pt5 = calc_embeddings_of_fasta_file_with_huggingface_model_pt5(
            'pt5', NEG_PT5_FASTA, pt5_embeddings_dir / "negative.npy")
        Xs_pt5 = np.concatenate([Xs_positive_pt5, Xs_negative_pt5])
        logger.info(f"PT5 embeddings calculated for nonsignificant sequences: shape {Xs_pt5.shape}")

        model_pt5 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5', f'model.pkl'))
        probs_pt5 = model_pt5.predict_proba(Xs_pt5)[:, 1]
        logger.info(f"PT5 model probabilities predicted for nonsignificant sequences.")

        final_probs[nonsignificant_mask] = probs_pt5
        assert np.sum(nonsignificant_mask) == len(probs_pt5)

    assert not np.any(np.isnan(final_probs))

    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    # Now, calculate the metrics
    mcc_on_test = matthews_corrcoef(Ys, final_probs > 0.5)
    auprc_on_test = average_precision_score(Ys, final_probs)

    logging.info(f"Best estimator - MCC on test: {mcc_on_test}, AUPRC on test: {auprc_on_test}, "
                 f"time took for embedding and prediction: {elapsed_time} seconds.")

    test_results = pd.DataFrame({f'test_mcc': [mcc_on_test], f'test_auprc': [auprc_on_test],
                                 f'test_elapsed_time': [elapsed_time]})
    return test_results


def main(lower_threshold, upper_threshold):
    output_classifiers_dir = os.path.join(CLASSIFIERS_OUTPUT_DIR, f'mixed_{lower_threshold}_{upper_threshold}')
    os.makedirs(output_classifiers_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(output_classifiers_dir, 'classification_with_classic_ML_test.log'), mode='w')])
    logger = logging.getLogger(__name__)

    test_results = test_on_test_data(logger, lower_threshold, upper_threshold, output_classifiers_dir)
    test_results.to_csv(os.path.join(output_classifiers_dir, 'mixed_classifier_test_results.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lower_threshold', type=float, default=0.4,
                        help='ESM-6 probabilities below this are classified directly (negative).')
    parser.add_argument('--upper_threshold', type=float, default=0.6,
                        help='ESM-6 probabilities above this are classified directly (positive). '
                             'Sequences with probabilities in [lower_threshold, upper_threshold] are escalated to PT5.')
    args = parser.parse_args()
    main(args.lower_threshold, args.upper_threshold)
