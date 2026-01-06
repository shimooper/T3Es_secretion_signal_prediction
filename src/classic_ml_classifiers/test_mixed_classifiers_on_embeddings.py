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

from sklearn.metrics import matthews_corrcoef, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import CLASSIFIERS_OUTPUT_DIR, FIXED_POSITIVE_TEST_FILE, FIXED_NEGATIVE_TEST_FILE
from src.pretrained_embeddings.calc_esm_embeddings import calc_embeddings_of_fasta_file_with_huggingface_model_esm


def test_on_test_data(logger, accuracy_threshold, output_dir):
    start_test_time = timer()

    # First, generate embeddings of all test sequences using esm6, estimate runtime, and predict probabilities.
    esm6_embeddings_output_dir = Path(output_dir) / 'esm_6_embeddings'
    esm6_embeddings_output_dir.mkdir(parents=True, exist_ok=True)

    positive_embeddings_output_file_path = esm6_embeddings_output_dir / f'positive_embeddings.npy'
    negative_embeddings_output_file_path = esm6_embeddings_output_dir / f'negative_embeddings.npy'
    Xs_positive = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
        'esm_6', FIXED_POSITIVE_TEST_FILE, positive_embeddings_output_file_path)
    Xs_negative = calc_embeddings_of_fasta_file_with_huggingface_model_esm(
        'esm_6', FIXED_NEGATIVE_TEST_FILE, negative_embeddings_output_file_path)

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    model_esm_6 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'esm_6', f'model.pkl'))
    Ys_test_predictions = model_esm_6.predict_proba(Xs)


    model_pt5 = joblib.load(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5', f'model.pkl'))


    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    # Now, calculate the metrics
    mcc_on_test = matthews_corrcoef(Ys_test, Ys_test_predictions.argmax(axis=1))
    auprc_on_test = average_precision_score(Ys_test, Ys_test_predictions[:, 1])

    logging.info(f"Best estimator - MCC on {split}: {mcc_on_test}, AUPRC on {split}: {auprc_on_test}, "
                 f"time took for embedding and prediction: {elapsed_time} seconds.")

    test_results = pd.DataFrame({f'{split}_mcc': [mcc_on_test], f'{split}_auprc': [auprc_on_test],
                                 f'{split}_elapsed_time': [elapsed_time]})
    return test_results


def main(accuracy_threshold):
    output_classifiers_dir = os.path.join(CLASSIFIERS_OUTPUT_DIR, 'mixed')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(output_classifiers_dir, 'classification_with_classic_ML_test.log'), mode='w')])
    logger = logging.getLogger(__name__)

    test_results = test_on_test_data(logger, accuracy_threshold, output_classifiers_dir)
    test_results.to_csv(os.path.join(output_classifiers_dir, 'mixed_classifier_test_results.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--accuracy_threshold', help='The threshold required to calculate the accurate model', type=float, required=True)
    args = parser.parse_args()
    main(args.accuracy_threshold)
