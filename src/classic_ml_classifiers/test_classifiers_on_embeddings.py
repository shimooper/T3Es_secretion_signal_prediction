import argparse
from timeit import default_timer as timer
import joblib

import pandas as pd
import os
import logging
import sys

from sklearn.metrics import matthews_corrcoef, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import CLASSIFIERS_OUTPUT_DIR, CLASSIFIERS_TEST_OUTPUT_DIR, MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION
from utils import prepare_Xs_and_Ys


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    return parser.parse_args()


def test_on_test_data(logger, model_id, model, split):
    # First, estimate time of embedding and prediction
    start_test_time = timer()

    Xs_test, Ys_test = prepare_Xs_and_Ys(logger, model_id, split, always_calc_embeddings=True)
    Ys_test_predictions = model.predict_proba(Xs_test)

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


def main(model_id):
    classifiers_dir = os.path.join(CLASSIFIERS_OUTPUT_DIR, model_id)
    classifiers_test_dir = os.path.join(CLASSIFIERS_TEST_OUTPUT_DIR, model_id)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(classifiers_test_dir, 'classification_with_classic_ML_test.log'), mode='w')])
    logger = logging.getLogger(__name__)

    model = joblib.load(os.path.join(classifiers_dir, f'model.pkl'))
    test_results = test_on_test_data(logger, model_id, model, 'test')
    xantomonas_results = test_on_test_data(logger, model_id, model,  'xantomonas')

    all_test_results = pd.concat([test_results, xantomonas_results], axis=1)
    all_test_results['model_id'] = [model_id]
    all_test_results['training_mode'] = ['only_head']
    all_test_results['number_of_parameters (millions)']: MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[model_id]

    all_test_results.to_csv(os.path.join(classifiers_test_dir, 'best_classifier_test_results.csv'), index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id)
