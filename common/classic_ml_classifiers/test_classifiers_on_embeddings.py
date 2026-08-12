try:
    import torch  # Important - without this I get weird error when trying to import torch in another module imported here
except Exception:
    pass
import argparse
import functools
from timeit import default_timer as timer
import joblib

import pandas as pd
import logging
import sys
from pathlib import Path

from sklearn.metrics import matthews_corrcoef, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.classic_ml_classifiers.utils import prepare_Xs_and_Ys


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--hidden_layer_number', help='Encoder layer index used during training (must match)', type=int, default=None)
    parser.add_argument('--embeddings_dir', help='Base directory to cache computed embeddings under', type=Path, required=True)
    parser.add_argument('--classifiers_output_dir', help='Base directory the trained classifier was written to', type=Path, required=True)
    parser.add_argument('--positive_fasta_file', help='Path to the positive-class test FASTA file', type=Path, required=True)
    parser.add_argument('--negative_fasta_file', help='Path to the negative-class test FASTA file', type=Path, required=True)
    return parser.parse_args()


def _default_calc_embeddings_fn(model_id, hidden_layer_number):
    # Lazily imported: these pull in torch/transformers, which must stay optional here so that this module can
    # still be imported (e.g. for the protein_bert model_id, via an explicitly-passed calc_embeddings_fn) in
    # environments that don't have the PyTorch stack installed (protein_bert runs in a separate TensorFlow env).
    if model_id == 'pt5':
        from common.pretrained_embeddings.calc_pt5_embeddings import calc_pt5_embeddings
        return functools.partial(calc_pt5_embeddings, model_id, hidden_layer_number=hidden_layer_number)
    from common.pretrained_embeddings.calc_esm_embeddings import calc_esm_embeddings
    return functools.partial(calc_esm_embeddings, model_id)


def test_on_test_data(logger, model, split, positive_fasta_file, negative_fasta_file, embeddings_dir,
                      calc_embeddings_fn):
    # First, estimate time of embedding and prediction
    start_test_time = timer()

    Xs_test, Ys_test = prepare_Xs_and_Ys(logger, calc_embeddings_fn, split, always_calc_embeddings=True,
                                          positive_fasta_file=positive_fasta_file,
                                          negative_fasta_file=negative_fasta_file,
                                          embeddings_dir=embeddings_dir)
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


def main(model_id, embeddings_dir, classifiers_output_dir, positive_fasta_file, negative_fasta_file,
         hidden_layer_number=None, calc_embeddings_fn=None):
    if calc_embeddings_fn is None:
        calc_embeddings_fn = _default_calc_embeddings_fn(model_id, hidden_layer_number)

    layer_subdir = f'layer_{hidden_layer_number}' if hidden_layer_number is not None else ''
    classifiers_dir = (classifiers_output_dir / model_id / layer_subdir
                       if layer_subdir else classifiers_output_dir / model_id)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            classifiers_dir / 'classification_with_classic_ML_test.log', mode='w')])
    logger = logging.getLogger(__name__)

    model = joblib.load(classifiers_dir / 'model.pkl')
    test_results = test_on_test_data(logger, model, 'test', positive_fasta_file, negative_fasta_file,
                                      embeddings_dir, calc_embeddings_fn)

    train_results = pd.read_csv(classifiers_dir / 'best_classifier_train_results.csv')
    all_results = pd.concat([train_results, test_results], axis=1)

    all_results.to_csv(classifiers_dir / 'best_classifier_all_results.csv', index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id, args.embeddings_dir, args.classifiers_output_dir, args.positive_fasta_file,
         args.negative_fasta_file, args.hidden_layer_number)
