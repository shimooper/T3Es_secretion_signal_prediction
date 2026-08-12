import functools
import logging
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
import json
import sys
from pathlib import Path
from sklearn import __version__ as sklearn_version

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from effectidor2_paper.src.utils.consts_paths import (FINAL_RESULTS, EMBEDDINGS_DIR, FIXED_POSITIVE_TRAIN_FILE,
                                                       FIXED_NEGATIVE_TRAIN_FILE, FIXED_POSITIVE_TEST_FILE,
                                                       FIXED_NEGATIVE_TEST_FILE)
from common.classic_ml_classifiers.utils import prepare_Xs_and_Ys
from common.pretrained_embeddings.calc_pt5_embeddings import calc_pt5_embeddings


def fit_on_data(Xs, Ys, output_dir):
    logging.info(f"Training Classifier with hyperparameters fixed")

    clf = MLPClassifier(activation='tanh', alpha=0.0001, early_stopping=True, hidden_layer_sizes=(50, 10),
                        learning_rate='constant', max_iter=400, random_state=500, solver='adam')
    clf.fit(Xs, Ys)

    # Save the best classifier to disk
    joblib.dump(clf, output_dir / "model.pkl")
    # Save metadata
    metadata = {
        'numpy_version': np.__version__,
        'joblib_version': joblib.__version__,
        'sklearn_version': sklearn_version
    }
    with open(output_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f)


def main():
    model_id = 'pt5'
    output_dir = FINAL_RESULTS / 'trained_pt5_head'
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            output_dir / 'classification_with_classic_ML.log', mode='w')])
    logger = logging.getLogger(__name__)

    calc_embeddings_fn = functools.partial(calc_pt5_embeddings, model_id)
    Xs_train, Ys_train = prepare_Xs_and_Ys(logger, calc_embeddings_fn, 'train', always_calc_embeddings=False,
                                            positive_fasta_file=FIXED_POSITIVE_TRAIN_FILE,
                                            negative_fasta_file=FIXED_NEGATIVE_TRAIN_FILE,
                                            embeddings_dir=EMBEDDINGS_DIR)
    Xs_test, Ys_test = prepare_Xs_and_Ys(logger, calc_embeddings_fn, 'test', always_calc_embeddings=False,
                                          positive_fasta_file=FIXED_POSITIVE_TEST_FILE,
                                          negative_fasta_file=FIXED_NEGATIVE_TEST_FILE,
                                          embeddings_dir=EMBEDDINGS_DIR)

    Xs = np.concatenate([Xs_train, Xs_test])
    Ys = np.concatenate([Ys_train, Ys_test])

    fit_on_data(Xs, Ys, output_dir)
    logging.info(f"Finished training classifier on embeddings for model {model_id}")


if __name__ == "__main__":
    main()
