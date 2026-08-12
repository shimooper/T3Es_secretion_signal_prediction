import argparse
import functools
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import logging
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn import __version__ as sklearn_version

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.consts import MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION
from common.classic_ml_classifiers.classifiers_params_grids import classifiers, update_grid_params
from common.classic_ml_classifiers.utils import prepare_Xs_and_Ys


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--n_jobs', help='The number of jobs to run in parallel', type=int, default=1)
    parser.add_argument('--hidden_layer_number', help='Encoder layer index to use for embeddings (1=first block, None=last hidden state)', type=int, default=None)
    parser.add_argument('--embeddings_dir', help='Base directory to cache computed embeddings under', type=Path, required=True)
    parser.add_argument('--classifiers_output_dir', help='Base directory to write classifier results to', type=Path, required=True)
    parser.add_argument('--positive_fasta_file', help='Path to the positive-class train FASTA file', type=Path, required=True)
    parser.add_argument('--negative_fasta_file', help='Path to the negative-class train FASTA file', type=Path, required=True)
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


def pca(Xs, Ys, output_dir, n_components=2):
    pca = PCA(n_components=n_components)
    Xs_pca = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Xs_pca[:, 0], Xs_pca[:, 1], c=Ys, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    fig.colorbar(sc, label='Class')
    fig.savefig(output_dir / 'train_examples_pca.png')


def fit_on_train_data(Xs_train, Ys_train, output_dir, n_jobs):
    best_classifiers = {}
    best_classifiers_metrics = {}
    for classifier, param_grid in classifiers:
        class_name = classifier.__class__.__name__
        logging.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
        grid = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring={'mcc': make_scorer(matthews_corrcoef), 'auprc': 'average_precision'},
            refit='mcc',
            return_train_score=True,
            verbose=1,
            n_jobs=n_jobs
        )

        try:
            grid.fit(Xs_train, Ys_train)
            grid_results = pd.DataFrame.from_dict(grid.cv_results_)
            grid_results.to_csv(output_dir / f'{class_name}_grid_results.csv')
            best_classifiers[class_name] = grid.best_estimator_
            joblib.dump(grid.best_estimator_, output_dir / f"best_{class_name}.pkl")

            # Note: grid.best_score_ == grid_results['mean_test_mcc'][grid.best_index_] (the mean cross-validated score of the best_estimator)
            logging.info(f"Best params: {grid.best_params_}, Best index: {grid.best_index_}, Best score: {grid.best_score_}")

            logging.info(f"Best estimator - Mean MCC on train folds: {grid_results['mean_train_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on train folds: {grid_results['mean_train_auprc'][grid.best_index_]}, "
                         f"Mean MCC on held-out folds: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on held-out folds: {grid_results['mean_test_auprc'][grid.best_index_]}")

            best_classifiers_metrics[class_name] = (grid.best_index_,
                                                    grid_results['mean_train_mcc'][grid.best_index_],
                                                    grid_results['mean_train_auprc'][grid.best_index_],
                                                    grid.best_score_,
                                                    grid_results['mean_test_auprc'][grid.best_index_])
        except Exception as e:
            logging.error(f"Failed to train classifier {class_name} with error: {e}")

    logging.info(f"Best classifiers scores: {best_classifiers_metrics}")
    best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                 columns=['best_index', 'mean_mcc_on_train_folds', 'mean_auprc_on_train_folds',
                                                          'mean_mcc_on_held_out_folds', 'mean_auprc_on_held_out_folds'])
    best_classifiers_df.index.name = 'classifier_class'
    best_classifiers_df.to_csv(output_dir / 'best_classifier_from_each_class.csv')

    best_classifier_class = best_classifiers_df['mean_mcc_on_held_out_folds'].idxmax()
    logging.info(f"Best classifier (according to mean_mcc_on_held_out_folds): {best_classifier_class}")

    # Save the best classifier to disk
    joblib.dump(best_classifiers[best_classifier_class], output_dir / "model.pkl")
    # Save metadata
    metadata = {
        'numpy_version': np.__version__,
        'joblib_version': joblib.__version__,
        'sklearn_version': sklearn_version
    }
    with open(output_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f)

    best_classifier_metrics = best_classifiers_df.loc[[best_classifier_class]].reset_index()

    return best_classifier_metrics


def main(model_id, n_jobs, embeddings_dir, classifiers_output_dir, positive_fasta_file, negative_fasta_file,
         hidden_layer_number=None, calc_embeddings_fn=None):
    if calc_embeddings_fn is None:
        calc_embeddings_fn = _default_calc_embeddings_fn(model_id, hidden_layer_number)

    model_embeddings_dir = embeddings_dir / model_id
    layer_subdir = f'layer_{hidden_layer_number}' if hidden_layer_number is not None else ''
    model_classifiers_output_dir = (classifiers_output_dir / model_id / layer_subdir
                                    if layer_subdir else classifiers_output_dir / model_id)
    model_embeddings_dir.mkdir(parents=True, exist_ok=True)
    model_classifiers_output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            model_classifiers_output_dir / 'classification_with_classic_ML.log', mode='w')])
    logger = logging.getLogger(__name__)

    Xs_train, Ys_train = prepare_Xs_and_Ys(logger, calc_embeddings_fn, 'train', always_calc_embeddings=False,
                                            positive_fasta_file=positive_fasta_file,
                                            negative_fasta_file=negative_fasta_file,
                                            embeddings_dir=embeddings_dir)
    update_grid_params(Ys_train)

    pca(Xs_train, Ys_train, model_embeddings_dir)

    best_classifier_metrics = fit_on_train_data(Xs_train, Ys_train, model_classifiers_output_dir, n_jobs)

    best_classifier_metrics['model_id'] = [model_id]
    best_classifier_metrics['hidden_layer_number'] = [hidden_layer_number]
    best_classifier_metrics['training_mode'] = ['only_head']
    best_classifier_metrics['number_of_parameters (millions)'] = [MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[model_id]]

    best_classifier_metrics.to_csv(model_classifiers_output_dir / 'best_classifier_train_results.csv', index=False)

    logging.info(f"Finished training classifiers on embeddings for model {model_id}")


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id, args.n_jobs, args.embeddings_dir, args.classifiers_output_dir, args.positive_fasta_file,
         args.negative_fasta_file, args.hidden_layer_number)
