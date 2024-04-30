import random
import argparse
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score

from consts import OUTPUTS_DIR, MODEL_ID_TO_MODEL_NAME
from utils import get_class_name
from classifiers_params_grids import classifiers, update_grid_params

ESM_EMBEDDINGS_BASE_DIR = os.path.join(OUTPUTS_DIR, 'embeddings')
CLASSIFIERS_OUTPUT_BASE_DIR = os.path.join(OUTPUTS_DIR, 'embeddings_classifiers')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='The pretrained model id', type=str, required=True)
    parser.add_argument('--output_dir', help='The output dir. By default it is just model_id', type=str)
    parser.add_argument('--esm_embeddings_calculation_mode', help='the esm embeddings calculation mode', type=str,
                        choices=['native_script', 'huggingface_model', 'trainer_api'], required=True)
    return parser.parse_args()


def prepare_Xs_and_Ys(model_name, embeddings_dir, split, esm_embeddings_calculation_mode, always_calc_embeddings):
    if model_name == 'protein_bert':
        from calc_proteinbert_embeddings import calc_embeddings
        Xs_positive, Xs_negative = calc_embeddings(embeddings_dir, split, always_calc_embeddings)
    else:
        from calc_esm_embeddings import calc_embeddings
        Xs_positive, Xs_negative = calc_embeddings(model_name, embeddings_dir, split, esm_embeddings_calculation_mode, always_calc_embeddings)

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    # Shuffle
    combined = list(zip(Xs, Ys))
    random.shuffle(combined)
    shuffled_Xs, shuffled_Ys = zip(*combined)
    shuffled_Xs = np.array(shuffled_Xs)

    logging.info(f"Loaded {split} data: Xs_{split}.shape = {Xs.shape}, Ys_{split}.shape = {len(Ys)}")

    return shuffled_Xs, shuffled_Ys


def pca(Xs, Ys, output_dir, n_components=2):
    pca = PCA(n_components=n_components)
    Xs_pca = pca.fit_transform(Xs)

    fig_dims = (7, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sc = ax.scatter(Xs_pca[:, 0], Xs_pca[:, 1], c=Ys, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    fig.colorbar(sc, label='Class')
    fig.savefig(os.path.join(output_dir, 'train_examples_pca.png'))


def fit_on_train_data(Xs_train, Ys_train, output_dir):
    grids = {}
    best_classifiers = {}
    for classifier, param_grid in classifiers:
        class_name = get_class_name(classifier)
        logging.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
        grid = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring={'mcc': make_scorer(matthews_corrcoef), 'auprc': 'average_precision'},
            refit='mcc',
            return_train_score=True,
            verbose=1,
            n_jobs=60
        )

        try:
            grid.fit(Xs_train, Ys_train)
            grid_results = pd.DataFrame.from_dict(grid.cv_results_)
            grid_results.to_csv(os.path.join(output_dir, f'{class_name}_grid_results.csv'))
            grids[class_name] = grid

            # Note: grid.best_score_ == grid_results['mean_test_mcc'][grid.best_index_] (the mean cross-validated score of the best_estimator)
            logging.info(f"Best params: {grid.best_params_}, Best index: {grid.best_index_}, Best score: {grid.best_score_}")

            logging.info(f"Best estimator - Mean MCC on train folds: {grid_results['mean_train_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on train folds: {grid_results['mean_train_auprc'][grid.best_index_]}, "
                         f"Mean MCC on held-out folds: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on held-out folds: {grid_results['mean_test_auprc'][grid.best_index_]}")

            best_classifiers[class_name] = (grid.best_index_,
                                            grid_results['mean_train_mcc'][grid.best_index_], grid_results['mean_train_auprc'][grid.best_index_],
                                            grid.best_score_, grid_results['mean_test_auprc'][grid.best_index_])
        except Exception as e:
            logging.error(f"Failed to train classifier {class_name} with error: {e}")

    logging.info(f"Best classifiers scores (on held-out validation folds): {best_classifiers}")
    best_classifiers_df = pd.DataFrame.from_dict(best_classifiers, orient='index',
                                                 columns=['best_index', 'mean_mcc_on_train_folds', 'mean_auprc_on_train_folds',
                                                          'mean_mcc_on_held_out_folds', 'mean_auprc_on_held_out_folds'])
    best_classifiers_df.index.name = 'classifier_class'
    best_classifiers_df.to_csv(os.path.join(output_dir, 'best_classifier_from_each_class.csv'))

    best_classifier_class = best_classifiers_df['mean_mcc_on_held_out_folds'].idxmax()
    logging.info(f"Best classifier (according to mean_mcc_on_held_out_folds): {best_classifier_class}")

    best_grid = grids[best_classifier_class]
    best_classifier_metrics = best_classifiers_df.loc[[best_classifier_class]].reset_index()
    return best_classifier_class, best_grid, best_classifier_metrics


def test_on_test_data(model_name, embeddings_dir, best_grid, split, esm_embeddings_calculation_mode):
    # First, estimate time of embedding and prediction
    start_test_time = timer()

    Xs_test, Ys_test = prepare_Xs_and_Ys(model_name, embeddings_dir, split, esm_embeddings_calculation_mode, always_calc_embeddings=True)
    Xs_test_predictions = best_grid.predict_proba(Xs_test)[:, 1]

    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    # Now, calculate the metrics
    mcc_on_test = best_grid.score(Xs_test, Ys_test)
    auprc_on_test = average_precision_score(Ys_test, Xs_test_predictions)

    logging.info(f"Best estimator - MCC on {split}: {mcc_on_test}, AUPRC on {split}: {auprc_on_test}, "
                 f"time took for embedding and prediction: {elapsed_time} seconds.")

    test_results = pd.DataFrame({f'{split}_mcc': [mcc_on_test], f'{split}_auprc': [auprc_on_test],
                                 f'{split}_elapsed_time': [elapsed_time]})
    return test_results


def main(model_id, output_dir, esm_embeddings_calculation_mode):
    if output_dir is None:
        output_dir = model_id

    embeddings_dir = os.path.join(ESM_EMBEDDINGS_BASE_DIR, output_dir)
    classifiers_output_dir = os.path.join(CLASSIFIERS_OUTPUT_BASE_DIR, output_dir)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(classifiers_output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(
                            os.path.join(classifiers_output_dir, 'classification_with_classic_ML.log'), mode='w')])

    model_name = MODEL_ID_TO_MODEL_NAME[model_id]

    Xs_train, Ys_train = prepare_Xs_and_Ys(model_name, embeddings_dir, 'train', esm_embeddings_calculation_mode, always_calc_embeddings=False)
    update_grid_params(Ys_train)

    pca(Xs_train, Ys_train, embeddings_dir)

    best_classifier_class, best_grid, best_classifier_metrics = fit_on_train_data(Xs_train, Ys_train, classifiers_output_dir)

    test_results = test_on_test_data(model_name, embeddings_dir, best_grid, 'test', esm_embeddings_calculation_mode)
    xantomonas_results = test_on_test_data(model_name, embeddings_dir, best_grid, 'xantomonas', esm_embeddings_calculation_mode)
    pd.concat([best_classifier_metrics, test_results, xantomonas_results], axis=1).to_csv(
        os.path.join(classifiers_output_dir, 'best_classifier_results.csv'), index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args.model_id, args.output_dir, args.esm_embeddings_calculation_mode)
