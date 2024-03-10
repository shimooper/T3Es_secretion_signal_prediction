import random
from collections import Counter

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import logging

import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score

from consts import (OUTPUTS_DIR, EMBEDDINGS_POSITIVE_TRAIN_DIR, EMBEDDINGS_NEGATIVE_TRAIN_DIR,
                    EMBEDDINGS_POSITIVE_TEST_DIR, EMBEDDINGS_NEGATIVE_TEST_DIR,
                    EMBEDDINGS_POSITIVE_XANTOMONAS_DIR, EMBEDDINGS_NEGATIVE_XANTOMONAS_DIR)
from utils import get_class_name

CLASSIFIERS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, 'classifiers_outputs')
os.makedirs(CLASSIFIERS_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(CLASSIFIERS_OUTPUT_DIR, 'esm_classification_with_classic_ML.log'), mode='w'),
                        logging.StreamHandler()
                    ])

EMB_LAYER = 33

classifiers = [KNeighborsClassifier(), SVC(), RandomForestClassifier(), GradientBoostingClassifier(),
               LogisticRegression(), MLPClassifier()]

knn_grid = {
        'n_neighbors': [5, 10],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [15, 30],
        'p': [1, 2],
}

svm_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3],
        'gamma': ['scale'],
        'probability': [True],
}

rfc_grid = {
        'n_estimators': [20],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4]
}

gbc_grid = {
    "loss": ["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse",  "squared_error"],
    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators": [10]
}

logistic_regression_grid = {
    "penalty": ['l1', 'l2'],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

mlp_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'early_stopping': [True, False],
}

param_grid_list = [knn_grid, svm_grid, rfc_grid, gbc_grid, logistic_regression_grid, mlp_grid]


def read_embeddings_from_dir(dir_path):
    embeddings = []
    for file in os.listdir(dir_path):
        embeddings_object = torch.load(os.path.join(dir_path, file))
        embeddings.append(embeddings_object['mean_representations'][EMB_LAYER])
    Xs = torch.stack(embeddings, dim=0).numpy()
    return Xs


def prepare_Xs_and_Ys(pos_dir, neg_dir):
    Xs_positive = read_embeddings_from_dir(pos_dir)
    Xs_negative = read_embeddings_from_dir(neg_dir)
    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    # Shuffle
    combined = list(zip(Xs, Ys))
    random.shuffle(combined)
    shuffled_Xs, shuffled_Ys = zip(*combined)
    shuffled_Xs = np.array(shuffled_Xs)

    return shuffled_Xs, shuffled_Ys


def pca(Xs, Ys, n_components=2):
    pca = PCA(n_components=n_components)
    Xs_pca = pca.fit_transform(Xs)

    fig_dims = (7, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sc = ax.scatter(Xs_pca[:, 0], Xs_pca[:, 1], c=Ys, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    fig.colorbar(sc, label='Class')
    fig.savefig(os.path.join(OUTPUTS_DIR, 'pca.png'))


def main():
    Xs_train, Ys_train = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_TRAIN_DIR, EMBEDDINGS_NEGATIVE_TRAIN_DIR)
    logging.info(f"Loadded train data: Xs_train.shape = {Xs_train.shape}, Ys_train.shape = {len(Ys_train)}")
    Xs_test, Ys_test = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_TEST_DIR, EMBEDDINGS_NEGATIVE_TEST_DIR)
    logging.info(f"Loadded test data: Xs_test.shape = {Xs_test.shape}, Ys_test.shape = {len(Ys_test)}")
    Xs_xantomonas, Ys_xantomonas = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_XANTOMONAS_DIR, EMBEDDINGS_NEGATIVE_XANTOMONAS_DIR)
    logging.info(f"Loadded xantomonas data: Xs_xantomonas.shape = {Xs_xantomonas.shape}, Ys_xantomonas.shape = {len(Ys_xantomonas)}")

    pca(Xs_train, Ys_train)

    best_classifiers_scores = {}
    for classifier, param_grid in zip(classifiers, param_grid_list):
        class_name = get_class_name(classifier)
        logging.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
        grid = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring={'mcc': make_scorer(matthews_corrcoef), 'auprc': 'average_precision'},
            refit='mcc',
            verbose=1,
            n_jobs=-1  # use all available cores
        )
        grid.fit(Xs_train, Ys_train)
        grid_results = pd.DataFrame.from_dict(grid.cv_results_)
        grid_results.to_csv(os.path.join(CLASSIFIERS_OUTPUT_DIR, f'{class_name}_grid_results.csv'))

        logging.info(f"Best estimator: {grid.best_estimator_}, Best params: {grid.best_params_}, "
                     f"Best index: {grid.best_index_}, Best score: {grid.best_score_}, Classes: {grid.classes_}")

        logging.info(f"Best estimator - Mean MCC on held-out folds: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                     f"Mean AUPRC on held-out folds: {grid_results['mean_test_auprc'][grid.best_index_]}")

        best_classifiers_scores[class_name] = grid.best_score_

        mcc_on_test = grid.score(Xs_test, Ys_test)
        mcc_on_xantomonas = grid.score(Xs_xantomonas, Ys_xantomonas)
        auprc_on_test = average_precision_score(Ys_test, grid.predict_proba(Xs_test)[:, 1])
        auprc_on_xantomonas = average_precision_score(Ys_xantomonas, grid.predict_proba(Xs_xantomonas)[:, 1])
        logging.info(f"Best estimator - MCC on test: {mcc_on_test}, MCC on xantomonas: {mcc_on_xantomonas}, "
                     f"AUPRC on test: {auprc_on_test}, AUPRC on xantomonas: {auprc_on_xantomonas}")

    logging.info(f"Best classifiers scores (on held-out validation folds): {best_classifiers_scores}")


if __name__ == "__main__":
    main()
