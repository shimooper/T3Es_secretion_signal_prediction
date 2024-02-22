import random
from collections import Counter
from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, matthews_corrcoef

from consts import (EMBEDDINGS_DIR, EMBEDDINGS_POSITIVE_TRAIN_DIR, EMBEDDINGS_NEGATIVE_TRAIN_DIR,
                    EMBEDDINGS_POSITIVE_TEST_DIR, EMBEDDINGS_NEGATIVE_TEST_DIR,
                    EMBEDDINGS_POSITIVE_XANTOMONAS_DIR, EMBEDDINGS_NEGATIVE_XANTOMONAS_DIR)

EMB_LAYER = 33

knn_grid = [
    {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': [5, 10],
        'model__weights': ['uniform', 'distance'],
        'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'model__leaf_size' : [15, 30],
        'model__p' : [1, 2],
    }
    ]

svm_grid = [
    {
        'model': [SVC()],
        'model__C' : [0.1, 1.0, 10.0],
        'model__kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'model__degree' : [3],
        'model__gamma': ['scale'],
    }
]

rfr_grid = [
    {
        'model': [RandomForestClassifier()],
        'model__n_estimators' : [20],
        'model__criterion' : ['gini', 'entropy'],
        'model__max_features': ['sqrt', 'log2'],
        'model__min_samples_split' : [2, 5, 10],
        'model__min_samples_leaf': [1, 4]
    }
]


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
    fig.savefig(os.path.join(EMBEDDINGS_DIR, 'pca.png'))


def main():
    Xs_train, Ys_train = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_TRAIN_DIR, EMBEDDINGS_NEGATIVE_TRAIN_DIR)
    Xs_test, Ys_test = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_TEST_DIR, EMBEDDINGS_NEGATIVE_TEST_DIR)
    Xs_xantomonas, Ys_xantomonas = prepare_Xs_and_Ys(EMBEDDINGS_POSITIVE_XANTOMONAS_DIR, EMBEDDINGS_NEGATIVE_XANTOMONAS_DIR)

    pca(Xs_train, Ys_train)

    cls_list = [KNeighborsClassifier, SVC, RandomForestClassifier]
    param_grid_list = [knn_grid, svm_grid, rfr_grid]

    pipe = Pipeline(
        steps=[
            ('model', 'passthrough')
        ]
    )

    mcc_scorer = make_scorer(matthews_corrcoef)

    result_list = []
    grid_list = []
    for cls_name, param_grid in zip(cls_list, param_grid_list):
        print(cls_name)
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring=mcc_scorer,
            verbose=1,
            n_jobs=-1  # use all available cores
        )
        grid.fit(Xs_train, Ys_train)
        result_list.append(pd.DataFrame.from_dict(grid.cv_results_))
        grid_list.append(grid)

    pass

if __name__ == "__main__":
    main()
