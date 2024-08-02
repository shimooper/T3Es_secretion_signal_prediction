from collections import Counter

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 500

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
        'random_state': [RANDOM_STATE],
        'class_weight': ['balanced', None],
}

logistic_regression_grid = {
    "solver": ['liblinear'],
    "penalty": ['l2'],
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "max_iter": [500],
    'random_state': [RANDOM_STATE],
    'class_weight': ['balanced', None],
}

mlp_grid = {
    'hidden_layer_sizes': [(10,3),(30,5),(50,10), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [40],
    'random_state': [RANDOM_STATE],
}

rfc_grid = {
        'n_estimators': [20],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4],
        'random_state': [RANDOM_STATE],
        'class_weight': ['balanced', None],
}

gbc_grid = {
    "loss": ["log_loss"],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4],
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse",  "squared_error"],
    "n_estimators": [100],
    'random_state': [RANDOM_STATE],
}

xgboost_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [10, 50, 200],
    'max_depth': [3, 5, 10],
    'booster': ['gbtree', 'dart'],
    'n_jobs': [1],
    'random_state': [RANDOM_STATE],
    'subsample': [0.6, 0.8, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0, 0.5, 3],
}

lgbm_grid = {
    'learning_rate': [0.005, 0.05, 0.1],
    'n_estimators': [10, 50, 200],
    'num_leaves': [10, 50, 100],  # large num_leaves helps improve accuracy but might lead to over-fitting
    'max_depth': [-1, 10, 100],
    'boosting_type': ['gbdt', 'dart'],  # for better accuracy -> try dart
    'objective': ['binary'],
    'max_bin': [255, 510],  # large max_bin helps improve accuracy but might slow down training progress
    'random_state': [RANDOM_STATE],
    'n_jobs': [1],
    'subsample': [0.6, 0.8, 1],
    'reg_alpha': [0, 0.5, 1, 10],
    'reg_lambda': [0, 0.5, 3, 100, 1000],
    'is_unbalance': [True, False],
}

classifiers = [
    (KNeighborsClassifier(), knn_grid),
    (SVC(), svm_grid),
    (LogisticRegression(), logistic_regression_grid),
    (MLPClassifier(), mlp_grid),
    (RandomForestClassifier(), rfc_grid),
    # (GradientBoostingClassifier(), gbc_grid),
    (XGBClassifier(), xgboost_grid),
    # (LGBMClassifier(), lgbm_grid)
]


def update_grid_params(train_labels):
    # See https://xgboost.readthedocs.io/en/stable/parameter.html - 'scale_pos_weight' parameter guidelines
    scale_pos_weight = Counter(train_labels)[0] / Counter(train_labels)[1]
    xgboost_grid['scale_pos_weight'] = [1, scale_pos_weight]
