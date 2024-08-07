import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import FINAL_RESULTS

CLASSIC_CLASSIFIERS_PATH = Path(FINAL_RESULTS) / 'all_classic_classifiers_results.csv'
FINETUNED_CLASSIFIERS_PATH = Path(FINAL_RESULTS) / 'all_finetuned_classifiers_results.csv'

ADD_BASELINES_TO_FIG = True


def main():
    classic_classifiers_df = pd.read_csv(CLASSIC_CLASSIFIERS_PATH)
    finetuned_classifiers_df = pd.read_csv(FINETUNED_CLASSIFIERS_PATH)

    classic_classifiers_df.rename(columns={'mean_mcc_on_train_folds': 'train_mcc',
                                           'mean_auprc_on_train_folds': 'train_auprc',
                                           'mean_mcc_on_held_out_folds': 'validation_mcc',
                                           'mean_auprc_on_held_out_folds': 'validation_auprc'}, inplace=True)
    classic_classifiers_df.drop(columns=['classifier_class', 'best_index'], inplace=True)

    all_results_df = pd.concat([classic_classifiers_df, finetuned_classifiers_df], ignore_index=True)

    all_results_df.sort_values('number_of_parameters (millions)', inplace=True)
    all_results_df.to_csv(Path(FINAL_RESULTS) / 'all_results.csv', index=False)

    fig, axs = plt.subplots(4, 3, figsize=(40, 35))

    sns.stripplot(data=all_results_df, x='model_id', y='train_mcc', hue='training_mode', ax=axs[0, 0])
    sns.stripplot(data=all_results_df, x='model_id', y='train_auprc', hue='training_mode', ax=axs[0, 1])
    axs[0, 2].set_visible(False)
    sns.stripplot(data=all_results_df, x='model_id', y='validation_mcc', hue='training_mode', ax=axs[1, 0])
    sns.stripplot(data=all_results_df, x='model_id', y='validation_auprc', hue='training_mode', ax=axs[1, 1])
    axs[1, 2].set_visible(False)

    sns.stripplot(data=all_results_df, x='model_id', y='test_mcc', hue='training_mode', ax=axs[2, 0])
    if ADD_BASELINES_TO_FIG:
        axs[2, 0].axhline(y=0.81, color='r', linestyle='--', linewidth=1, label='y=0.81')
        axs[2, 0].axhline(y=0.83, color='r', linestyle='--', linewidth=1, label='y=0.83')
    sns.stripplot(data=all_results_df, x='model_id', y='test_auprc', hue='training_mode', ax=axs[2, 1])
    if ADD_BASELINES_TO_FIG:
        axs[2, 1].axhline(y=0.88, color='r', linestyle='--', linewidth=1, label='y=0.88')
        axs[2, 1].axhline(y=0.91, color='r', linestyle='--', linewidth=1, label='y=0.91')
    sns.stripplot(data=all_results_df, x='model_id', y='test_elapsed_time', hue='training_mode', ax=axs[2, 2])
    axs[2, 2].set_ylabel('Test Elapsed time (in seconds)')

    sns.stripplot(data=all_results_df, x='model_id', y='xantomonas_mcc', hue='training_mode', ax=axs[3, 0])
    if ADD_BASELINES_TO_FIG:
        axs[3, 0].axhline(y=0.71, color='r', linestyle='--', linewidth=1, label='y=0.71')
        axs[3, 0].axhline(y=0.72, color='r', linestyle='--', linewidth=1, label='y=0.72')
    sns.stripplot(data=all_results_df, x='model_id', y='xantomonas_auprc', hue='training_mode', ax=axs[3, 1])
    if ADD_BASELINES_TO_FIG:
        axs[3, 1].axhline(y=0.77, color='r', linestyle='--', linewidth=1, label='y=0.77')
        axs[3, 1].axhline(y=0.87, color='r', linestyle='--', linewidth=1, label='y=0.87')
    sns.stripplot(data=all_results_df, x='model_id', y='xantomonas_elapsed_time', hue='training_mode', ax=axs[3, 2])
    axs[3, 2].set_ylabel('Xantomonas Elapsed time (in seconds)')

    fig.text(0.06, 0.80, 'Train', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.60, 'Validation', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.40, 'Test', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.20, 'Test Xantomonas', va='center', rotation='vertical', fontsize=14)

    fig.text(0.25, 0.90, 'MCC', ha='center', fontsize=14)
    fig.text(0.5, 0.90, 'AUPRC', ha='center', fontsize=14)
    fig.text(0.75, 0.90, 'Elapsed time (in seconds)', ha='center', fontsize=14)

    if ADD_BASELINES_TO_FIG:
        plt.savefig(Path(FINAL_RESULTS) / 'results_with_hlines.png')
    else:
        plt.savefig(Path(FINAL_RESULTS) / 'results.png')


if __name__ == '__main__':
    main()
