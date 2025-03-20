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

ADD_BASELINES_TO_FIG = False


def plot_full(all_results_df):
    fig, axs = plt.subplots(3, 3, figsize=(35, 35))

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

    fig.text(0.06, 0.75, 'Train', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.50, 'Validation', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.25, 'Test', va='center', rotation='vertical', fontsize=14)

    fig.text(0.25, 0.90, 'MCC', ha='center', fontsize=14)
    fig.text(0.5, 0.90, 'AUPRC', ha='center', fontsize=14)
    fig.text(0.75, 0.90, 'Elapsed time (in seconds)', ha='center', fontsize=14)

    if ADD_BASELINES_TO_FIG:
        plt.savefig(Path(FINAL_RESULTS) / 'results_with_hlines.png')
    else:
        plt.savefig(Path(FINAL_RESULTS) / 'results.png')


def plot_for_paper(all_results_df):
    plt.rcParams.update({
        "font.size": 22,  # Base font size
        "axes.titlesize": 34,  # Title size
        "axes.labelsize": 22,  # X & Y label size
        "axes.labelweight": "bold",  # X & Y label bold
        "xtick.labelsize": 22,  # X-axis tick size
        "ytick.labelsize": 22,  # Y-axis tick size
        # "font.weight": "bold"  # Global font weight
    })
    fig, axs = plt.subplots(1, 3, figsize=(30, 12))

    sns.stripplot(data=all_results_df, x='model_id', y='test_mcc', hue='training_mode', ax=axs[0], size=10)
    if ADD_BASELINES_TO_FIG:
        axs[0].axhline(y=0.81, color='r', linestyle='--', linewidth=1, label='y=0.81')
        axs[0].axhline(y=0.83, color='r', linestyle='--', linewidth=1, label='y=0.83')

    sns.stripplot(data=all_results_df, x='model_id', y='test_auprc', hue='training_mode', ax=axs[1], size=10)
    if ADD_BASELINES_TO_FIG:
        axs[1].axhline(y=0.88, color='r', linestyle='--', linewidth=1, label='y=0.88')
        axs[1].axhline(y=0.91, color='r', linestyle='--', linewidth=1, label='y=0.91')

    sns.stripplot(data=all_results_df, x='model_id', y='test_elapsed_time', hue='training_mode', ax=axs[2], size=10)
    axs[2].set_ylabel('Test Elapsed time (in seconds)')

    axs[0].set_title('MCC', fontsize=30)
    axs[1].set_title('AUPRC', fontsize=30)
    axs[2].set_title('Elapsed time (in seconds)', fontsize=30)

    # fig.text(0.2, 0.90, 'MCC', ha='center', fontsize=24)
    # fig.text(0.5, 0.90, 'AUPRC', ha='center', fontsize=24)
    # fig.text(0.8, 0.90, 'Elapsed time (in seconds)', ha='center', fontsize=24)

    for i in [0, 1, 2]:
        axs[i].tick_params(axis='x', rotation=45)

    plt.subplots_adjust(hspace=0.6)
    plt.tight_layout()

    if ADD_BASELINES_TO_FIG:
        plt.savefig(Path(FINAL_RESULTS, dpi=600) / 'results_with_hlines_for_paper.png')
    else:
        plt.savefig(Path(FINAL_RESULTS, dpi=600) / 'results_for_paper.png')


def main():
    all_results_path = Path(FINAL_RESULTS) / 'all_results.csv'
    if not all_results_path.is_file():
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
    else:
        all_results_df = pd.read_csv(all_results_path)

    plot_full(all_results_df)
    plot_for_paper(all_results_df)


if __name__ == '__main__':
    main()
