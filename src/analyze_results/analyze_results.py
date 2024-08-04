import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


CLASSIC_CLASSIFIERS_PATH = Path(OUTPUTS_DIR) / 'all_classic_classifiers_results.csv'
FINETUNED_CLASSIFIERS_PATH = Path(OUTPUTS_DIR) / 'all_finetuned_classifiers_results.csv'


def flatten(x):
    return [item for sublist in x for item in sublist]


def main():
    classic_classifiers_df = pd.read_csv(CLASSIC_CLASSIFIERS_PATH)
    finetuned_classifiers_df = pd.read_csv(FINETUNED_CLASSIFIERS_PATH)
    classic_classifiers_df['training_mode'] = 'only_head'
    classic_classifiers_df.rename(columns={'mean_mcc_on_train_folds': 'train_mcc',
                                           'mean_auprc_on_train_folds': 'train_auprc',
                                           'mean_mcc_on_held_out_folds': 'validation_mcc',
                                           'mean_auprc_on_held_out_folds': 'validation_auprc'}, inplace=True)

    finetuned_classifiers_df['training_mode'] = 'finetuned'
    all_results_df = pd.concat([classic_classifiers_df, finetuned_classifiers_df], ignore_index=True)

    all_results_df['Number of parameters (in million)'] = all_results_df['backbone'].apply(lambda x: MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[x])
    all_results_df.sort_values('Number of parameters (in million)', inplace=True)
    all_results_df['Model name'] = all_results_df['backbone'].apply(lambda x: MODEL_ID_TO_MODEL_NAME[x])
    all_results_df.to_csv(Path('../../outputs') / 'all_results.csv', index=False)

    fig, axs = plt.subplots(4, 3, figsize=(40, 35))

    sns.stripplot(data=all_results_df, x='Model name', y='train_mcc', hue='training_mode', ax=axs[0, 0])
    sns.stripplot(data=all_results_df, x='Model name', y='train_auprc', hue='training_mode', ax=axs[0, 1])
    axs[0, 2].set_visible(False)
    sns.stripplot(data=all_results_df, x='Model name', y='validation_mcc', hue='training_mode', ax=axs[1, 0])
    sns.stripplot(data=all_results_df, x='Model name', y='validation_auprc', hue='training_mode', ax=axs[1, 1])
    axs[1, 2].set_visible(False)

    sns.stripplot(data=all_results_df, x='Model name', y='test_mcc', hue='training_mode', ax=axs[2, 0])
    sns.stripplot(data=all_results_df, x='Model name', y='test_auprc', hue='training_mode', ax=axs[2, 1])
    sns.stripplot(data=all_results_df, x='Model name', y='test_elapsed_time', hue='training_mode', ax=axs[2, 2])
    axs[2, 2].set_ylabel('Test Elapsed time (in seconds)')

    sns.stripplot(data=all_results_df, x='Model name', y='xantomonas_mcc', hue='training_mode', ax=axs[3, 0])
    sns.stripplot(data=all_results_df, x='Model name', y='xantomonas_auprc', hue='training_mode', ax=axs[3, 1])
    sns.stripplot(data=all_results_df, x='Model name', y='xantomonas_elapsed_time', hue='training_mode', ax=axs[3, 2])
    axs[3, 2].set_ylabel('Xantomonas Elapsed time (in seconds)')

    fig.text(0.06, 0.80, 'Train', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.60, 'Validation', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.40, 'Test', va='center', rotation='vertical', fontsize=14)
    fig.text(0.06, 0.20, 'Test Xantomonas', va='center', rotation='vertical', fontsize=14)

    fig.text(0.25, 0.90, 'MCC', ha='center', fontsize=14)
    fig.text(0.5, 0.90, 'AUPRC', ha='center', fontsize=14)
    fig.text(0.75, 0.90, 'Elapsed time (in seconds)', ha='center', fontsize=14)

    plt.savefig(Path(OUTPUTS_DIR) / 'results_2.png')
    plt.clf()


if __name__ == '__main__':
    main()
