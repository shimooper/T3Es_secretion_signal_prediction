import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

OUTPUTS_PATH = '/groups/pupko/yairshimony/secretion_signal_prediction/outputs'

CLASSIC_CLASSIFIERS_PATH = os.path.join(OUTPUTS_PATH, 'embeddings_classifiers')
CLASSIC_CLASSIFIERS_RESULTS = {
    '6': os.path.join(CLASSIC_CLASSIFIERS_PATH, '6', 'best_classifier_results.csv'),
    '12': os.path.join(CLASSIC_CLASSIFIERS_PATH, '12', 'best_classifier_results.csv'),
    '30': os.path.join(CLASSIC_CLASSIFIERS_PATH, '30', 'best_classifier_results.csv'),
    '33': os.path.join(CLASSIC_CLASSIFIERS_PATH, '33', 'best_classifier_results.csv'),
    '36': os.path.join(CLASSIC_CLASSIFIERS_PATH, '36', 'best_classifier_results.csv'),
    '48': os.path.join(CLASSIC_CLASSIFIERS_PATH, '48', 'best_classifier_results.csv'),
    'protein_bert': os.path.join(CLASSIC_CLASSIFIERS_PATH, 'protein_bert', 'best_classifier_results.csv'),
}

ESM_FINETUNED_CLASSIFIERS_PATH = os.path.join(OUTPUTS_PATH, 'esm_finetune_runs')
FINETUNED_CLASSIFIERS_RESULTS = {
    '6': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t6_8M_UR50D-3-epochs', 'esm_finetune_results.csv'),
    '12': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t12_35M_UR50D-3-epochs', 'esm_finetune_results.csv'),
    '30': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t30_150M_UR50D-3-epochs', 'esm_finetune_results.csv'),
    '33': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t33_650M_UR50D-3-epochs', 'esm_finetune_results.csv'),
    '36': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t36_3B_UR50D-3-epochs', 'esm_finetune_results.csv'),
    '48': os.path.join(ESM_FINETUNED_CLASSIFIERS_PATH, 'esm2_t48_15B_UR50D-3-epochs', 'esm_finetune_results.csv'),
    'protein_bert': os.path.join(OUTPUTS_PATH, 'protein_bert_finetune', 'esm_finetune_results_3_epochs.csv'),
}


def main():
    all_classic_results = []
    for backbone, results_path in CLASSIC_CLASSIFIERS_RESULTS.items():
        results = pd.read_csv(results_path)
        results['backbone'] = backbone
        all_classic_results.append(results)

    all_results_df = pd.concat(all_classic_results)
    all_results_df.to_csv(os.path.join(OUTPUTS_PATH, 'all_classic_classifiers_results.csv'), index=False)

    all_finetuned_results = []
    for backbone, results_path in FINETUNED_CLASSIFIERS_RESULTS.items():
        results = pd.read_csv(results_path)
        results['backbone'] = backbone
        all_finetuned_results.append(results)

    all_finetuned_results_df = pd.concat(all_finetuned_results)
    all_finetuned_results_df.to_csv(os.path.join(OUTPUTS_PATH, 'all_finetuned_classifiers_results.csv'), index=False)


if __name__ == '__main__':
    main()
