import os
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.consts import FINAL_RESULTS, CLASSIFIERS_OUTPUT_DIR, FINETUNED_MODELS_OUTPUT_DIR, MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION


def main():
    os.makedirs(FINAL_RESULTS, exist_ok=True)

    all_classic_results = []
    for model_id in MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION:
        model_results = pd.read_csv(os.path.join(CLASSIFIERS_OUTPUT_DIR, model_id, 'best_classifier_all_results.csv'))
        all_classic_results.append(model_results)

    all_classic_results_df = pd.concat(all_classic_results)
    all_classic_results_df.to_csv(os.path.join(FINAL_RESULTS, 'all_classic_classifiers_results_v2.csv'), index=False)

    all_finetuned_results = []
    for model_id in MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION:
        model_results = pd.read_csv(os.path.join(FINETUNED_MODELS_OUTPUT_DIR, model_id, 'best_model_results.csv'))
        all_finetuned_results.append(model_results)

    all_finetuned_results_df = pd.concat(all_finetuned_results)
    all_finetuned_results_df.to_csv(os.path.join(FINAL_RESULTS, 'all_finetuned_classifiers_results_v2.csv'), index=False)


if __name__ == '__main__':
    main()
