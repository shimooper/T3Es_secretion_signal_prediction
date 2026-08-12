import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from effectidor2_paper.src.utils.consts_paths import FINAL_RESULTS, CLASSIFIERS_OUTPUT_DIR, FINETUNED_MODELS_OUTPUT_DIR
from common.consts import MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION


def main():
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)

    all_classic_results = []
    for model_id in MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION:
        model_results = pd.read_csv(CLASSIFIERS_OUTPUT_DIR / model_id / 'best_classifier_all_results.csv')
        all_classic_results.append(model_results)

    all_classic_results_df = pd.concat(all_classic_results)
    all_classic_results_df.to_csv(FINAL_RESULTS / 'all_classic_classifiers_results.csv', index=False)

    all_finetuned_results = []
    for model_id in MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION:
        model_results = pd.read_csv(FINETUNED_MODELS_OUTPUT_DIR / model_id / 'best_model_results.csv')
        all_finetuned_results.append(model_results)

    all_finetuned_results_df = pd.concat(all_finetuned_results)
    all_finetuned_results_df.to_csv(FINAL_RESULTS / 'all_finetuned_classifiers_results.csv', index=False)


if __name__ == '__main__':
    main()
