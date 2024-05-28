import pandas as pd

CLASSIC_CLASSIFIERS_PATH = r'C:\repos\T3Es_secretion_signal_prediction\outputs\all_classic_classifiers_results.csv'
FINETUNED_CLASSIFIERS_PATH = r'C:\repos\T3Es_secretion_signal_prediction\outputs\all_finetuned_classifiers_results.csv'

classic_classifiers_df = pd.read_csv(CLASSIC_CLASSIFIERS_PATH)
finetuned_classifiers_df = pd.read_csv(FINETUNED_CLASSIFIERS_PATH)

pass