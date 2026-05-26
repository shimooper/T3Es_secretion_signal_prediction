import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.consts import CLASSIFIERS_OUTPUT_DIR

NUM_PT5_LAYERS = 24  # ProtT5 (T5-3B) has 24 encoder transformer blocks

OUTPUT_DIR = os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5_layers_expirement')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_layer_results(hidden_layer_number):
    """Load best_classifier_all_results.csv for a given layer (None = default full model)."""
    if hidden_layer_number is None:
        classifiers_dir = os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5')
        label = 'default\n(full model)'
    else:
        classifiers_dir = os.path.join(CLASSIFIERS_OUTPUT_DIR, 'pt5', f'layer_{hidden_layer_number}')
        label = str(hidden_layer_number)

    results_file = os.path.join(classifiers_dir, 'best_classifier_all_results.csv')
    if not os.path.exists(results_file):
        print(f"Missing: {results_file}")
        return None

    df = pd.read_csv(results_file)
    df['hidden_layer_number'] = hidden_layer_number
    df['label'] = label
    return df


def main():
    rows = []

    for layer in range(1, NUM_PT5_LAYERS + 1):
        result = load_layer_results(layer)
        if result is not None:
            rows.append(result)

    baseline = load_layer_results(None)

    if not rows:
        print("No layer results found. Have the experiment jobs finished?")
        return

    combined = pd.concat(rows, ignore_index=True)
    combined_with_baseline = pd.concat(rows + ([baseline] if baseline is not None else []), ignore_index=True)
    combined_with_baseline.to_csv(os.path.join(OUTPUT_DIR, 'pt5_layers_comparison.csv'), index=False)
    print(f"Saved pt5_layers_comparison.csv ({len(combined_with_baseline)} rows)")

    # --- plots ---
    layer_nums = combined['hidden_layer_number'].tolist()

    for metric_col, metric_label, filename in [
        ('test_mcc',   'MCC (test set)',   'pt5_layers_test_mcc.png'),
        ('test_auprc', 'AUPRC (test set)', 'pt5_layers_test_auprc.png'),
        ('mean_mcc_on_held_out_folds',   'MCC (CV held-out folds)',   'pt5_layers_cv_mcc.png'),
        ('mean_auprc_on_held_out_folds', 'AUPRC (CV held-out folds)', 'pt5_layers_cv_auprc.png'),
    ]:
        if metric_col not in combined.columns:
            print(f"Column '{metric_col}' not found, skipping plot.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(layer_nums, combined[metric_col].tolist(), marker='o', label='layer sweep')

        if baseline is not None and metric_col in baseline.columns:
            ax.axhline(y=baseline[metric_col].iloc[0], color='red', linestyle='--',
                       label=f'default full model ({baseline[metric_col].iloc[0]:.3f})')

        ax.set_xlabel('Encoder layer number')
        ax.set_ylabel(metric_label)
        ax.set_title(f'ProtT5 — {metric_label} by encoder layer')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
        plt.close(fig)
        print(f"Saved {filename}")

    # Print summary table
    summary_cols = ['hidden_layer_number', 'classifier_class',
                    'mean_mcc_on_held_out_folds', 'mean_auprc_on_held_out_folds',
                    'test_mcc', 'test_auprc', 'test_elapsed_time']
    present = [c for c in summary_cols if c in combined_with_baseline.columns]
    print("\n=== Summary ===")
    print(combined_with_baseline[present].to_string(index=False))


if __name__ == '__main__':
    main()
