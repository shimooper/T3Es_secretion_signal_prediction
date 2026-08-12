import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from runtime_optimization.src.utils.consts_paths import CLASSIFIERS_OUTPUT_DIR

RESULTS_DIR = Path(CLASSIFIERS_OUTPUT_DIR)

metrics = ["test_mcc", "test_auprc", "test_elapsed_time"]
metric_labels = {
    "test_mcc": "MCC",
    "test_auprc": "AUPRC",
    "test_elapsed_time": "Elapsed Time (s, 10 cpus)",
}
# For elapsed time lower is better, so reverse the colormap
metric_cmaps = {
    "test_mcc": "RdYlGn",
    "test_auprc": "RdYlGn",
    "test_elapsed_time": "RdYlGn_r",
}
metric_fmts = {
    "test_mcc": ".3f",
    "test_auprc": ".3f",
    "test_elapsed_time": ".1f",
}

# ---------- Load ----------
rows = []

for d in RESULTS_DIR.iterdir():
    if not d.is_dir():
        continue

    m = re.fullmatch(r"mixed_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)", d.name)
    if not m:
        continue

    csv_path = d / "mixed_classifier_test_results.csv"
    if not csv_path.exists():
        continue

    lower, upper = float(m.group(1)), float(m.group(2))
    df = pd.read_csv(csv_path)
    row = df.iloc[0][metrics].copy()
    row["lower"] = lower
    row["upper"] = upper
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(RESULTS_DIR / "combined_model_results.csv", index=False)

# ---------- Highlight mask: elapsed time <= 25% of PT5-only run ----------
pt5_time = df.loc[(df["lower"] == 0.0) & (df["upper"] == 1.0), "test_elapsed_time"].iloc[0]
elapsed_pivot_raw = df.pivot(index="lower", columns="upper", values="test_elapsed_time")
elapsed_pivot_raw = elapsed_pivot_raw.sort_index(ascending=False)
highlight = (elapsed_pivot_raw <= pt5_time * 0.25).fillna(False)

# ---------- Heatmaps ----------
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, metric in zip(axes, metrics):
    pivot = df.pivot(index="lower", columns="upper", values=metric)
    # Lower threshold descending (largest at top) so ESM-6-only (0.5/0.5) is top-left
    pivot = pivot.sort_index(ascending=False)
    pivot.columns = [f"{c:.1f}" for c in sorted(pivot.columns)]
    pivot.index = [f"{i:.1f}" for i in sorted(pivot.index, reverse=True)]

    mask = pivot.isna()

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=metric_fmts[metric],
        mask=mask,
        cmap=metric_cmaps[metric],
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": metric_labels[metric]},
        annot_kws={"size": 8},
    )
    ax.set_title(metric_labels[metric], fontsize=13, fontweight="bold")
    ax.set_xlabel("Upper Threshold", fontsize=10)
    ax.set_ylabel("Lower Threshold", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Highlight cells where elapsed time <= 25% of PT5-only run
    for i in range(highlight.shape[0]):
        for j in range(highlight.shape[1]):
            if highlight.iloc[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=False, edgecolor="blue", lw=2.5, clip_on=False,
                ))

plt.suptitle(
    "Mixed ESM-6 / PT5 Cascade — Threshold Grid\n"
    "(top-left = ESM-6 only, bottom-right = PT5 only)",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison_heatmaps.png", dpi=600, bbox_inches="tight")
plt.savefig(RESULTS_DIR / "model_comparison_heatmaps.svg", dpi=600, bbox_inches="tight")
plt.close()
