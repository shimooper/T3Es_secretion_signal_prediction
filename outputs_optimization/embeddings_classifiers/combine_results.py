import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# Mapping: directory -> filename
dir_file_map = {
    "esm_6": "best_classifier_all_results.csv",
    "mixed_0.1": "mixed_classifier_test_results.csv",
    "mixed_0.2": "mixed_classifier_test_results.csv",
    "mixed_0.3": "mixed_classifier_test_results.csv",
    "mixed_0.4": "mixed_classifier_test_results.csv",
    "pt5": "best_classifier_all_results.csv",
}

model_order = ["esm_6", "mixed_0.4", "mixed_0.3", "mixed_0.2", "mixed_0.1", "pt5"]
metrics = ["test_mcc", "test_auprc", "test_elapsed_time"]

# ---------- Load & combine ----------
rows = []

for model_name, filename in dir_file_map.items():
    path = BASE_DIR / model_name / filename
    df = pd.read_csv(path)

    # each CSV has exactly one row
    row = df.iloc[0][metrics].copy()
    row["model"] = model_name
    rows.append(row)

df = pd.DataFrame(rows)

# enforce model order
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
df = df.sort_values("model")

df.to_csv(BASE_DIR / "combined_model_results.csv", index=False)


# ---------- Plot: one figure, two subplots ----------
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 6),
    sharex=True
)

# Top: MCC & AUPRC
ax1.plot(df["model"], df["test_mcc"], marker="o", label="MCC")
ax1.plot(df["model"], df["test_auprc"], marker="o", label="AUPRC")
ax1.set_ylabel("Score")
ax1.set_title("Model Comparison")
ax1.legend()
ax1.grid(True, axis="y", alpha=0.3)

# Bottom: elapsed time
ax2.bar(df["model"], df["test_elapsed_time"])
ax2.set_ylabel("Elapsed Time (using 10 cpus)")
ax2.set_xlabel("Model")
ax2.grid(True, axis="y", alpha=0.3)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig("model_comparison_all_metrics.png", dpi=300)
plt.close()

