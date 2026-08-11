from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------- paths --------
root = Path(__file__).resolve().parent / 'results'
output_csv = root / "aggregated_elapsed_times.csv"
output_plot = root / "elapsed_time_barplot.png"

# -------- mapping --------
name_map = {
    "mixed_0.0": "pt5",
    "mixed_1.0": "esm_6",
}

# -------- desired column order --------
column_order = [
    "esm_6",
    "mixed_0.4",
    "mixed_0.3",
    "mixed_0.2",
    "mixed_0.1",
    "pt5",
]

rows = []

for sample_dir in sorted(root.iterdir(), key=lambda p: int(p.name)):
    if not sample_dir.is_dir():
        continue

    sample_id = sample_dir.name
    row = {"sample": sample_id}

    for mixed_dir in sample_dir.iterdir():
        if not mixed_dir.is_dir():
            continue

        csv_path = mixed_dir / "mixed_classifier_test_results.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        time_val = df.loc[0, "test_elapsed_time"]

        col_name = name_map.get(mixed_dir.name, mixed_dir.name)
        row[col_name] = time_val

    rows.append(row)

# -------- wide dataframe --------
wide_df = pd.DataFrame(rows).set_index("sample")

# enforce column order (only keep existing columns)
wide_df = wide_df[[c for c in column_order if c in wide_df.columns]]

# -------- save combined CSV --------
wide_df.to_csv(output_csv)
print(f"Saved CSV to: {output_csv}")

# -------- long format for plotting --------
long_df = (
    wide_df
    .reset_index()
    .melt(id_vars="sample", var_name="model", value_name="elapsed_time")
)

# enforce plot ordering explicitly
long_df["model"] = pd.Categorical(
    long_df["model"],
    categories=column_order,
    ordered=True
)

long_df = long_df.sort_values(["model", "sample"])

# -------- bar plot --------
plt.figure(figsize=(10, 6))

bar_width = 0.8 / long_df["sample"].nunique()
samples = sorted(long_df["sample"].unique())

x = range(len(column_order))

for i, sample in enumerate(samples):
    df_s = long_df[long_df["sample"] == sample]
    plt.bar(
        [xi + i * bar_width for xi in x],
        df_s["elapsed_time"],
        width=bar_width,
        label=f"sample {sample}"
    )

plt.xticks(
    [xi + bar_width * (len(samples) - 1) / 2 for xi in x],
    column_order
)

plt.xlabel("Model")
plt.ylabel("Test elapsed time (seconds, using 10 cpus)")
plt.title("Test elapsed time per sample and model")
plt.legend(title="Sample")
plt.tight_layout()

# -------- save plot --------
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"Saved plot to: {output_plot}")
