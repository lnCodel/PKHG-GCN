import pandas as pd

# ==========================
# 1. Read CSV files and merge them by rows (stacking)
# ==========================
csv_files = [
    "M1_common_features_top200.csv",
    "M2_common_features_top20.csv",
    "I_common_features_top20.csv",
    "M4_common_features_top20.csv",
]

dfs = []
for f in csv_files:
    # Load each CSV file into a DataFrame
    df = pd.read_csv(f)
    dfs.append(df)

# Concatenate all DataFrames along rows
all_features_df = pd.concat(dfs, axis=0, ignore_index=True)

print(f"Total features after merge: {all_features_df.shape[0]}")

# ==========================
# 2. For each model, sort features and select Top-N
# ==========================
# Identify model score columns
model_cols = [c for c in all_features_df.columns if c.startswith("Model_")]

topN = 30
top_features_per_model = {}

for model in model_cols:
    # Sort by model importance score and take Top-N features
    top_feats = (
        all_features_df
        .sort_values(by=model, ascending=False)
        .head(topN)["feature"]
        .tolist()
    )
    top_features_per_model[model] = top_feats

    print(f"\n{model} Top-{topN} features:")
    for f in top_feats:
        print(f"  {f}")

# ==========================
# 3. Compute the intersection of Top-N features across models
# ==========================
top_sets = [set(v) for v in top_features_per_model.values()]
common_top_features = set.intersection(*top_sets)

print("\n==========================")
print(f"Number of common features in Top-{topN}: {len(common_top_features)}/{topN}")
print("==========================")

import matplotlib.pyplot as plt

# ==========================
# 4. Compute mean and standard deviation of importance scores
# ==========================
if len(common_top_features) == 0:
    print("No common features found. Stop.")
    exit()

# Filter rows corresponding to common features
df_common = all_features_df[
    all_features_df["feature"].isin(common_top_features)
].copy()

# Compute mean importance across models
df_common["mean_importance"] = df_common[model_cols].mean(axis=1)

# Compute standard deviation across models
df_common["std_importance"] = df_common[model_cols].std(axis=1)

# Sort features by mean importance
df_common = df_common.sort_values(
    by="mean_importance",
    ascending=False
)

print(f"\nPlotting {df_common.shape[0]} common Top-{topN} features")

import numpy as np

# ==========================
# 5. Visualization using a horizontal bar chart
# ==========================
# Adjust figure height based on number of features
plt.figure(figsize=(10, 0.5 * len(df_common) + 2))

# Y-axis positions
y_pos = np.arange(len(df_common))

# Plot mean importance with error bars (standard deviation)
plt.barh(
    y_pos,
    df_common["mean_importance"],
    xerr=df_common["std_importance"],
    height=0.55,
    capsize=4,
    error_kw={
        "elinewidth": 1,
        "ecolor": "black"
    }
)

# Set feature names as y-axis labels
plt.yticks(
    y_pos,
    df_common["feature"],
    fontsize=9
)

# Label x-axis
plt.xlabel("Mean Feature Importance", fontsize=12)

# Place the most important features at the top
plt.gca().invert_yaxis()

# Add grid lines only along the x-axis
plt.grid(axis="x", linestyle="--", alpha=0.4)

# Adjust layout and save the figure
plt.tight_layout()
output_path = "Top30_overlapping_features_mean_std.jpg"
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
