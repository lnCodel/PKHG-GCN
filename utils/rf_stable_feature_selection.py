import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import csv

# Mapping from region name to row index in the feature CSV files
MAP = {
    "M1": 0, "M2": 1, "M3": 2, "M4": 3, "M5": 4,
    "M6": 5, "L": 6, "I": 7, "C": 8, "IC": 9
}

# ==========================
# 1. Random Forest parameter configurations
# ==========================
# Each parameter set defines an independent Random Forest model
param_sets = [
    {'n_estimators': 500, 'criterion': 'entropy'},
    {'n_estimators': 350, 'criterion': 'gini'},
    {'n_estimators': 1000, 'criterion': 'gini'},
    {'n_estimators': 500, 'criterion': 'gini'},
    {'n_estimators': 700, 'criterion': 'gini'}
]

# ==========================
# 2. Data loading utilities
# ==========================
def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def get_ids(path=r"D:\Users\00807908\Desktop\TMI\subjects.txt"):
    """
    Load subject IDs from a text file.
    """
    return np.genfromtxt(path, dtype=str)


def get_subject_label(subject_list, score="M1"):
    """
    Retrieve subject labels for a given score type.
    """
    label_path = r"D:\Users\00807908\Desktop\TMI\labels_best.csv"
    score_dict = {}

    with open(label_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] in subject_list:
                score_dict[row["id"]] = abs(int(row[score]))

    return np.array([score_dict[subj] for subj in subject_list])


def load_radiomics_data(normalize=True, region="M1"):
    """
    Load radiomics features and labels for a specific region.
    """
    subject_IDs = get_ids()
    feature_dir = r"D:\Users\00807908\Desktop\TMI\features"

    # Read the first file to obtain feature names
    first_data = load_data(os.path.join(feature_dir, f"{subject_IDs[0]}.csv"))
    feature_names = first_data.columns.tolist()[40:]

    X = []
    for subj in subject_IDs:
        data = load_data(os.path.join(feature_dir, f"{subj}.csv"))
        # Select region-specific feature row and feature columns
        X.append(data.iloc[MAP[region], 40:].values)

    X = np.array(X, dtype=np.float32)

    # Apply standard normalization if enabled
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    y = get_subject_label(subject_IDs)
    return X, y, feature_names

# ==========================
# 3. Core analysis logic and CSV export
# ==========================
def analyze_common_feature_ranking(
    X,
    y,
    feature_names,
    param_sets,
    region="M1",
    top_k=200,
    output_csv="M1_common_features_top200.csv"
):
    """
    Train multiple Random Forest models, identify common Top-K features,
    and export feature importance statistics to a CSV file.
    """
    model_topK_sets = {}
    model_importances_full = {}

    # Step 1: Train Random Forest models
    for idx, params in enumerate(param_sets):
        tag = f"Model_{idx + 1}"

        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Extract feature importance scores
        importances = rf.feature_importances_
        importance_dict = {
            feature_names[i]: importances[i]
            for i in range(len(feature_names))
        }
        model_importances_full[tag] = importance_dict

        # Rank features by importance
        sorted_features = sorted(
            importance_dict,
            key=importance_dict.get,
            reverse=True
        )

        # Store Top-K feature set for this model
        model_topK_sets[tag] = set(sorted_features[:top_k])
        print(f"{tag}: extracted Top-{top_k} features")

    # Step 2: Compute intersection of Top-K features across models
    common_features = set.intersection(*model_topK_sets.values())
    print(f"\nCommon features in Top-{top_k}: {len(common_features)}")

    # ==========================
    # Step 3: Build CSV records
    # ==========================
    records = []

    for feat in common_features:
        row = {
            "feature": f"{region}_{feat}",
            "region": region
        }

        importances = []
        for tag in model_importances_full:
            val = model_importances_full[tag][feat]
            row[tag] = val
            importances.append(val)

        # Compute mean and standard deviation across models
        row["mean_importance"] = np.mean(importances)
        row["std_importance"] = np.std(importances)

        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values("mean_importance", ascending=False)

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved to: {output_csv}")

    return df

# ==========================
# 4. Main execution pipeline
# ==========================
if __name__ == "__main__":
    # Region-specific Top-K settings
    MA = {"M1": 200, "M2": 20, "I": 20, "M4": 20}

    for region in ["M1", "M2", "M4", "I"]:
        # Load data for the current region
        X, y, feature_names = load_radiomics_data(region=region)

        # Perform common feature ranking analysis
        df_common = analyze_common_feature_ranking(
            X=X,
            y=y,
            feature_names=feature_names,
            param_sets=param_sets,
            region=region,
            top_k=MA[region],
            output_csv=f"{region}_common_features_top200.csv"
        )

        # Preview the saved CSV content
        print("\n===== Preview of saved CSV =====")
        print(df_common.head())
