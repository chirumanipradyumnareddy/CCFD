import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

print("Starting universal dataset split...")

# ===============================
# LOAD DATASET (CHANGE FILE NAME HERE IF NEEDED)
# ===============================
DATASET_NAME = "creditcard_2023.csv"   # Change this when using new dataset

df = pd.read_csv(DATASET_NAME)

# ===============================
# UNIVERSAL PREPROCESSING
# ===============================

# Drop ID-like columns
for col in df.columns:
    if col.lower() in ["id", "transactionid", "index"]:
        df = df.drop(col, axis=1)

# Auto-detect target column
possible_targets = ["class", "label", "isfraud", "target", "fraud"]
target_col = None

for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

if target_col is None:
    raise Exception("No valid target column found.")

df = df.rename(columns={target_col: "Class"})

# Convert categorical target to numeric if needed
if df["Class"].dtype == "object":
    df["Class"] = df["Class"].astype("category").cat.codes

# Fill missing values
df = df.fillna(0)

# Encode categorical features
for col in df.columns:
    if df[col].dtype == "object":
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# ===============================
# STRATIFIED FEDERATED SPLIT
# ===============================

num_clients = 5
skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)

# Remove old clients folder if exists
if os.path.exists("clients"):
    import shutil
    shutil.rmtree("clients")

os.makedirs("clients", exist_ok=True)

for i, (_, test_index) in enumerate(skf.split(df, df["Class"])):
    client_df = df.iloc[test_index]

    bank_folder = f"clients/bank{i+1}"
    os.makedirs(bank_folder, exist_ok=True)

    client_df.to_csv(f"{bank_folder}/data.csv", index=False)

    fraud_count = client_df["Class"].sum()
    total = len(client_df)

    print(f"Created data for Bank {i+1} | Total: {total} | Fraud cases: {fraud_count}")

print("\nDataset successfully split into federated banks (Stratified).")
