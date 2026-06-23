import pandas as pd

def preprocess_dataset(path):
    data = pd.read_csv(path)

    # Drop ID-like columns
    for col in data.columns:
        if col.lower() in ["id", "transactionid", "index"]:
            data = data.drop(col, axis=1)

    # Auto-detect target column
    possible_targets = ["class", "label", "isfraud", "target", "fraud"]
    target_col = None

    for col in data.columns:
        if col.lower() in possible_targets:
            target_col = col
            break

    if target_col is None:
        raise Exception("No valid target column found.")

    data = data.rename(columns={target_col: "Class"})

    # Convert to binary if needed
    if data["Class"].dtype == "object":
        data["Class"] = data["Class"].astype("category").cat.codes

    # Handle missing values
    data = data.fillna(0)

    # Encode categorical features
    for col in data.columns:
        if data[col].dtype == "object":
            data = pd.get_dummies(data, columns=[col], drop_first=True)

    return data
