import pandas as pd
from river.drift import ADWIN
import sys
import pickle
import subprocess
import os
import numpy as np

# ==============================
# ARGUMENT
# ==============================

bank = sys.argv[1]
print(f"Production monitoring started for {bank}")

# ==============================
# LOAD DATA
# ==============================

data = pd.read_csv(f"clients/{bank}/data.csv")

# ==============================
# LOAD GLOBAL MODEL
# ==============================

with open("server/global_model/global_model.pkl", "rb") as f:
    model = pickle.load(f)

# Default production threshold
fraud_threshold = 0.5

# ==============================
# PREPARE FEATURES
# ==============================

X = data.drop("Class", axis=1)
y = data["Class"]

# ==============================
# DRIFT DETECTOR
# ==============================

adwin = ADWIN()
drift_events = 0

# ==============================
# FRAUD METRICS
# ==============================

fraud_count = 0
true_fraud_detected = 0
false_positives = 0
false_negatives = 0

# ==============================
# MONITOR STREAM
# ==============================

for i in range(len(X)):
    x = X.iloc[i:i+1]
    y_true = y.iloc[i]

    # Predict probability
    prob = model.predict_proba(x)[0][1]

    # Final fraud decision
    y_pred = 1 if prob > fraud_threshold else 0

    # ======================
    # FRAUD METRIC TRACKING
    # ======================

    if y_pred == 1:
        fraud_count += 1

    if y_pred == 1 and y_true == 1:
        true_fraud_detected += 1

    if y_pred == 1 and y_true == 0:
        false_positives += 1

    if y_pred == 0 and y_true == 1:
        false_negatives += 1

    # ======================
    # CONCEPT DRIFT LOGIC
    # ======================

    error = int(y_true != y_pred)
    adwin.update(error)

    if adwin.drift_detected:
        drift_events += 1
        print(f" Drift detected at transaction {i}")
        subprocess.call(["python", "trigger_federated_round.py"])

# ==============================
# FINAL SUMMARY
# ==============================

print("\n--- Production Monitoring Summary ---")

print(f"Total Transactions Processed: {len(X)}")
print(f"Total Predicted Frauds: {fraud_count}")
print(f"True Frauds Correctly Detected: {true_fraud_detected}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")
print(f"Total Drift Events: {drift_events}")

# Precision / Recall calculation
if fraud_count > 0:
    precision = true_fraud_detected / fraud_count
else:
    precision = 0

if (true_fraud_detected + false_negatives) > 0:
    recall = true_fraud_detected / (true_fraud_detected + false_negatives)
else:
    recall = 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
