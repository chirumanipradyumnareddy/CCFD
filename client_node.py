import pandas as pd
import pickle
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

bank = sys.argv[1]
print(f"Starting training for {bank}")

data = pd.read_csv(f"clients/{bank}/data.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# STRATIFIED SPLIT (VERY IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Base Model
base_model = LogisticRegression(
    max_iter=5000,
    solver="liblinear"
)

# Probability Calibration
model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
model.fit(X_train, y_train)

# Evaluation
probs = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, probs)

print(f"{bank} ROC-AUC: {auc:.4f}")

# Determine optimal threshold using PR curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1_scores)]

print(f"{bank} Optimal Threshold: {best_threshold:.4f}")

# Save model AND threshold
with open(f"clients/{bank}/local_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "threshold": best_threshold
    }, f)

print(f"{bank} production-ready model saved.")
