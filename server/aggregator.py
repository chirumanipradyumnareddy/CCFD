import pickle
import os
import numpy as np

print("Loading local bank models...")

models = []
thresholds = []

clients_path = "clients"

for bank in os.listdir(clients_path):
    model_path = f"{clients_path}/{bank}/local_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            package = pickle.load(f)
            models.append(package["model"])
            thresholds.append(package["threshold"])

print("All bank models loaded")

# Extract real logistic models from CalibratedClassifierCV
coefs = []
intercepts = []

for model in models:
    calibrated_model = model.calibrated_classifiers_[0].estimator
    coefs.append(calibrated_model.coef_)
    intercepts.append(calibrated_model.intercept_)

# Federated averaging
avg_coef = np.mean(coefs, axis=0)
avg_intercept = np.mean(intercepts, axis=0)
avg_threshold = np.mean(thresholds)

# Update first model as global model
global_model = models[0]
calibrated_model = global_model.calibrated_classifiers_[0].estimator
calibrated_model.coef_ = avg_coef
calibrated_model.intercept_ = avg_intercept

# Save global package
os.makedirs("server/global_model", exist_ok=True)

