import subprocess
import os

# Absolute path to flenv python
PYTHON_PATH = os.path.join(os.getcwd(), "flenv", "Scripts", "python.exe")

print("Concept drift detected. Triggering federated retraining...")

subprocess.call([PYTHON_PATH, os.path.join("server", "aggregator.py")])

print("Federated retraining completed.")
