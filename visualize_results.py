import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fraud_log.csv")

WINDOW = 200
df["rolling_error"] = df["error"].rolling(WINDOW).mean()

plt.figure()
plt.plot(df["rolling_error"])
plt.title("Model Degradation Over Time")
plt.xlabel("Transaction")
plt.ylabel("Rolling Error Rate")
plt.savefig("model_degradation.png")
plt.close()

plt.figure()
plt.plot(df["actual"].cumsum(), label="Actual Frauds")
plt.plot(df["predicted"].cumsum(), label="Detected Frauds")
plt.title("Cumulative Fraud Detection")
plt.xlabel("Transaction")
plt.ylabel("Count")
plt.legend()
plt.savefig("fraud_detection_growth.png")
plt.close()

plt.figure()
drifts = df[df["drift"] == 1]
plt.scatter(drifts["txn"], drifts["rolling_error"])
plt.title("Drift Severity Map")
plt.xlabel("Transaction")
plt.ylabel("Rolling Error")
plt.savefig("drift_severity_map.png")
plt.close()

print("Visualizations generated successfully.")
