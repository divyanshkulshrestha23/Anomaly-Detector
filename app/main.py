"""HTTP entry point for the anomaly detector."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "anomaly_model.joblib"
FEATURES = ("temperature_c", "fan_rpm", "power_w")

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    row = [[float(payload[name]) for name in FEATURES]]

    label = int(model.predict(row)[0])  # 1 = normal, -1 = anomaly
    score = float(model.decision_function(row)[0])  # lower = more anomalous

    return jsonify({"anomaly": label == -1, "score": score})


if __name__ == "__main__":
    app.run(port=8080)
