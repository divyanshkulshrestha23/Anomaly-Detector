"""HTTP entry point for the anomaly detector."""

from pathlib import Path

import joblib
from flask import Flask, jsonify, request

from app.validation import ValidationError, validate_payload

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "anomaly_model.joblib"

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.errorhandler(ValidationError)
def handle_validation_error(err: ValidationError):
    return jsonify({"error": "invalid request", "details": err.errors}), 400


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    row = [validate_payload(payload)]

    label = int(model.predict(row)[0])  # 1 = normal, -1 = anomaly
    score = float(model.decision_function(row)[0])  # lower = more anomalous

    return jsonify({"anomaly": label == -1, "score": score})


if __name__ == "__main__":
    app.run(port=8080)
