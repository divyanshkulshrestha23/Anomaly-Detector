"""HTTP entry point for the anomaly detector."""

import logging
import time
import uuid
from pathlib import Path

import joblib
from flask import Flask, g, jsonify, request

from app.logging_utils import configure_logging
from app.validation import ValidationError, validate_payload

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "anomaly_model.joblib"

logger = configure_logging()

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.before_request
def start_request():
    g.request_id = uuid.uuid4().hex
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    latency_ms = round((time.perf_counter() - g.start_time) * 1000, 2)
    response.headers["X-Request-ID"] = g.request_id
    logger.info(
        "request completed",
        extra={
            "request_id": g.request_id,
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


@app.errorhandler(ValidationError)
def handle_validation_error(err: ValidationError):
    logger.warning(
        "validation failed",
        extra={"request_id": g.request_id, "error_details": err.errors},
    )
    return jsonify({"error": "invalid request", "details": err.errors}), 400


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    row = [validate_payload(payload)]

    label = int(model.predict(row)[0])  # 1 = normal, -1 = anomaly
    score = float(model.decision_function(row)[0])  # lower = more anomalous
    anomaly = label == -1

    logger.info(
        "prediction made",
        extra={"request_id": g.request_id, "anomaly": anomaly, "score": score},
    )
    return jsonify({"anomaly": anomaly, "score": score})


if __name__ == "__main__":
    app.run(port=8080)
