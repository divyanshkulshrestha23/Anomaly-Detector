"""HTTP entry point for the credit card fraud detector.

POST /predict takes one transaction and returns:
    anomaly          True when the fraud score is at or above the alert threshold
    score            estimated fraud probability (the model is class-balanced, so
                     this is a ranking score, not a calibrated probability)
    alert_threshold  the score cut-off used, set on validation data for a 0.5% alert rate
    top_features     the three features that pushed the score up or down most
"""

import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, g, jsonify, request

from app.logging_utils import configure_logging
from app.validation import FEATURES, ValidationError, validate_payload

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "logistic_regression.joblib"
TOP_N = 3

logger = configure_logging()

app = Flask(__name__)

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["model"]
scaler = pipeline.named_steps["standardscaler"]
classifier = pipeline.named_steps["logisticregression"]
ALERT_THRESHOLD = bundle["thresholds"][bundle["default_alert_rate"]]

if list(bundle["features"]) != list(FEATURES):
    raise RuntimeError("model feature order does not match app/validation.py")


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


def top_contributions(row: pd.DataFrame, n: int = TOP_N) -> list[dict]:
    """Per-feature contribution to the logit: standardized value times coefficient."""
    contributions = scaler.transform(row)[0] * classifier.coef_[0]
    order = np.argsort(-np.abs(contributions))[:n]
    return [
        {"feature": FEATURES[i], "contribution": round(float(contributions[i]), 4)}
        for i in order
    ]


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    row = pd.DataFrame([validate_payload(payload)], columns=FEATURES)

    score = float(pipeline.predict_proba(row)[0, 1])
    anomaly = score >= ALERT_THRESHOLD
    top = top_contributions(row)

    logger.info(
        "prediction made",
        extra={"request_id": g.request_id, "anomaly": anomaly, "score": score},
    )
    return jsonify(
        {
            "anomaly": anomaly,
            "score": score,
            "alert_threshold": ALERT_THRESHOLD,
            "top_features": top,
        }
    )


if __name__ == "__main__":
    app.run(port=8080)
