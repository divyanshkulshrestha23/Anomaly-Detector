"""Tests for the /predict endpoint and its request validation."""

import json

import pytest

from app.main import app
from app.validation import FEATURES, ValidationError, validate_payload

# A typical-looking transaction: small amount, features near zero.
NORMAL = {f: 0.0 for f in FEATURES} | {"Time": 50000.0, "Amount": 15.0}

# Large amount with strongly negative V14 and V12, which the model weights toward fraud.
SUSPICIOUS = NORMAL | {"Amount": 1500.0, "V14": -8.0, "V12": -6.0, "V4": 4.0, "V2": 2.0}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_normal_transaction_is_not_flagged(client):
    resp = client.post("/predict", json=NORMAL)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["anomaly"] is False
    assert isinstance(body["score"], float)
    assert 0.0 <= body["score"] < body["alert_threshold"]


def test_suspicious_transaction_is_flagged(client):
    resp = client.post("/predict", json=SUSPICIOUS)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["anomaly"] is True
    assert body["score"] >= body["alert_threshold"]


def test_response_lists_three_top_features_pushing_toward_fraud(client):
    body = client.post("/predict", json=SUSPICIOUS).get_json()
    top = body["top_features"]
    assert len(top) == 3
    assert {item["feature"] for item in top} <= set(FEATURES)
    # Positive contribution means the feature pushes the score toward fraud.
    assert top[0]["feature"] == "V14"
    assert top[0]["contribution"] > 0


def test_non_json_body_returns_400(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert "body" in resp.get_json()["details"]


def test_missing_field_returns_400(client):
    payload = {k: v for k, v in NORMAL.items() if k != "V28"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert resp.get_json()["details"] == {"V28": "missing"}


def test_non_numeric_value_returns_400(client):
    resp = client.post("/predict", json={**NORMAL, "Amount": "fifteen"})
    assert resp.status_code == 400
    assert resp.get_json()["details"] == {"Amount": "must be a number"}


def test_boolean_value_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload({**NORMAL, "V1": True})
    assert exc.value.errors == {"V1": "must be a number"}


def test_non_finite_value_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload({**NORMAL, "V3": float("nan")})
    assert exc.value.errors == {"V3": "must be a finite number"}


def test_all_errors_reported_together():
    with pytest.raises(ValidationError) as exc:
        validate_payload({"Amount": "lots"})
    assert set(exc.value.errors) == set(FEATURES)
    assert exc.value.errors["Amount"] == "must be a number"
    assert exc.value.errors["V28"] == "missing"


def test_validate_payload_returns_features_in_model_order():
    values = validate_payload(NORMAL)
    assert values == [NORMAL[name] for name in FEATURES]


def test_each_request_gets_request_id_header(client):
    resp = client.post("/predict", json=NORMAL)
    assert len(resp.headers["X-Request-ID"]) == 32


def test_successful_request_emits_json_log(client, capsys):
    client.post("/predict", json=NORMAL)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    completed = [entry for entry in lines if entry["message"] == "request completed"]
    assert completed, "expected a 'request completed' log line"
    entry = completed[-1]
    assert entry["path"] == "/predict"
    assert entry["method"] == "POST"
    assert entry["status"] == 200
    assert isinstance(entry["latency_ms"], float)
    assert entry["request_id"]
    assert any(line["message"] == "prediction made" and "anomaly" in line for line in lines)


def test_validation_failure_is_logged_as_warning(client, capsys):
    client.post("/predict", json={"Amount": 15.0})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    warnings = [entry for entry in lines if entry["message"] == "validation failed"]
    assert warnings
    assert warnings[-1]["level"] == "WARNING"
    assert "V1" in warnings[-1]["error_details"]
