"""Tests for the /predict endpoint and its request validation."""

import json

import pytest

from app.main import app
from app.validation import ValidationError, validate_payload

VALID = {"temperature_c": 45, "fan_rpm": 3000, "power_w": 250}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_valid_normal_reading_is_not_anomaly(client):
    resp = client.post("/predict", json=VALID)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["anomaly"] is False
    assert isinstance(body["score"], float)


def test_extreme_reading_is_anomaly(client):
    resp = client.post(
        "/predict", json={"temperature_c": 90, "fan_rpm": 6000, "power_w": 600}
    )
    assert resp.status_code == 200
    assert resp.get_json()["anomaly"] is True


def test_non_json_body_returns_400(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400
    assert "body" in resp.get_json()["details"]


def test_missing_field_returns_400(client):
    resp = client.post("/predict", json={"temperature_c": 45, "fan_rpm": 3000})
    assert resp.status_code == 400
    assert resp.get_json()["details"] == {"power_w": "missing"}


def test_non_numeric_value_returns_400(client):
    resp = client.post("/predict", json={**VALID, "fan_rpm": "fast"})
    assert resp.status_code == 400
    assert resp.get_json()["details"] == {"fan_rpm": "must be a number"}


def test_boolean_value_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload({**VALID, "power_w": True})
    assert exc.value.errors == {"power_w": "must be a number"}


def test_non_finite_value_is_rejected():
    with pytest.raises(ValidationError) as exc:
        validate_payload({**VALID, "temperature_c": float("nan")})
    assert exc.value.errors == {"temperature_c": "must be a finite number"}


def test_all_errors_reported_together():
    with pytest.raises(ValidationError) as exc:
        validate_payload({"temperature_c": "hot"})
    assert set(exc.value.errors) == {"temperature_c", "fan_rpm", "power_w"}


def test_each_request_gets_request_id_header(client):
    resp = client.post("/predict", json=VALID)
    assert len(resp.headers["X-Request-ID"]) == 32


def test_successful_request_emits_json_log(client, capsys):
    client.post("/predict", json=VALID)
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
    client.post("/predict", json={"temperature_c": 45})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    warnings = [entry for entry in lines if entry["message"] == "validation failed"]
    assert warnings
    assert warnings[-1]["level"] == "WARNING"
    assert "power_w" in warnings[-1]["error_details"]
