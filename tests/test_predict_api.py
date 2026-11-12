"""Tests for the /predict endpoint and its request validation."""

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
