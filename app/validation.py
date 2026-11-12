"""Validation for incoming prediction requests."""

import math

# Must match the feature order the model was trained with (model/train.py).
FEATURES = ("Time", "Amount") + tuple(f"V{i}" for i in range(1, 29))


class ValidationError(ValueError):
    """Raised when a request payload is missing fields or contains invalid values."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("invalid request")
        self.errors = errors


def validate_payload(payload: object) -> list[float]:
    """Check a request body and return the feature values in FEATURES order.

    Raises ValidationError with a message for each problem found.
    """
    if not isinstance(payload, dict):
        raise ValidationError({"body": "expected a JSON object"})

    errors: dict[str, str] = {}
    values: list[float] = []

    for name in FEATURES:
        if name not in payload:
            errors[name] = "missing"
            continue

        raw = payload[name]
        # bool is a subclass of int in Python, so reject it explicitly.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            errors[name] = "must be a number"
            continue

        value = float(raw)
        if not math.isfinite(value):
            errors[name] = "must be a finite number"
            continue

        values.append(value)

    if errors:
        raise ValidationError(errors)

    return values
