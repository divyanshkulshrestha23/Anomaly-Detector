"""Structured (JSON) logging for the serving app."""

import json
import logging
import sys
from datetime import datetime, timezone

# Fields passed via `extra=` that should appear in the JSON output.
EXTRA_FIELDS = (
    "request_id",
    "method",
    "path",
    "status",
    "latency_ms",
    "anomaly",
    "score",
    "error_details",
)


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            if hasattr(record, field):
                entry[field] = getattr(record, field)
        return json.dumps(entry)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is at emit time, not at construction time."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Attach a JSON handler to the app logger. Safe to call more than once."""
    logger = logging.getLogger("anomaly.api")
    logger.setLevel(level)
    if not any(isinstance(h, _StdoutHandler) for h in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    return logger
