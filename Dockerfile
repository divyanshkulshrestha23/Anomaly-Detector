# Serving image for the anomaly detector.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model artifact.
COPY app/ app/
COPY model/anomaly_model.joblib model/anomaly_model.joblib

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

# Cloud Run sets PORT; default to 8080 for local runs.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 app.main:app"]
