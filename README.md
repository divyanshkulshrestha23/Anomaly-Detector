# Anomaly Detector

A model-serving backend for an anomaly detector. It wraps a trained model in an HTTP service with request validation, structured logging, automated tests, containerization, CI/CD, and load testing.

## Status

Skeleton only. Components are being added step by step.

## Planned structure

```
app/        HTTP endpoint, request validation, logging
model/      Training script and saved model artifact
tests/      Unit and API tests
load/       Load test script and recorded results
.github/    CI workflow
```

## Data

The model is trained on synthetic sensor-style readings (temperature, fan speed, power draw) generated in code. No real or proprietary data is used.

## Roadmap

- [ ] Train and save anomaly model (Isolation Forest)
- [ ] `/predict` endpoint
- [ ] Request validation
- [ ] Structured logging
- [ ] Unit and API tests
- [ ] Dockerfile
- [ ] GitHub Actions: tests on every push
- [ ] Deploy to Google Cloud Run
- [ ] Load test with p50 / p95 latency results
