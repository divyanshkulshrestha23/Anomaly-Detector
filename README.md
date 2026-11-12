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

## Alternatives checked

**Isolation Forest** (unsupervised, fit on normal transactions only) was tried and dropped. On the holdout set it reached a PR-AUC of 0.0269, against 0.7715 for logistic regression. At the 0.5% alert rate it had precision 0.0176 and recall 0.0556. Its anomaly scores separated fraud from normal transactions poorly on this dataset, so it is not in the serving path.

## Roadmap

- [x] Train and compare models (Isolation Forest tried and dropped; see above)
- [ ] `/predict` endpoint
- [ ] Request validation
- [ ] Structured logging
- [ ] Unit and API tests
- [ ] Dockerfile
- [ ] GitHub Actions: tests on every push
- [ ] Deploy to Google Cloud Run
- [ ] Load test with p50 / p95 latency results
