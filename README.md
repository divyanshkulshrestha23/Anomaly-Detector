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

The model is trained on the Credit Card Fraud Detection dataset on Kaggle, collected by the Machine Learning Group at ULB (Université Libre de Bruxelles) in a research collaboration with Worldline. It has 284,807 transactions, of which 492 (0.17%) are fraud. The features V1–V28 are anonymized principal components, and the only other raw fields are `Time` and `Amount`. The dataset's license is the [Database Contents License (DbCL) 1.0](http://opendatacommons.org/licenses/dbcl/1.0/). The data is not committed to this repository; download it to `data/creditcard.csv` to retrain.

Citation, as requested by the dataset authors:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

The Kaggle page lists further papers from the same group. Because the features are anonymized and the data is from 2013, the results here do not show how the models would perform on current transactions.

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
