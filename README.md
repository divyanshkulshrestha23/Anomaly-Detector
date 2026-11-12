# Credit Card Fraud API

A model-serving backend for credit card fraud alert triage. A logistic regression model trained on public transaction data is served over HTTP. For each transaction, it returns an alert decision, a fraud score, and the features that drove the score. The service has request validation, structured logging, automated tests, a Docker image, a CI/CD pipeline, and a load test.

The service is deployed on Google Cloud Run in `asia-south1`.

## Problem framing

The goal is to flag the small share of transactions most likely to be fraud, so an analyst can review them. The alert threshold sets how many transactions get flagged. Lower thresholds catch more fraud but send more transactions to review. The threshold is set on validation data for a target alert rate (0.5% by default). The holdout set is only used to report results.

## Data

The model is trained on the Credit Card Fraud Detection dataset on Kaggle, collected by the Machine Learning Group at ULB (Université Libre de Bruxelles) in a research collaboration with Worldline. It has 284,807 transactions, of which 492 (0.17%) are fraud. The features V1–V28 are anonymized principal components, and the only other raw fields are `Time` and `Amount`. The dataset's license is the [Database Contents License (DbCL) 1.0](http://opendatacommons.org/licenses/dbcl/1.0/). The data is not committed to this repository; download it to `data/creditcard.csv` to retrain.

Citation, as requested by the dataset authors:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.

The Kaggle page lists further papers from the same group.

## Model and results

Data is split by time. The earliest 70% of transactions are for training (of which the earliest 80% is fit and the rest is validation), and the latest 30% is the holdout set. The holdout set has 85,443 transactions, with 108 frauds (0.126%).

Logistic regression, with standardized features and balanced class weights, is the model being served. Results on the holdout set:

| Model | PR-AUC | Alert rate | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.7715 | 0.2% | 0.438 | 0.824 |
| Logistic regression | 0.7715 | 0.5% | 0.206 | 0.861 |
| Logistic regression | 0.7715 | 1.0% | 0.102 | 0.870 |
| Gradient boosting | 0.1846 | 0.2% | 0.278 | 0.648 |
| Gradient boosting | 0.1846 | 0.5% | 0.172 | 0.657 |
| Gradient boosting | 0.1846 | 1.0% | 0.069 | 0.657 |

Gradient boosting was tried as a stronger alternative, but it scored lower on this holdout set. Isolation Forest, an unsupervised model, was also tried and dropped. Its holdout PR-AUC was 0.0269, and at the 0.5% alert rate its precision was 0.0176 and its recall was 0.0556.

The threshold is the trade-off. At a 0.5% alert rate, about one in five flagged transactions is fraud, and about 86% of frauds are caught. Tightening to 0.2% raises precision to about 44% but drops recall to about 82%.

Full numbers are in `model/metrics.json`. Retrain with `python model/train.py`.

## API

### `POST /predict`

Send one transaction as a JSON object with all 30 fields: `Time`, `Amount`, and `V1` to `V28`. All fields are numbers.

```bash
curl -X POST https://anomaly-detector-oncwff5yqa-el.a.run.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"Time": 50000, "Amount": 15, "V1": 0, "V2": 0, "V3": 0, "V4": 0, "V5": 0, "V6": 0, "V7": 0, "V8": 0, "V9": 0, "V10": 0, "V11": 0, "V12": 0, "V13": 0, "V14": 0, "V15": 0, "V16": 0, "V17": 0, "V18": 0, "V19": 0, "V20": 0, "V21": 0, "V22": 0, "V23": 0, "V24": 0, "V25": 0, "V26": 0, "V27": 0, "V28": 0}'
```

A response looks like this. The values come from the all-zero example above:

```json
{
  "anomaly": false,
  "score": 0.0172,
  "alert_threshold": 0.9426,
  "top_features": [
    {"feature": "Amount", "contribution": -0.4528},
    {"feature": "V3", "contribution": -0.1695},
    {"feature": "V4", "contribution": -0.1116}
  ]
}
```

- `anomaly` is true when `score` reaches `alert_threshold`, which is set for a 0.5% alert rate.
- `score` is the model's fraud score. Because the model is trained with balanced class weights, treat it as a ranking, not a calibrated probability.
- `top_features` lists the three features that moved the score most. A positive contribution pushes toward fraud, and a negative one pushes away.

Invalid requests get a `400` response that lists each problem by field. A request with missing fields, non-numeric values, or booleans in place of numbers is rejected.

Each response has an `X-Request-ID` header. The service writes one JSON log line per request, with the request ID and latency.

## Latency

Load test run from a laptop on a home network against the deployed service, with 2,000 requests and 10 concurrent workers. All requests returned `200`.

| Metric | Value |
|---|---|
| Throughput | 50.4 requests/s |
| p50 | 105.7 ms |
| p95 | 423.7 ms |
| p99 | 1174.2 ms |
| Mean | 197.1 ms |

These numbers include the network from the laptop to Cloud Run, so the server's own time is lower. Cold starts can drive up p99. Run the test yourself with `load/load_test.py` (see `load/README.md`).

## Running locally

```bash
pip install -r requirements.txt
python -m pytest -q
gunicorn --bind 0.0.0.0:8080 --workers 2 app.main:app
```

Or with Docker:

```bash
docker build -t credit-card-fraud-api .
docker run -p 8080:8080 credit-card-fraud-api
```

## Limitations

- **Old and anonymized data.** The transactions are from 2013 and most features are anonymized principal components, so the model can't be expected to match current fraud patterns. V1–V28 have no published meaning in this dataset, so `top_features` names them but can't explain them in business terms.
- **Few frauds.** The dataset has 492 frauds, and the holdout has 108. Precision and recall at each threshold move a lot with a few transactions, so the numbers are rough estimates.
- **One split.** Results come from a single time-based split and are not averaged over several splits.
- **Score is not calibrated.** Balanced class weights inflate the scores, so the score can't be read as a fraud probability.
- **No drift monitoring.** The service doesn't check whether incoming transactions still look like the training data.
- **No authentication.** The Cloud Run service is public, so anyone with the URL can call it. Fine for a demo, not for production traffic.
- **Latency from one location.** The load test ran from one home network. Results from other locations will differ.

## Project layout

```
app/        HTTP endpoint, request validation, logging
model/      Training script, saved model bundle, metrics
tests/      Unit and API tests
load/       Load test script and instructions
.github/    CI/CD workflow
```
