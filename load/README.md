# Load test

`load_test.py` sends transactions to `/predict` concurrently and reports status codes, throughput, and p50/p95/p99 latency. It uses only the standard library.

The Cloud Run service is private, so requests need a Google identity token. Run the test from Cloud Shell, which is already signed in to the project:

```bash
python load/load_test.py \
  --url "$SERVICE_URL" \
  --requests 2000 --concurrency 10 \
  --token "$(gcloud auth print-identity-token)"
```

`SERVICE_URL` is the service's URL from `gcloud run services describe anomaly-detector --region asia-south1 --format='value(status.url)'`. Upload `data/creditcard.csv` to Cloud Shell first to sample real transactions. Without it, the test sends one fixed sample transaction.

Latency numbers include network time between the place the script runs and the service. Run it from Cloud Shell in `asia-south1`, the same region as the service.
