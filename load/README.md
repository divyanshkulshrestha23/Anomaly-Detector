# Load test

`load_test.py` sends transactions to `/predict` concurrently and reports status codes, throughput, and p50/p95/p99 latency. It uses only the standard library and needs no authentication, since the service is public.

Run it from any machine with Python 3 and internet access, for example your laptop:

```bash
python load/load_test.py --url "$SERVICE_URL" --requests 2000 --concurrency 10 --csv path/to/creditcard.csv
```

`SERVICE_URL` is the service's URL, which you can get with `gcloud run services describe anomaly-detector --region asia-south1 --format='value(status.url)'`. Pass `--csv` to sample real transactions from the dataset. Without it, the test sends one fixed sample transaction.

Latency numbers include network time between the machine running the script and the service. Note the location of the machine when you report the results.
