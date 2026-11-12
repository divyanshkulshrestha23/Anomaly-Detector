"""Load test for the /predict endpoint. Standard library only.

Sends transactions concurrently and reports status counts and latency
percentiles (p50, p95, p99) measured on the client side.

Usage:
    python load/load_test.py --url https://SERVICE_URL --requests 1000 --concurrency 10

Options:
    --csv PATH       sample real transactions from the dataset (default: data/creditcard.csv,
                     falls back to a fixed sample transaction if the file is absent)
    --token TOKEN    Google identity token for a private Cloud Run service; or set the
                     CLOUD_RUN_TOKEN environment variable
                     (e.g. $(gcloud auth print-identity-token))

Latency here includes network time from where the script runs, so run it from
the same region as the service for numbers comparable to the server's own latency.
"""

import argparse
import csv
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"
FALLBACK = {f: 0.0 for f in FEATURES} | {"Time": 50000.0, "Amount": 15.0}


def load_transactions(path: Path) -> list[dict]:
    """Return every row as a feature dict, or a fallback sample if the file is missing."""
    if not path.exists():
        print(f"{path} not found; sending a single fixed sample transaction")
        return [FALLBACK]
    with path.open(newline="") as f:
        return [{name: float(row[name]) for name in FEATURES} for row in csv.DictReader(f)]


def send(url: str, payload: dict, token: str | None) -> tuple[int, float]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as err:
        status = err.code
    except urllib.error.URLError:
        status = 0
    return status, (time.perf_counter() - start) * 1000


def percentile(sorted_values: list[float], pct: float) -> float:
    index = min(len(sorted_values) - 1, round(pct / 100 * (len(sorted_values) - 1)))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="base URL of the service, e.g. https://host")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--token", default=os.environ.get("CLOUD_RUN_TOKEN"))
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/predict"
    transactions = load_transactions(args.csv)
    rng = random.Random(42)
    payloads = [rng.choice(transactions) for _ in range(args.requests)]

    # One warm-up request so cold-start time doesn't skew the percentiles.
    send(endpoint, payloads[0], args.token)

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda p: send(endpoint, p, args.token), payloads))
    wall_s = time.perf_counter() - wall_start

    latencies = sorted(ms for _, ms in results)
    status_counts: dict[int, int] = {}
    for status, _ in results:
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"endpoint:      {endpoint}")
    print(f"requests:      {args.requests}  concurrency: {args.concurrency}")
    print(f"status codes:  {dict(sorted(status_counts.items()))}")
    print(f"throughput:    {args.requests / wall_s:.1f} req/s")
    print(f"latency p50:   {percentile(latencies, 50):.1f} ms")
    print(f"latency p95:   {percentile(latencies, 95):.1f} ms")
    print(f"latency p99:   {percentile(latencies, 99):.1f} ms")
    print(f"latency mean:  {statistics.fmean(latencies):.1f} ms")


if __name__ == "__main__":
    main()
