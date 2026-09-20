"""Load-test the Laya decision service: concurrency, p50/p95/p99 and throughput.

    python scripts/loadtest.py http://localhost:8000 --concurrency 64 --requests 2000

Uses only the standard library so it runs anywhere the client does.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this request?",
                   "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                "sales": "pricing, new contracts", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}
STATES = [
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
    {"body": "The API has been returning 502s since 9am and our checkout is down."},
    {"body": "Can I get pricing for 50 seats on the enterprise tier?"},
    {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    {"body": "Je veux annuler mon forfait, votre service ne marche pas."},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--api-key")
    args = ap.parse_args()

    latencies, errors, lock = [], [0], threading.Lock()
    counter = [0]

    def worker():
        while True:
            with lock:
                if counter[0] >= args.requests:
                    return
                i = counter[0]
                counter[0] += 1
            body = json.dumps({"state": STATES[i % len(STATES)], "questions": QUESTIONS}).encode()
            req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/decide", data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            if args.api_key:
                req.add_header("Authorization", "Bearer " + args.api_key)
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    r.read()
                dt = time.perf_counter() - t0
                with lock:
                    latencies.append(dt)
            except Exception as e:      # noqa: BLE001 - a load test reports, it does not crash
                with lock:
                    errors[0] += 1
                print("error:", e, file=sys.stderr)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start
    if not latencies:
        print("no successful requests")
        return 1
    lat = sorted(latencies)
    pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] * 1000  # noqa: E731
    print("requests   %d ok, %d errors, %.1fs wall" % (len(lat), errors[0], wall))
    print("throughput %.1f req/s, %.1f questions/s" % (len(lat) / wall, len(lat) * len(QUESTIONS) / wall))
    print("latency ms p50 %.1f  p95 %.1f  p99 %.1f  mean %.1f  max %.1f" % (
        pct(0.50), pct(0.95), pct(0.99), statistics.mean(lat) * 1000, lat[-1] * 1000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
