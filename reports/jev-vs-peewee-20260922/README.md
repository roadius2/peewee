# Peewee mix-v1 against Jev 1.13.0, measured (2026-09-22)

Both systems answered the same cases, and `peewee_decide.evaluate` scored both.
Made with `scripts/jev_compare.py` (`jev`, `peewee`, `report`).

- **Data:** typed-decisions test (400 cases, 2,000 questions), Open-Jev test (1,920 / 10,356)
  and Open-Jev OOD (1,790 / 15,701), release-v2-redistributable at the pinned revision.
- **Jev:** `jev-latest`, which the API reported as `jev-1.13.0`. Called over HTTPS from a home
  connection on the same LAN as the GPU box. For each split, the first 200 cases were sent one at a
  time (`latency_ms_sequential`) and the rest 5 at a time. Every case answered.
  - Six OOD cases carry about 576 questions each, which is over the 64k-token request limit. They
    were sent in parts of 50 questions, with latency and tokens summed.
  - The run used 4.50M input tokens, about $0.19 at the listed $42 per billion.
- **Peewee:** `mix-v1`, one case per `predict` call, as `peewee eval` runs it. Measured on an RTX 5090
  (bf16 autocast, `rtx5090-*.json`) and on an Apple M5 Max (MPS, fp32, `m5max-*.json`). No state was
  truncated.
- **Confidence:** Jev's `confidence` field measures how concentrated its distribution is. For ECE,
  both systems are scored on the probability of the answer they give: the top option, or
  `max(p, 1-p)` for noul. Jev's probabilities are rounded to two decimals, so they were renormalised.
- **Scope:** Jev's answers are only used for this comparison. They are not used for training,
  calibration or data selection.

| | typed-decisions | Open-Jev test | Open-Jev OOD |
|---|---|---|---|
| Peewee mix-v1 accuracy | **0.8005** | **0.9403** | **0.8310** |
| Jev 1.13.0 accuracy | 0.7385 | 0.8104 | 0.8031 |
| Peewee ECE | 0.1876 | **0.0128** | 0.1354 |
| Jev ECE | **0.0447** | 0.0216 | **0.0487** |
| Peewee Brier (vs target distribution) | **0.0567** | **0.0589** | 0.2953 |
| Jev Brier | 0.1481 | 0.2398 | **0.2663** |
| same answer | 74.3% | 80.4% | 71.6% |
| Jev p50 per case (HTTPS round trip) | 282 ms | 280 ms | 286 ms |
| Peewee p50 per case, RTX 5090 | 17 ms | 10 ms | 11 ms |
| Peewee p50 per case, M5 Max | 299 ms | 133 ms | 156 ms |

Caveats:

- mix-v1 was trained on the typed-decisions and Open-Jev *train* splits, so those two test splits
  are in-distribution for Peewee and not for Jev. OOD is the fairest comparison.
- Jev's latency includes the network round trip. Peewee's is in-process, with no HTTP.
- The M5 Max runs Peewee in fp32. It gives the same accuracy as the 5090's bf16 run to within 0.001,
  and 0.3% of answers differ.
