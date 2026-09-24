# Peewee benchmarks

Every checkpoint answered **byte-identical questions** in each run (fixed seed). Jev figures in the older sections are **third-party published**, so their sample sizes and prompts differ; treat them as indicative. [Measured against Jev](#measured-against-jev-fork-2026-09-22) has Jev 1.13.0 measured by us over its API, on the same cases as Peewee and scored the same way.

| run | what | where |
|---|---|---|
| T4 Colab | typed-decisions, MASSIVE (14 langs), XNLI (15 langs), English suites, latency, option-order robustness, calibration repair | `research/results/t4_colab_benchmark.json` |
| CPU sweep | MASSIVE intent across **all 51 languages**, typed-decisions on all three checkpoints | `research/results/cpu_51_language_sweep.json` |
| Applications | the six workflow themes + the datasets where Jev numbers exist, all three checkpoints | `research/results/app_benchmark.json` |

---

## Headline

| | Peewee | Jev (published) |
|---|---|---|
| typed-decisions (2,000 decisions) | **0.766** | 0.727 |
| AG News (4 labels) | **0.953** | 0.910 |
| DAIR Emotion (6 labels) | **0.600** | 0.480 |
| ECE after temperature fitting | **0.081** | 0.246 |
| p50 latency, 1 question (T4) | **32.8 ms** | 236-276 ms |

---

## Languages

### All 51 MASSIVE languages — intent, 20 options (random = 0.050)

| | laya | laya-multilingual |
|---|---|---|
| macro accuracy | 0.2269 | **0.3661** |
| macro ECE *(lower better)* | 0.7331 | **0.3869** |
| languages clearing 3× random | 23 / 51 | **45 / 51** |

<details><summary><b>Per language (51)</b> — sorted by how much routing gains</summary>

| lang | laya | laya-multilingual | Δ | laya ECE | multilingual ECE |
|---|---|---|---|---|---|
| `th` | 0.080 | 0.480 | +0.400 | 0.881 | 0.336 |
| `ko` | 0.110 | 0.450 | +0.340 | 0.850 | 0.329 |
| `he` | 0.060 | 0.400 | +0.340 | 0.911 | 0.350 |
| `ur` | 0.070 | 0.400 | +0.330 | 0.883 | 0.311 |
| `hi` | 0.100 | 0.430 | +0.330 | 0.850 | 0.321 |
| `ar` | 0.110 | 0.400 | +0.290 | 0.800 | 0.341 |
| `pl` | 0.240 | 0.510 | +0.270 | 0.713 | 0.350 |
| `el` | 0.130 | 0.380 | +0.250 | 0.839 | 0.383 |
| `fa` | 0.140 | 0.390 | +0.250 | 0.820 | 0.399 |
| `ru` | 0.310 | 0.540 | +0.230 | 0.668 | 0.316 |
| `tr` | 0.140 | 0.370 | +0.230 | 0.788 | 0.417 |
| `lv` | 0.100 | 0.320 | +0.220 | 0.847 | 0.480 |
| `bn` | 0.080 | 0.290 | +0.210 | 0.865 | 0.408 |
| `nb` | 0.330 | 0.530 | +0.200 | 0.648 | 0.327 |
| `vi` | 0.060 | 0.260 | +0.200 | 0.891 | 0.521 |
| `az` | 0.100 | 0.300 | +0.200 | 0.825 | 0.368 |
| `hu` | 0.090 | 0.290 | +0.200 | 0.857 | 0.422 |
| `is` | 0.110 | 0.300 | +0.190 | 0.835 | 0.469 |
| `sv` | 0.380 | 0.570 | +0.190 | 0.596 | 0.276 |
| `km` | 0.000 | 0.180 | +0.180 | 0.952 | 0.412 |
| `ml` | 0.070 | 0.240 | +0.170 | 0.857 | 0.414 |
| `it` | 0.340 | 0.500 | +0.160 | 0.647 | 0.302 |
| `fi` | 0.130 | 0.290 | +0.160 | 0.849 | 0.436 |
| `ms` | 0.270 | 0.430 | +0.160 | 0.688 | 0.392 |
| `da` | 0.350 | 0.500 | +0.150 | 0.626 | 0.263 |
| `id` | 0.360 | 0.510 | +0.150 | 0.613 | 0.305 |
| `te` | 0.090 | 0.220 | +0.130 | 0.858 | 0.370 |
| `sl` | 0.200 | 0.330 | +0.130 | 0.756 | 0.433 |
| `jv` | 0.160 | 0.270 | +0.110 | 0.803 | 0.506 |
| `ta` | 0.120 | 0.230 | +0.110 | 0.822 | 0.397 |
| `ja` | 0.530 | 0.640 | +0.110 | 0.460 | 0.228 |
| `hy` | 0.050 | 0.150 | +0.100 | 0.835 | 0.506 |
| `zh-TW` | 0.460 | 0.540 | +0.080 | 0.520 | 0.327 |
| `de` | 0.420 | 0.500 | +0.080 | 0.558 | 0.301 |
| `tl` | 0.290 | 0.360 | +0.070 | 0.676 | 0.374 |
| `nl` | 0.390 | 0.450 | +0.060 | 0.591 | 0.378 |
| `af` | 0.290 | 0.350 | +0.060 | 0.687 | 0.484 |
| `my` | 0.060 | 0.120 | +0.060 | 0.861 | 0.455 |
| `sq` | 0.210 | 0.260 | +0.050 | 0.755 | 0.476 |
| `sw` | 0.130 | 0.180 | +0.050 | 0.828 | 0.549 |
| `cy` | 0.120 | 0.160 | +0.040 | 0.841 | 0.591 |
| `kn` | 0.110 | 0.150 | +0.040 | 0.842 | 0.437 |
| `es` | 0.510 | 0.530 | +0.020 | 0.480 | 0.275 |
| `ka` | 0.090 | 0.110 | +0.020 | 0.845 | 0.528 |
| `ro` | 0.330 | 0.350 | +0.020 | 0.658 | 0.404 |
| `zh-CN` | 0.620 | 0.630 | +0.010 | 0.376 | 0.212 |
| `am` | 0.120 | 0.110 | -0.010 | 0.825 | 0.463 |
| `pt` | 0.470 | 0.450 | -0.020 | 0.512 | 0.342 |
| `mn` | 0.130 | 0.100 | -0.030 | 0.837 | 0.558 |
| `fr` | 0.590 | 0.540 | -0.050 | 0.388 | 0.277 |
| `en` | 0.820 | 0.680 | -0.140 | 0.179 | 0.209 |

</details>

### English vs the rest

| task | | laya | laya-multilingual |
|---|---|---|---|
| MASSIVE intent — English | **0.783** | 0.657 |
| MASSIVE intent — other languages | 0.306 | **0.451** |
| MASSIVE scenario — English | **0.603** | 0.560 |
| MASSIVE scenario — other languages | 0.281 | **0.439** |
| XNLI — English | **0.860** | 0.843 |
| XNLI — other languages | 0.521 | **0.731** |

The English checkpoint does not degrade gracefully outside English — it collapses, and stays confident doing so. Khmer: **0.000 accuracy at 0.952 confidence**. Its mean confidence never drops below 0.885 at any accuracy level, so confidence gating cannot catch it — which is why routing happens *before* the forward pass.

---

## Themes — the application workflows

Each is real labelled data, 400 cases, all three checkpoints. *held out* means the source was **not** in Peewee's training mix.

| theme | laya | laya-multilingual | laya-typed-decisions | data |
|---|---|---|---|---|
| Email spam | **0.993** | 0.993 | 0.958 | in training |
| Phishing | 0.980 | **0.993** | 0.940 | in training |
| LLM guardrails (jailbreak) | 0.708 | 0.755 | **0.762** | **held out** |
| Moderation (toxicity) | **0.530** | 0.525 | 0.530 | **held out** |
| RAG passage relevance | 0.625 | **0.657** | 0.625 | in training |
| Support triage (10-way queue) | 0.502 | **0.522** | 0.505 | in training |
| Model routing (domain) | 0.639 | 0.123 | **0.659** | held out |

**Where it is strong:** email spam 0.993 and phishing 0.993, both with ECE around 0.01 — production-grade, though both were in the training mix.

**Where it is weak:** moderation on held-out toxic-chat is 0.530 with macro-F1 0.400 — barely above chance on a balanced split. The demo Space has a Moderation tab; hand-picked examples work, real traffic does not. Guardrails at 0.708–0.762 is the honest jailbreak-detection number, consistent across two unrelated datasets (deepset prompt-injections measured 0.698 separately).

### On the public datasets where Jev numbers exist

| dataset | laya | laya-multilingual | laya-typed-decisions | Jev (published) |
|---|---|---|---|---|
| AG News (4 labels) | 0.950 | 0.930 | **0.953** | 0.910 |
| DAIR Emotion (6 labels) | 0.595 | 0.530 | **0.600** | 0.480 |
| banking77 (77 labels) | 0.425 | 0.425 | **0.492** | 0.870 |

banking77 is the one clear loss, and it is architectural: a choice question's options share a fixed `head_max_len` budget, so 77 labels get roughly 4 tokens each and stop being distinguishable. Both checkpoints score **exactly 0.425**, which is what you would expect from a budget ceiling rather than a capability gap. Keep choice questions under ~20 options.

---

## typed-decisions — 400 cases, 2,000 decisions

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` | **0.766** | 0.471 | 0.061 | 0.213 | 0.242 |
| `laya` | 0.361 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya-multilingual` | 0.342 | 0.326 | 0.439 | 0.285 | 0.687 |
| *Jev 1.13.0 (published)* | *0.727* | *0.580* | *0.148* | *0.144* | *0.391* |
| *teacher ceiling* | *0.735* | *—* | *—* | *—* | *—* |
| *majority class* | *0.461* | *—* | *—* | *—* | *—* |
| *random guess* | *0.318* | *—* | *—* | *—* | *—* |

| workflow | laya-typed-decisions |
|---|---|
| agent trace observability | 0.730 |
| customer service | 0.764 |
| invoice processing | 0.804 |
| security incidents | 0.766 |

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461). All of the capability on this benchmark comes from fine-tuning.

---

## Speed (Tesla T4)

| questions per call | laya | laya-multilingual |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms | **72.3 ms** |
| 50 | 771.3 ms | **337.4 ms** |

103–332 questions/sec batched. Jev independently measured at 236-276 ms p50, so Peewee answers one question roughly **6–7× faster**.

## Decision service throughput (fork, 2026-09-20)

`peewee serve` with `english` and `multilingual` resident, driven by `scripts/loadtest.py`: each
request is one state and three questions (choice, score, noul), five states rotating so both
checkpoints see traffic. Questions/s counts every question answered; p50 is per request.
Measured on one host (AMD Ryzen 9 9950X3D, RTX 5090); the CPU runs are the `docker/Dockerfile`
image limited to 8 cores with `PEEWEE_BACKEND=onnx` and the fp32 export, or the torch CPU path.

| backend | clients | questions/s | requests/s | p50 ms | p99 ms |
|---|---|---|---|---|---|
| torch, RTX 5090 | 4 | 362 | 121 | 23.5 | 262 |
| torch, RTX 5090 | 32 | **1,338** | 446 | 77 | 87 |
| torch, RTX 5090 | 128 | **1,466** | 489 | 281 | 368 |
| ONNX fp32, 8 CPU cores | 4 | 33 | 11 | 383 | 562 |
| ONNX fp32, 8 CPU cores | 32 | 41 | 14 | 2,513 | 2,811 |
| ONNX int8 (weight-only), 8 CPU cores | 4 | 26 | 9 | 457 | 572 |
| torch, 8 CPU cores | 32 | 20 | 7 | 5,148 | 5,531 |

Before the batcher fix in this release the GPU numbers at 32 clients were 467 questions/s at
p50 229 ms: flushes were timer-driven, so under load each forward pass carried about 1.5
requests. Draining the queue after each pass (about 30 questions per pass at 32 clients) is
what the 2.9x comes from. On CPU the same change moved throughput from 36 to 41 questions/s,
because the CPU is compute-bound at any batch size. Weight-only int8 is slower than fp32 on
this CPU; its value is the 60% smaller file. Rule of thumb: one 8-core CPU box serves about
40 questions/s at sub-second latency; one RTX 5090 serves about 1,400 questions/s, roughly
35x, with the p99 under 100 ms up to 32 concurrent clients.

## Reproducing typed-decisions with `peewee train` (fork, 2026-09-21)

`peewee train --base english --max-len 1024 --head-max-len 256` on `LocalLLaMA/typed-decisions`
train (1,200 cases, 120 held out for calibration), one RTX 5090, bf16, defaults otherwise
(4 epochs, effective batch 64). Both checkpoints scored by `peewee eval` on the 2,000-decision test
split, accuracy against the dataset's hard labels as in the original notebook.

| checkpoint | accuracy | soft acc | Brier | ECE | choice | score | noul |
|---|---|---|---|---|---|---|---|
| published `typed-decisions` | 0.7685 | 0.4707 | 0.0615 | 0.2157 | 0.7367 | 0.7262 | 0.8567 |
| `peewee train` (td-v1) | 0.7560 | 0.5113 | 0.0534 | 0.1357 | 0.7350 | 0.7075 | 0.8417 |

Training took 298.6 seconds (about 5.0 minutes). Held-out ECE before and after temperature fitting:
0.1697 -> 0.1844. Temperatures are fitted to each question's teacher distribution (negative
log-likelihood against the soft targets), not to minimise hard-label ECE. On soft-target data the
fit tracks the teacher's spread rather than the argmax accuracy; that is why held-out ECE rose
here, on typed-decisions' flat teacher distribution, while it fell on Open-Jev's near one-hot
targets (0.0237 -> 0.0106, from `reports/trinity-prime-20260921/open-jev-oj-v1-train_meta.json`).
On the full test split td-v1's overall ECE (0.1357) is well below the published checkpoint's
(0.2157). Evidence in `reports/trinity-prime-20260921/`.

The acceptance rule passed: td-v1's accuracy (0.7560) is within 0.02 of the published
checkpoint's (0.7685), a gap of 0.0125. That gap is largest on score-type questions, where
accuracy fell from 0.7262 to 0.7075 (choice and noul each lost less), so score-type questions
are the most likely cause even though td-v1 improves score-type ECE from 0.2024 to 0.1512.
The published checkpoint scored 0.7685 under this eval code versus the 0.766 the upstream
notebook reported. The new eval code agrees with the notebook's scoring to within 0.003 (5 of
2,000 questions), consistent with fp16 versus bf16 tie-breaking.

## Training on Open-Jev (fork, 2026-09-21)

`peewee train --base english --max-len 1024 --head-max-len 256` on Open-Jev's
`release-v2-redistributable` train split (CC0, pinned revision
`c67699e13d0ae25e35b77165a4b6b079bedc8aba`), converted with `peewee prepare-data open-jev`, one
RTX 5090, bf16, defaults otherwise (oj-v1). Training used 13,048 cases (69,473 items) and held
out 1,493 cases (9,643 items) for calibration; 0 questions were skipped. Training took 4,785.6
seconds, about 79.8 minutes. Both oj-v1 and the earlier td-v1 (trained on typed-decisions, see
above) were then evaluated with `peewee eval` on Open-Jev test, Open-Jev OOD and typed-decisions
test.

| checkpoint | Open-Jev test acc | Open-Jev test ECE | Open-Jev OOD acc | Open-Jev OOD ECE | typed-decisions test acc | typed-decisions test ECE |
|---|---|---|---|---|---|---|
| oj-v1 (Open-Jev) | 0.9390 | 0.0122 | 0.8274 | 0.1309 | 0.4215 | 0.3894 |
| td-v1 (typed-decisions) | 0.5455 | 0.1080 | 0.6193 | 0.0809 | 0.7560 | 0.1357 |

oj-v1's accuracy on Open-Jev test, per source (`by_workflow`), sorted by question count:

| source | n | accuracy |
|---|---|---|
| painting-geometry-v1 | 3072 | 1.0000 |
| snake-v1 | 1392 | 0.8994 |
| workflow-controls-v1/security_incidents | 1326 | 0.9910 |
| workflow-controls-v1/invoice_processing | 930 | 0.9925 |
| vizdoom-basic-v1 | 858 | 0.9755 |
| reasoning-control-v1 | 681 | 0.6814 |
| customer-control-v1 | 606 | 0.9719 |
| tic_tac_toe-v1 | 472 | 0.5805 |
| workflow-controls-v1/customer_service | 456 | 1.0000 |
| workflow-controls-v1/agent_trace_observability | 357 | 0.9860 |
| tile_platformer-v1 | 202 | 0.9406 |
| trex_runner-v1 | 4 | 0.2500 |

oj-v1 is the stronger model on Open-Jev: higher accuracy than td-v1 on both Open-Jev test
(0.9390 vs 0.5455) and Open-Jev OOD (0.8274 vs 0.6193), and better calibrated on Open-Jev test
(ECE 0.0122 vs 0.1080). td-v1 is better calibrated on Open-Jev OOD (ECE 0.0809 vs 0.1309)
despite its lower accuracy there. td-v1 remains the stronger model on typed-decisions test,
both more accurate (0.7560 vs 0.4215) and better calibrated (ECE 0.1357 vs 0.3894).

Open-Jev's own synthetic versions of the four typed-decisions workflows
(`workflow-controls-v1/security_incidents`, `invoice_processing`, `customer_service` and
`agent_trace_observability`) score 0.9910, 0.9925, 1.0000 and 0.9860 on Open-Jev test, near
99 to 100 percent. On the real teacher-labelled typed-decisions cases, oj-v1 only reaches
0.4215 accuracy. The synthetic controls do not transfer to the real typed-decisions task.

The weakest sources on Open-Jev test are tic_tac_toe-v1 (accuracy 0.5805, n=472) and
reasoning-control-v1 (accuracy 0.6814, n=681). trex_runner-v1 scores lower still (0.2500) but
has only 4 questions, too few to draw a conclusion from.

Held-out cross-entropy rose in the last two training epochs, from 0.1673 in epoch 2 to 0.1907
in epoch 3 and 0.2309 in epoch 4, while held-out accuracy kept rising every epoch: 0.9205,
0.9289, 0.9411, 0.9428.

Open-Jev's project reports its released 2B and 9B models on the same test and OOD splits, but
scores only one-hot rows, so those published figures are not directly comparable to the numbers
above.

Evidence in `reports/trinity-prime-20260921/`.

### Calibration

| | as shipped | temperature refit | 
|---|---|---|
| `laya` | 0.466 | **0.081** |
| `laya-multilingual` | 0.314 | **0.106** |

Both ship over-confident; `laya-multilingual` ships with no fitted temperatures at all. Refitting one temperature per (question type, option count) on held-out data is the single highest-value fix available, and takes ECE below Jev's measured 0.246.

### Option-order robustness

How often the answer changes when the options are permuted. Jev measured at 0.13.

| suite | laya | laya-multilingual |
|---|---|---|
| massive_intent.en | 0.150 | 0.230 |
| en.emotion | 0.040 | 0.090 |
| xnli.en | 0.000 | 0.015 |

At 20 options both are less order-stable than Jev — worth fixing with more aggressive option-order shuffling during training.

---

## Training on both datasets at once (fork, 2026-09-22)

Each single-dataset checkpoint is strong only on its own data (td-v1 scores 0.5455 on Open-Jev
test, oj-v1 scores 0.4215 on typed-decisions test), so mix-v1 trains on both:

```bash
peewee train --data data/typed-decisions/train.jsonl:4 --data data/open-jev/train.jsonl \
           --calibration-target label --base english --out runs/mix-v1 --max-len 1024 --head-max-len 256
```

`:4` repeats typed-decisions' 1,200 training cases four times per epoch, after the held-out
split, lifting them from 7% to 21% of the 92,827 training items; the remaining 69,473 are
Open-Jev's. 1,463 cases (8,339 items) were held out for calibration, 0 questions were skipped,
and training took 6,079 seconds (101.3 minutes) on one RTX 5090 in bf16, defaults otherwise.

| checkpoint | typed-decisions test | Open-Jev test | Open-Jev OOD |
|---|---|---|---|
| published `typed-decisions` | 0.7685 | — | — |
| td-v1 (typed-decisions) | 0.7560 | 0.5455 | 0.6193 |
| oj-v1 (Open-Jev) | 0.4215 | 0.9390 | 0.8274 |
| **mix-v1 (both)** | **0.8005** | **0.9403** | **0.8310** |

Mixing wins everywhere. On typed-decisions mix-v1 beats not just td-v1 (+0.0445) but the
published checkpoint (+0.0320) and the 0.735 teacher self-agreement ceiling, while matching
oj-v1 on Open-Jev test (+0.0013) and OOD (+0.0036). The gain is largest on the question types
td-v1 was weakest at: typed-decisions score-type accuracy goes 0.7075 -> 0.7788, choice
0.7350 -> 0.7667 and noul 0.8417 -> 0.8633. Open-Jev's synthetic controls transfer as general
practice at reading a state under a typed question, even though a model trained on them alone
answers real teacher-labelled cases at 0.42.

| metric | td-v1 | oj-v1 | mix-v1 |
|---|---|---|---|
| typed-decisions ECE | 0.1357 | 0.3894 | 0.1876 |
| Open-Jev test ECE | 0.1080 | 0.0122 | 0.0128 |
| Open-Jev OOD ECE | 0.0809 | 0.1309 | 0.1354 |

**One temperature map cannot serve both datasets.** mix-v1 was calibrated with
`--calibration-target label`, the setting that fits temperatures to the answer accuracy is
scored against, and held-out ECE still fell only 0.0231 -> 0.0178 — because 86% of the
held-out items are Open-Jev's, whose near one-hot targets pull the temperatures towards
confidence that typed-decisions does not earn. So typed-decisions ECE is worse than td-v1's
(0.1876 vs 0.1357) even though accuracy is much better. Calibrate per workload: fit on the
workload's own data with `peewee train --calib-data`, or afterwards with `Agent.fit_temperatures`
and `peewee_decide.load(..., calibration=...)`. Accuracy is unaffected — temperature scaling never moves
an argmax.

Evidence in `reports/trinity-prime-20260922/`.

---

## Measured against Jev (fork, 2026-09-22)

Jev 1.13.0 and Peewee mix-v1 answered the same 4,110 cases (28,057 questions). `peewee eval`'s
scorer graded both, using `scripts/jev_compare.py`. Method, caveats and raw reports are in
`reports/jev-vs-peewee-20260922/`. Jev's answers are used only for this comparison, never for
training or calibration.

| accuracy | typed-decisions | Open-Jev test | Open-Jev OOD |
|---|---|---|---|
| **Peewee mix-v1** | **0.8005** | **0.9403** | **0.8310** |
| Jev 1.13.0 (measured) | 0.7385 | 0.8104 | 0.8031 |
| — choice (Peewee / Jev) | 0.767 / 0.737 | 0.841 / 0.637 | 0.750 / 0.750 |
| — score | 0.779 / 0.696 | 0.944 / 0.832 | 0.839 / 0.647 |
| — noul | 0.863 / 0.797 | 0.977 / 0.871 | 0.855 / 0.860 |
| ECE (Peewee / Jev) | 0.188 / **0.045** | **0.013** / 0.022 | 0.135 / **0.049** |
| mean confidence (Peewee / Jev) | 0.613 / 0.756 | 0.947 / 0.800 | 0.966 / 0.771 |
| same answer | 74.3% | 80.4% | 71.6% |

**Wins and losses:**

- **Accuracy:** Peewee is ahead on all three splits. Two of them (typed-decisions and Open-Jev
  test) come from the data it was trained on, so OOD is the fairest comparison. There, Peewee
  leads 0.831 to 0.803, and nearly all of that lead is on score questions (0.839 against 0.647).
  Choice and noul are even.
- **Calibration:** Jev is better calibrated on typed-decisions and on OOD.
  - On OOD, mix-v1 is overconfident: 0.966 mean confidence against 0.831 accuracy.
  - On typed-decisions, it is underconfident, with the temperatures fitted mostly on Open-Jev
    (see [Training on both datasets at once](#training-on-both-datasets-at-once-fork-2026-09-22)).
  - The ECE figures published earlier (Peewee 0.081, Jev 0.144) came from different samples and
    prompts. On the same cases, Jev is the better-calibrated system except on Open-Jev test.
- **Latency:** Jev took 280–286 ms per case at p50 over HTTPS, the same whether its first 200 cases
  were sent one at a time or 5 at a time. Peewee took 10–17 ms per case in-process on the RTX 5090.
- **Cost:** 4.50M Jev input tokens, about $0.19.

### Calibrating mix-v1 per workload (2026-09-23)

`peewee calibrate` fits temperatures on labelled cases the model never trained on:

- the 130 typed-decisions cases (650 questions) that mix-v1's training held out;
- Open-Jev's official `calibration` split (1,081 cases, 4,672 questions);
- a pool of both;
- a *balanced* pool, with the typed-decisions cases repeated 7× to match Open-Jev's size.

Every fit targets the reference answer (`--target label`), and every row was scored on the untouched
test splits. Temperatures never change accuracy, which stayed at 0.8005, 0.9403 and 0.8310
throughout.

| ECE with these temperatures | typed-decisions | Open-Jev test | Open-Jev OOD |
|---|---|---|---|
| the map from training (86% Open-Jev held-out) | 0.188 | 0.013 | 0.135 |
| typed-decisions fit | **0.021** | 0.031 | 0.152 |
| Open-Jev fit | 0.274 | **0.011** | 0.133 |
| pooled | 0.173 | 0.016 | 0.140 |
| **balanced (the published default)** | 0.055 | 0.023 | 0.147 |
| *Jev 1.13.0, measured* | *0.045* | *0.022* | *0.049* |

What the table shows:

- **Per-workload temperatures work.** Fitted on each dataset's own data, mix-v1 is better calibrated
  than Jev on both in-distribution splits.
- **The two datasets need opposite corrections.** typed-decisions needs sharpening (T < 1) and
  Open-Jev needs softening, so a pool dominated by Open-Jev cannot serve both. Balancing the pool
  gives a usable single default.
- **No fit fixes OOD.** mix-v1's mean confidence there is 0.98 against 0.83 accuracy. Fitting on
  familiar data cannot correct overconfidence on unfamiliar tasks, which is why the model card tells
  users to calibrate on their own cases.
- **Sharpening costs Brier.** It improves ECE against the reference answers but moves probabilities
  away from typed-decisions' soft teacher distributions: Brier goes from 0.057 to 0.083.

The published checkpoint (`roadius/peewee-mix-v1`) has the same weights as `runs/mix-v1`. It carries
the balanced temperatures in `rl_agent_config.json`, and the typed-decisions and Open-Jev fits under
`calibration/`. The same files are in this repo as `calibration/mix-v1-*.json`, and every evaluation
is in `reports/mix-v1-calibration-20260923/`. Downloaded from the Hub into an empty cache, the
checkpoint reproduced typed-decisions accuracy 0.8000 and ECE 0.0545 on an M5 Max.

### Apple silicon (M5 Max, 2026-09-22)

On a MacBook Pro with an M5 Max (18 cores, 128 GB), mix-v1 runs on MPS in fp32. On the same cases
it matches the RTX 5090's accuracy to within 0.001, and 0.3% of answers differ (bf16 against fp32).

| | typed-decisions | Open-Jev test | Open-Jev OOD |
|---|---|---|---|
| p50 per case, M5 Max (MPS) | 299 ms | 133 ms | 156 ms |
| p50 per case, RTX 5090 | 17 ms | 10 ms | 11 ms |

Worst case, with 1,024-token states:

- **M5 Max, MPS:** about 3–5 questions per second at any batch size, using 6.7 GB of unified
  memory at a batch of 32.
- **M5 Max, CPU:** about 1 question per second.
- **Speed-up measured but not enabled:** bf16 autocast on MPS roughly doubles throughput (184 → 110
  ms per real case, 3.0 → 6.1 questions/s) and moves confidences by at most 0.025. fp16 autocast
  and half-precision weights crash inside Metal Performance Shaders.

The same worst case on the RTX 5090, as GPU memory for the whole process:

| questions per pass | memory |
|---|---|
| model loaded, idle | 2.2 GB |
| 1 | 3.1 GB |
| 32 (`PEEWEE_MAX_BATCH`) | 5.1 GB |
| 64 | 7.8 GB |

Each additional loaded checkpoint adds about 1.6 GB. The weights stay in fp32 on the GPU and only
the compute runs in bf16.

## Limits, stated plainly

- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400).
- **Keep `choice` questions under ~20 options.**
- **Both checkpoints ship over-confident.** Fit temperatures on your own data.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route.
