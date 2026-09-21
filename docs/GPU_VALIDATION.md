# Validating on a GPU box

Everything in this fork was developed and unit-tested without the real checkpoints (the
development sandbox could not reach the Hugging Face hub). `scripts/gpu_validate.py` closes
that gap in one run and produces artefacts to commit.

```bash
git clone https://github.com/roadius2/ultra_laya && cd ultra_laya && git checkout claude/main
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124     # match your CUDA
pip install -e ".[dev,server,onnx]" datasets
python -m pytest                                                          # 178 weight-free tests
python tests/test_local_e2e.py ~/laya_models   # optional: upstream's e2e script, needs local copies
python scripts/gpu_validate.py --out reports/$(hostname)-$(date +%Y%m%d)
```

The run takes roughly 20 to 40 minutes on one GPU, most of it the calibration and length
sweeps. Steps can be skipped (`--skip calibrate,lengths`) or shrunk (`--per-locale 100
--length-samples 100`).

## What comes back

| artefact | commit it? | what it answers |
|---|---|---|
| `reports/<host>/report.md`, `report.json` | yes | every question below |
| `reports/<host>/calibration/*.json` | yes, copy to `calibration/` | shipped temperatures for `english` and `multilingual` |
| `reports/<host>/onnx/<name>/` | no (large) | ONNX exports for the service |

## Questions the report answers

1. **Do the real checkpoints load and predict sensibly** after the Phase 0 and Phase 1
   refactors (`load` step: sample predictions in English, French, Hindi, German)?
2. **Latency on this GPU** per questions-per-call, and per question with `predict_many`.
3. **Does ONNX export work on ModernBERT / mmBERT**, and how close are fp32 and int8 to
   torch (`onnx` step: max probability difference, argmax agreement, CPU latency)? Parity was
   only verified on a small BERT-style model before this.
4. **Calibration**: temperatures fitted on MASSIVE validation across six locales, ECE and NLL
   on MASSIVE test before and after, per-locale accuracy. The resulting JSON is what
   `laya.load(..., calibration=...)` and `LAYA_CALIBRATION` consume.
5. **Phase 3, context length**: accuracy on IMDB reviews (many exceed 512 tokens) at
   `max_len` 512, 1024, 2048, 4096 with left and right truncation. If accuracy holds or improves
   past the trained length, the multilingual and typed-decisions defaults can be raised without
   retraining. If it collapses, Phase 3 needs the long-context fine-tune.

## Phase 3: the length sweep

`scripts/gpu_validate.py` step 5 runs on a random IMDB sample, which is too short to say
anything past 1,024 tokens (1% of reviews are cut there). `scripts/length_sweep.py` is the
targeted version:

```bash
python scripts/length_sweep.py --out reports/$(hostname)-$(date +%Y%m%d)
```

It writes `length_sweep.md` and `.json` next to the validation report, with two experiments per
checkpoint: the 874 labelled IMDB reviews of at least 1,024 tokens at each `max_len` and
truncation side, bucketed by review length; and short reviews placed after neutral news text so
the state is about 900, 1,900, 3,900 and 7,900 tokens long, answered with no truncation. The
label only depends on the review, so the second experiment isolates whether the model can still
read past the position it was trained to, and its control run (left-truncated at the trained
length) is what the service does today. About 15 minutes on one GPU.

## After the run

- Commit `report.md` and the calibration files; open a PR against the mainline.
- If ONNX parity is good, rebuild the Docker image with `LAYA_BACKEND=onnx` and run
  `scripts/loadtest.py` against it for CPU throughput numbers.
- Paste the length-sweep table into `REVIEW.md` Phase 3 so the next decision is made on data.
