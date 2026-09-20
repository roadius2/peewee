# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`ultra_laya` is a fork of Laya (`NandhaKishorM/laya`, Apache 2.0), a non-autoregressive
"System 1" decision model: typed questions (`choice` / `score` / `noul`) answered over a state
in one encoder forward pass, with calibrated probabilities. The fork's goal is a fast reflex
layer for agent workloads (tool routing, model routing, agent gating, spam and security
gating, a shared decision service, per-workload fine-tunes). The plan and its status live in
`REVIEW.md`; what changed and why is in `CHANGELOG.md`; the open hand-off is `docs/HANDOFF.md`.

Upstream history is preserved and the original authors are credited in `NOTICE`; keep it that
way. Ported upstream PRs are credited by author in commit messages and the changelog.

## Commands

```bash
pip install -e ".[dev,server,onnx]"        # dev includes pytest, ruff, fastapi, httpx, onnxruntime
python -m pytest                            # whole suite, ~10 s, no weights or network
python -m pytest tests/test_router.py -k concurrent     # one file / one test
python -m pytest -q -p no:warnings          # quieter
ruff check laya/ tests/ scripts/ examples/ --select=E9,F63,F7,F82,F401,F811 --line-length=120   # what CI runs
ruff check laya/ tests/ scripts/ examples/ --line-length=120                                     # stricter, advisory
python -m build --wheel                     # package; version comes from laya/__init__.py
laya serve --models english,multilingual    # the service (downloads weights on first run)
laya export-onnx convaiinnovations/laya ./out --quantize
python tests/test_local_e2e.py ~/laya_models            # upstream's e2e script; needs local weights
python scripts/gpu_validate.py --out reports/$(hostname) # full validation on a GPU box; see docs/GPU_VALIDATION.md
```

Use `python -m pytest`, not the bare `pytest` binary: on some machines it belongs to a different
interpreter and cannot import `laya`. `tests/test_local_e2e.py` is excluded from collection in
`tests/conftest.py` because it needs real weights.

The Hugging Face hub was unreachable from the environment that built Phases 0 to 2, so nothing
in the suite loads real checkpoints. Everything that touches the model is tested through a
fake tokenizer (`tests/conftest.py`), synthetic logits, a stub `Router._build`, or a tiny random
BERT-style `DecisionModel` (`tests/test_onnx.py`). Keep new tests weight-free too.

## Architecture

**Inference pipeline** (`laya/agent.py`) is three explicit stages, and everything else hangs
off them:

1. `Agent._build_items(state, questions, truncate)` tokenises each question against the state
   with `build_sequence` (`laya/common.py`): `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK]
   opt1 ... [SEP] state [SEP]`. Each option gets a `[MASK]` marker whose hidden state is scored.
   The option prompt competes with the state for `max_len` (`head_max_len` caps the options).
   `build_sequence(..., return_info=True)` reports what was cut; items carry that `info`.
2. `Agent._forward_logits(items)` collates and runs one forward pass, returning raw logits.
   It never falls back to CPU on a GPU error; the caller handles failures.
3. `Agent._postprocess(...)` applies temperatures (`temperature` per type, overridden per
   `(type, option-count)` bucket from `temperature_by_options`), builds answers, and fills
   `usage` with token counts, truncation and option-budget reports.

`predict` runs the three in sequence; `predict_many` builds items for many requests, forwards
them in chunks, and post-processes per request. `OnnxAgent` (`laya/onnx_backend.py`) subclasses
`Agent`, reuses stages 1 and 3, and replaces stage 2 with ONNX Runtime. Any new backend should
do the same. `resolve_checkpoint` and `Agent._init_common` are the shared constructor pieces.

**Semantics to preserve.** `confidence` is the probability of the reported answer for every
question type (top option for `choice`/`score`, `max(p, 1-p)` for `noul`); `entropy` is the
normalised entropy. Temperature scaling never changes an argmax. Default truncation keeps the
start of a string/dict state and the end of a list state (transcripts).

**Routing** (`laya/router.py`, `laya/lang.py`) is pure Python and runs before any weights load.
Precedence: explicit `model` > `task` > detected typed-decisions workflow (opt-in) > `lang` >
script/language detection > default. Script detection is exact; the Latin-script language
guess is a weighted stop-word heuristic biased towards catching non-English (the English
checkpoint collapses on other languages, the multilingual one is only slightly weaker on
English). `register_language_detector` plugs in an external detector. `Router` holds an LRU of
loaded agents behind a re-entrant lock; `_build` is the seam tests stub.

**Service** (`laya/serving.py`) wraps a preloaded `Router` in `DecisionService`: one
`DynamicBatcher` per checkpoint collects requests and flushes them into `predict_many` at
`max_batch` questions or after `max_wait_ms`, in a thread pool. `create_app` is the FastAPI
factory; configuration is `Settings.from_env` (all `LAYA_*` variables documented in the module
docstring). Prometheus collectors are process-wide singletons (`_prometheus_collectors`).

**Calibration** (`laya/calibrate.py`): `collect_records` runs labelled examples through an
agent and keeps raw logits; `fit_temperature_map` fits one temperature per type and per
bucket by NLL; `Agent.save_calibration` / `load_calibration` persist JSON that
`laya.load(..., calibration=)` and `LAYA_CALIBRATION` consume.

**ONNX export** needs the decision head's attention in plain ops (`_manual_encoder_layer` in
`laya/common.py`, enabled via `DecisionModel.manual_head_attention` only during export),
because PyTorch's fused encoder-layer fast path traces with the sample's sequence length
baked in.

## Conventions

- Commit as the repository owner with Claude as co-author (see `docs/HANDOFF.md` for the exact
  identity lines). Never put model identifiers in code or docs.
- Work on `claude/main` and merge to the mainline through pull requests once the branch
  ruleset is in place; do not force-push a branch anyone else has pulled.
- `CHANGELOG.md` gets an entry for every user-visible change; `REVIEW.md` phase headings get a
  status note when a phase completes or changes.
- Warnings go through `logging` (`laya`, `laya.router`, `laya.serving`, `laya.onnx` loggers),
  never `print`.
