# Changelog

All notable changes to Peewee. It began as a fork of `NandhaKishorM/laya`, diverging at
upstream v0.3.4 (`d113dca`).

## 0.4.0.dev0 (unreleased): Peewee

### Changed
- **The project is now Peewee.** The fork has its own name, README and banner
  (`docs/assets/peewee-banner.svg`), and is managed here rather than tracked against upstream
  branding. Concretely: the Python package is `peewee_decide` (was `laya`), the distribution is
  `peewee-decide`, the console script is `peewee` (`peewee serve|train|eval|export-onnx|prepare-data`),
  service environment variables are `PEEWEE_*` (was `LAYA_*`), loggers are `peewee`, `peewee.router`,
  `peewee.serving`, `peewee.onnx` and `peewee.train`, Prometheus metrics are `peewee_*`, the service
  response `model` field is `peewee-rl-agent`, and `LayaClient` / `LayaServiceError` are
  `PeeweeClient` / `PeeweeServiceError`. Upstream checkpoint ids (`convaiinnovations/laya`, and the
  `laya`, `laya-multilingual`, `laya-typed-decisions` routing aliases) are unchanged — they name
  someone else's artefacts. Upstream's Laya brand assets were removed from `assets/`; the upstream
  notebook and the full commit history stay. Attribution is in `NOTICE` and the README footnote.

### Fixed
- `peewee export-onnx --help` crashed: the `--quantize` help text had an unescaped `%`.
- `peewee eval` failed on choice questions whose criteria list non-string keys (`[1, 2]`): it looked
  the answers up by stringified key, while the runtime reports the keys as given.
- `peewee prepare-data open-jev` rejected score questions with repeated level texts; only choice
  options need to be distinct.
- The held-out split rounds half a group up (`round(2.5)` used to give 2). A split whose size lands
  exactly on .5 now holds out one more group than before.

### Added
- **mix-v1 is published** on the Hugging Face Hub as `roadius/peewee-mix-v1` (Apache 2.0). It is the
  model name `mix-v1` (aliases `peewee`, `mix`), so `peewee serve --models mix-v1`,
  `peewee eval mix-v1` and `peewee_decide.load("roadius/peewee-mix-v1")` all work. Its built-in
  temperatures are fitted on a balanced pool of typed-decisions and Open-Jev held-out data (ECE
  0.188 → 0.055 on typed-decisions test, accuracy unchanged). Per-dataset calibration files ship
  with it and are in `calibration/mix-v1-*.json`. `Router(preload=True)` now also loads mix-v1;
  the `peewee serve` default (`english,multilingual`) and language routing are unchanged.
- `peewee calibrate MODEL --data FILE --out calib.json [--target label|probabilities]` fits
  temperatures for a workload from labelled cases. `peewee eval --calibration FILE` scores with
  them. `peewee train` now fits its held-out temperatures through the same code
  (`peewee_decide.calibrate.fit_on_cases`).
- `scripts/jev_compare.py` benchmarks Peewee against TypeSafe's Jev on the same cases and scores
  both with `peewee eval`'s scorer. It runs Jev over the API (resumable, and splits cases over the
  request limit), runs a Peewee checkpoint, and reports accuracy, ECE, Brier, agreement, latency and
  tokens. The first measured results (mix-v1 against Jev 1.13.0, on an RTX 5090 and an M5 Max)
  are in `reports/jev-vs-peewee-20260922/` and `BENCHMARKS.md`. The README's Jev figures are now
  measured rather than published. On the same cases Jev is the better-calibrated system, which
  replaces the earlier published-figure comparison that said the opposite.
- `peewee eval` reports `latency_ms_per_question` next to `latency_ms` (which stays per case).
- Tests for the renamed surface end to end: the installed `peewee` console script and
  `python -m peewee_decide`, every subcommand's `--help`, `peewee serve` handing its flags to the app
  as `PEEWEE_*` variables, and `peewee export-onnx` on a checkpoint directory answering like the source.
- Mixed-dataset training is the best recipe measured so far: typed-decisions upsampled 4x plus
  Open-Jev (mix-v1) scores 0.8005 on typed-decisions test — above the published checkpoint's
  0.7685 — while matching the Open-Jev specialist on Open-Jev test (0.9403) and OOD (0.8310).
  Evidence in `reports/trinity-prime-20260922/`, write-up in `BENCHMARKS.md`.
- `peewee train` mixes datasets and chooses what calibration fits to. `--data` repeats, and
  `--data FILE:N` upsamples that file's training cases N times per epoch (after the held-out
  split). `--calib-data FILE` calibrates on a given file, such as Open-Jev's official calibration
  split, instead of holding out part of `--data`; overlapping ids or groups are rejected.
  `--calibration-target label` fits temperatures to the reference answer instead of the teacher
  distribution (the default, `probabilities`): fitting to the teacher raised typed-decisions'
  hard-label ECE from 0.170 to 0.184. `train_meta.json` records every data file's hash, the
  calibration file, and training items before and after upsampling.
- `peewee_decide.patterns.speculative_choice`: ask a primary `choice` and every option's follow-up
  question in one forward pass, then keep only the follow-up matching the chosen option. This
  is the "speculative fan-out" pattern from browser-use's `jev-ultrafast` agent (operation +
  per-operation target in one request); on Peewee the extra questions are free because a call
  is one pass regardless of question count.

## 0.4.0.dev0 (unreleased): Phase 2, the decision service

### Added
- **`peewee train`** (`peewee_decide/train.py`): the notebook's RLCD fine-tuning loop as a tested single-GPU
  command. Reads JSONL cases (`peewee_decide/data.py`), holds out whole cases for calibration, trains in
  bf16 where the GPU supports it, fits per-bucket temperatures on the held-out cases, and writes a
  checkpoint the runtime loads plus `train_meta.json`. Unlike the notebook it trains at the
  configured `max_len` instead of preprocessing at the base's shorter length, and it never fits
  temperatures on training data. Reproduces the published typed-decisions checkpoint on one RTX
  5090 in 5 minutes: 0.7560 accuracy against the published checkpoint's 0.7685 under the same
  eval code (`BENCHMARKS.md`).
- **`peewee eval`** (`peewee_decide/evaluate.py`): accuracy, soft accuracy, Brier score, ECE and latency per
  question type and workflow for any checkpoint on JSONL cases.
- **`peewee prepare-data typed-decisions`**: writes the public typed-decisions dataset as JSONL,
  keeping both the teacher distribution and the hard label. `pip install "peewee-decide[train]"`.
  `peewee prepare-data open-jev` does the same for the public Open-Jev dataset (CC0, pinned
  revision), and the calibration split keeps variants of one case together. Training oj-v1 on
  this data is written up in `BENCHMARKS.md` under "Training on Open-Jev (fork, 2026-09-21)".
- **`peewee serve`** (`peewee_decide/serving.py`): a FastAPI service around a preloaded `Router` with one
  `DynamicBatcher` per checkpoint. Requests are routed in pure Python, queued per checkpoint, and
  flushed into `Agent.predict_many` when `PEEWEE_MAX_BATCH` questions are waiting or
  `PEEWEE_MAX_WAIT_MS` has passed. Endpoints: `POST /v1/decide`, `POST /v1/decide/batch`,
  `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus). Optional bearer auth
  (`PEEWEE_API_KEY`), state size limit, per-checkpoint calibration files, default truncation side.
  Inference runs in a thread pool (`PEEWEE_WORKERS`, default 1) so the event loop keeps accepting.
- **`peewee_decide.client.LayaClient`**: standard-library HTTP client with `decide`, `decide_many`, `health`.
- **ONNX export and runtime** (`peewee_decide/onnx_backend.py`): `peewee export-onnx <checkpoint> <dir>
  [--quantize]` writes a self-contained export directory; `OnnxAgent(dir)` is an `Agent` whose
  forward pass runs in ONNX Runtime (CUDA when available, else CPU) with the same `predict` and
  `predict_many` API. `PEEWEE_BACKEND=onnx` serves exports. The decision head's attention has an
  explicit, shape-dynamic implementation used during export because PyTorch's fused
  encoder-layer fast path bakes the sample sequence length into the graph.
- **Docker**: `docker/Dockerfile` (CPU), `docker/Dockerfile.cuda`, `docker/compose.yml`, with
  readiness-based health checks and a persistent hub cache volume.
- **`scripts/loadtest.py`**: concurrency load test reporting p50/p95/p99 and throughput.
- **`examples/litellm_guardrail.py`**: a LiteLLM pre-call guardrail and a model-picking helper
  that call the service.
- `pip install "peewee-decide[server]"`, `"peewee-decide[onnx]"`, and a `peewee` console script (`serve`,
  `export-onnx`).

### Changed
- Session notes left the repo: the hand-off document and bootstrap script are gone, Claude's
  project instructions and permissions are git-ignored local files, and the open questions moved
  to `REVIEW.md` section 7. CI now also runs on pushes to `claude/main`.
- `Agent.__init__` is split into `resolve_checkpoint` (locate/download, load config) and
  `Agent._init_common` (tokenizer, temperatures, truncation default) so alternative backends
  reuse them. `Agent.model_dir` records where the checkpoint was loaded from.

### Added (validation artefacts)
- `scripts/length_sweep.py`: the Phase 3 measurement, see `docs/GPU_VALIDATION.md`. Natural long
  IMDB reviews and short reviews behind neutral padding, per checkpoint, at each `max_len`. Result
  in `REVIEW.md` Phase 3: the defaults stay at 1,024 because the checkpoints cannot read evidence
  placed past their trained length (chance at 4,000 tokens), so Phase 3 is the fine-tune.
- `reports/trinity-prime-20260920/`: the first validation report on real weights (RTX 5090),
  and `calibration/{english,multilingual}.json`, temperatures fitted on MASSIVE by
  `scripts/gpu_validate.py`. Not yet loaded by default.

### Fixed
- `DynamicBatcher` now batches under load. Flushes were purely timer-driven (`PEEWEE_MAX_WAIT_MS`
  after the first queued request), so once the inference worker was busy the timer kept slicing
  the queue into passes of one or two requests that then waited behind each other. Arrivals now
  accumulate while a pass runs (one per `PEEWEE_WORKERS`) and go out together the moment it
  finishes. On an RTX 5090 with 32 concurrent clients this took the service from 467 to 1,338
  questions/s and p50 from 229 to 77 ms; numbers in `BENCHMARKS.md`.
- `--quantize` now writes a weight-only int8 export (`MatMulNBits` on the encoder's linear
  weights, activations and the decision head in fp32). The previous dynamic int8 recipe broke the
  GeGLU feed-forward blocks on the real checkpoints: probabilities moved by up to 0.91 and 13 to
  30% of argmaxes flipped. The new export tracks fp32 to within 0.06 on all three checkpoints at
  about 40% of the file size. It is not faster than fp32 on CPU; the gain is memory. Needs
  onnxruntime>=1.22 (Python >= 3.10) and the `onnx-ir` package, added to the `onnx` and `dev`
  extras; on an older runtime `--quantize` stops before exporting with an error saying so.
- `OnnxAgent` no longer drops to CPU quietly. An explicit `providers` list naming a provider the
  runtime lacks raises at construction, and the automatic choice logs a warning when torch can see
  a CUDA device but the installed `onnxruntime` wheel has no CUDA provider. The service maps
  `PEEWEE_DEVICE` onto the ONNX backend (`providers_for_device`), so `PEEWEE_DEVICE=cuda` with
  `PEEWEE_BACKEND=onnx` fails at startup instead of serving from CPU.
- `export_onnx` moved the caller's model to fp32 on CPU for tracing and left it there, so the
  next `predict` on a CUDA agent failed with a device mismatch. The model is moved back to its
  original device and dtype after export (found on the first real GPU run).
- `scripts/gpu_validate.py` loads MASSIVE from the hub's parquet conversion (one config, filtered
  by `locale`) because `datasets` 4+ no longer runs dataset scripts, and IMDB as
  `stanfordnlp/imdb` because the bare alias is gone.

## 0.4.0.dev0 (unreleased): Phase 1, safe to gate on

### Added
- **Truncation reporting.** `usage` now carries `state_tokens`, `state_tokens_dropped`,
  `truncated` and `truncation` (`"left"` or `"right"`). `predict(..., truncate="left")` keeps the
  end of an over-long state; `peewee_decide.load(..., truncate=...)` sets the default. With no setting,
  strings and dicts keep their start and lists (conversation transcripts) keep their end. The
  first truncation on an agent is logged at WARNING, later ones at DEBUG.
- **Option-budget reporting.** Questions whose options were squeezed below the 48-token cap to
  fit `head_max_len`, or whose option text exceeded the cap, are listed in
  `usage["option_budget"]` with the tokens per option actually used, and logged once.
  `build_sequence(..., return_info=True)` exposes the same detail.
- **Calibration API** (`peewee_decide.calibrate`, based on upstream PR #19 by mvanhorn):
  `collect_records` runs labelled `(state, questions, labels)` examples through an agent,
  `fit_temperature_map` fits one temperature per question type and per option-count bucket by
  NLL, `calibration_report` gives accuracy / ECE / NLL / Brier before and after, and
  `Agent.fit_temperatures`, `save_calibration`, `load_calibration` and
  `peewee_decide.load(..., calibration=path)` apply and persist the result. Labels may be an option
  key, a score level (fractional levels split between neighbours), a bool, a probability or a
  distribution.
- **Batch API.** `Agent.predict_many(requests, max_batch=64)` collates every question of every
  request into as few forward passes as possible and returns per-request results in order.
- **`peewee_decide.patterns`**: `hierarchical_choice` (pick a group, then pick within it) and
  `select_tool` (flat up to `max_flat` tools, grouped above that).
- **Language-detector plug-in.** `peewee_decide.register_language_detector(fn)` routes the Latin-script
  guess through an external detector such as lingua or fastText.
- `score` answers include `level`, the argmax level, next to the expected `score`.

### Changed
- **`confidence` has one meaning.** It is now the calibrated probability of the reported
  answer for every question type (top option for `choice` and `score`, `max(p, 1-p)` for
  `noul`). Previously `choice` and `score` reported normalised entropy, so a 90/10 two-way
  choice scored 0.53 while the identical `noul` scored 0.90. The entropy measure is kept as
  `entropy` (0 = certain, 1 = uniform) and `confidence_from_probs` is unchanged for old callers.
- **Short non-English Latin text now routes to the multilingual checkpoint.** The stop-word
  lists cover the short first-person words that dominate support messages, shared words
  ("la", "de", "no", "me") count half, strong markers ("merci", "ich", "gracias") decide on
  their own when no English function word is present, and diacritics count on short inputs.
  "Je veux annuler mon forfait", "Mon compte ne marche pas", "Necesito cancelar mi cuenta hoy"
  and "Ich kann mich nicht einloggen" all route multilingual; "refund me", "no problem" and
  "a la carte menu please" still route English.

## 0.4.0.dev0 (unreleased): Phase 0, hygiene

### Fixed
- `transformers` floor raised to 4.48.0, the first release with ModernBERT. On 4.45 to 4.47 the
  English and typed-decisions checkpoints could not load.
- `Router` is now thread-safe: loading, eviction, `attach`, `preload` and `unload` are
  serialised with a re-entrant lock, and concurrent requests for the same checkpoint build it
  once.
- `Router.preload` no longer evicts already-resident checkpoints when called incrementally
  (ports upstream PR #27 by wbx398).
- `Router.route` returns a string `repo` on the auto-detected-workflow branch, like every other
  branch, instead of the raw `(repo, subfolder)` tuple.
- Armenian script is recognised and routed to the multilingual checkpoint (ports upstream
  PR #23 by ROTl24, upstream issue #20).
- Tokenizer-config patching no longer writes into the shared Hugging Face cache. When a
  checkpoint's `tokenizer_config.json` needs patching, the tokenizer folder is copied to a
  per-process temporary directory and patched there. Failures are logged, not swallowed.
- Removed the duplicate `email_questions` from `peewee_decide.email`; `peewee_decide.presets` is the single
  definition and is what the package exports.

### Changed
- **`Router()` defaults to `max_loaded=2`** (was 1) so English and multilingual traffic no
  longer rebuild a model on every language switch. An eviction during `load` is logged at
  WARNING level with the evicted checkpoint's name.
- **The inference-time CPU fallback is gone.** A GPU error inside `predict` now propagates
  instead of silently moving a shared model to CPU for the rest of the process. The load-time
  fallback (model does not fit on the requested device at construction) is kept and reported
  via `logging` and `Agent.fell_back_to_cpu`.
- All warnings go through the `peewee` and `peewee.router` loggers instead of `print`.
- `Agent.system_one` is split into `_build_items` (tokenise), `_forward_logits` (one forward
  pass) and `_postprocess` (temperature, confidence, labels) so the pre- and post-processing
  can be tested without weights and so a batch API can reuse the same pieces.
- Version is single-sourced from `peewee_decide.__version__`; `setup.py` is removed.
- Python 3.8 dropped from the supported range (upstream CI never tested it).

### Tests and CI
- Tests moved to pytest. New weight-free coverage for `build_sequence` (layout, option caps,
  left and right truncation), `collate_items`, `Agent._postprocess`, tokenizer-config
  patching, router eviction logging and router thread safety. The `inspect`-based assertions
  on `Agent.__init__` source text are gone.
- Known gap recorded as a strict `xfail`: short French, Spanish and German sentences still
  route to the English checkpoint (Phase 1).
- CI and release workflows run `python -m pytest` and read the version from `peewee_decide/__init__.py`.
