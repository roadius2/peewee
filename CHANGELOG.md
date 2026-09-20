# Changelog

All notable changes to this fork. Upstream is `NandhaKishorM/laya`; this fork diverged at
upstream v0.3.4 (`d113dca`).

## 0.4.0.dev0 (unreleased): Phase 1, safe to gate on

### Added
- **Truncation reporting.** `usage` now carries `state_tokens`, `state_tokens_dropped`,
  `truncated` and `truncation` (`"left"` or `"right"`). `predict(..., truncate="left")` keeps the
  end of an over-long state; `laya.load(..., truncate=...)` sets the default. With no setting,
  strings and dicts keep their start and lists (conversation transcripts) keep their end. The
  first truncation on an agent is logged at WARNING, later ones at DEBUG.
- **Option-budget reporting.** Questions whose options were squeezed below the 48-token cap to
  fit `head_max_len`, or whose option text exceeded the cap, are listed in
  `usage["option_budget"]` with the tokens per option actually used, and logged once.
  `build_sequence(..., return_info=True)` exposes the same detail.
- **Calibration API** (`laya.calibrate`, based on upstream PR #19 by mvanhorn):
  `collect_records` runs labelled `(state, questions, labels)` examples through an agent,
  `fit_temperature_map` fits one temperature per question type and per option-count bucket by
  NLL, `calibration_report` gives accuracy / ECE / NLL / Brier before and after, and
  `Agent.fit_temperatures`, `save_calibration`, `load_calibration` and
  `laya.load(..., calibration=path)` apply and persist the result. Labels may be an option
  key, a score level (fractional levels split between neighbours), a bool, a probability or a
  distribution.
- **Batch API.** `Agent.predict_many(requests, max_batch=64)` collates every question of every
  request into as few forward passes as possible and returns per-request results in order.
- **`laya.patterns`**: `hierarchical_choice` (pick a group, then pick within it) and
  `select_tool` (flat up to `max_flat` tools, grouped above that).
- **Language-detector plug-in.** `laya.register_language_detector(fn)` routes the Latin-script
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
- Removed the duplicate `email_questions` from `laya.email`; `laya.presets` is the single
  definition and is what the package exports.

### Changed
- **`Router()` defaults to `max_loaded=2`** (was 1) so English and multilingual traffic no
  longer rebuild a model on every language switch. An eviction during `load` is logged at
  WARNING level with the evicted checkpoint's name.
- **The inference-time CPU fallback is gone.** A GPU error inside `predict` now propagates
  instead of silently moving a shared model to CPU for the rest of the process. The load-time
  fallback (model does not fit on the requested device at construction) is kept and reported
  via `logging` and `Agent.fell_back_to_cpu`.
- All warnings go through the `laya` and `laya.router` loggers instead of `print`.
- `Agent.system_one` is split into `_build_items` (tokenise), `_forward_logits` (one forward
  pass) and `_postprocess` (temperature, confidence, labels) so the pre- and post-processing
  can be tested without weights and so a batch API can reuse the same pieces.
- Version is single-sourced from `laya.__version__`; `setup.py` is removed.
- Python 3.8 dropped from the supported range (upstream CI never tested it).

### Tests and CI
- Tests moved to pytest. New weight-free coverage for `build_sequence` (layout, option caps,
  left and right truncation), `collate_items`, `Agent._postprocess`, tokenizer-config
  patching, router eviction logging and router thread safety. The `inspect`-based assertions
  on `Agent.__init__` source text are gone.
- Known gap recorded as a strict `xfail`: short French, Spanish and German sentences still
  route to the English checkpoint (Phase 1).
- CI and release workflows run `python -m pytest` and read the version from `laya/__init__.py`.
