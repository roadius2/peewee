# Changelog

All notable changes to this fork. Upstream is `NandhaKishorM/laya`; this fork diverged at
upstream v0.3.4 (`d113dca`).

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
