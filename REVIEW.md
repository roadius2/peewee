# Peewee v0.3.4 review and improvement plan for ultra_laya

This repo starts as a verbatim copy of upstream Laya (`NandhaKishorM/laya`, commit `d113dca`,
v0.3.4) with full history. This document is the review of that starting point and the plan
for turning it into the fast System-1 reflex layer described in the project goals:

- Tool routing, model routing, agent gating, tool-result verification
- Claude Code supervisor (route files/tests, validate coding-agent progress cheaply)
- Email/SMS spam, phishing, scam and abuse scoring
- Security gate (allow / review / quarantine / escalate)
- One high-throughput decision service behind hundreds of agents
- Fine-tuned per-workload classifiers

Everything below was checked against the code in this tree. Where something could not be
measured (the Hugging Face hub is unreachable from the review sandbox), it is marked as an
estimate.

---

## 1. Verdict

Peewee is a small, readable, honest codebase (about 1,100 lines of library code) with the right
shape for a self-hosted decision model: typed questions, one forward pass, calibrated-ish
probabilities, Apache 2.0 weights. The authors' own benchmark write-up is candid about its
limits.

It is **not production-ready for the goals above as shipped**. The blockers, in order of how
much they matter for this project:

1. **Context is tiny.** 512 tokens on the English checkpoint, 1,024 on the others, and the
   option prompt eats up to 192/256 of that. Anything longer is silently cut. This alone rules
   out agent gating on transcripts, tool-result verification and diff review until it is
   addressed.
2. **No service layer.** Single-state, single-process, not thread-safe, no batching across
   requests, no HTTP API, no ONNX/CPU path. "Hundreds of agents" cannot sit behind it yet.
3. **Calibration is not usable out of the box.** The headline ECE of 0.081 is after refitting
   temperatures on labelled data. The multilingual checkpoint ships with no temperatures at
   all, and there is no calibration API in the library. Gating on `confidence` today is gating
   on a number that has not been fitted.
4. **The language router misses short non-English Latin text** (French, Spanish, German
   sentences under a dozen words go to the English model). For Quebec French traffic this is
   a correctness bug, not a nicety.
5. **Zero-shot accuracy on generic typed decisions is near chance.** Base checkpoints score
   0.36 against a 0.32 random baseline on the typed-decisions benchmark. The value comes from
   fine-tuning per workload, which the repo supports only through a Kaggle notebook.

None of these are architectural dead ends. Items 2, 3 and 4 are straightforward engineering.
Item 1 has a real path (the encoders support 8k positions; only the training length was
short). Item 5 is the actual project: build labelled data per workload and fine-tune.

---

## 2. What is solid

- **Sequence design** (`peewee_decide/common.py:49`, `build_sequence`). Options are marked with `[MASK]`
  tokens and scored from their hidden states, so any number of options is handled without a
  fixed classifier head. Question type is injected via an embedding. Clean.
- **Strict checkpoint loading** (`peewee_decide/agent.py:53`, `_verify_compatibility`). Safetensors
  only, shape-checked, `strict=True`. Good hygiene for a model you will self-host.
- **Routing is pure and testable** (`peewee_decide/router.py`, `Router.route`). No weights loaded to
  decide; the decision object carries the reason. The precedence order (explicit model > task >
  workflow > lang > detection > default) is sensible.
- **Script detection is exact** (`peewee_decide/lang.py`, `detect_script`). Non-Latin scripts always go
  to the multilingual model, which is the case that matters most given the English model's
  confident collapse on them.
- **Presets** (`peewee_decide/presets.py`) are a decent starting vocabulary for model routing,
  guardrails, moderation and triage.
- **CI** runs on three Python versions, lints, byte-compiles and builds the package.
- **Benchmarks doc** does not hide the bad numbers.

---

## 3. Defects and gaps

Ordered by impact on this project. File references are to this tree.

### 3.1 Silent truncation of state
`peewee_decide/common.py:84`. The state is cut to whatever room remains after the option prompt, with
no warning and nothing in the result to say it happened. `usage.input_tokens` reports what
was fed, not what was dropped. A `truncate_left` flag exists on `build_sequence` but `Agent`
never exposes it, so a conversation transcript is always cut from the end, which is exactly
the part an agent gate needs.

Estimated budget at defaults (ModernBERT 50k BPE, roughly 4 characters per English token,
about 3 per token for code or JSON): with six tool options of one line each plus a one-line
instruction, about 400 tokens remain for state on the English model, which is roughly 1,600
characters of prose or 1,200 characters of code. A single medium tool result or a small diff
exceeds that. This is an estimate; measure it with the real tokenizer once the hub is reachable.

### 3.2 Language guess misses short Latin-script non-English
`peewee_decide/lang.py:133`, `guess_latin_language`. Needs at least four words, then a stop-word margin
of two over English. Verified on this tree:

| input | routed to |
|---|---|
| Je veux annuler mon forfait | english |
| Mon compte Fongo ne marche pas | english |
| Bonjour, je n'arrive pas à me connecter à mon compte depuis hier | english |
| Pouvez-vous me rappeler demain matin? | english |
| Necesito cancelar mi cuenta hoy | english |
| Ich kann mich nicht einloggen | english |
| Muchas gracias por su ayuda con la factura | multilingual |

The stop-word lists lack the most common short-message words (`je`, `mon`, `ma`, `mes`, `ne`,
`ich`, `mich`, `mi`, `hoy`...) and diacritics only count once a 2% character rate is reached,
which a short sentence rarely hits. Since the multilingual model is only slightly weaker on
English (0.657 vs 0.783 on MASSIVE) while the English model collapses on other languages, the
tie-break should lean multilingual for short ambiguous inputs.

### 3.3 Permanent CPU fallback on a single GPU error during inference
`peewee_decide/agent.py:278-289`. If one request raises a `RuntimeError` whose text contains "memory"
or "cuda", the model is moved to CPU and stays there for the life of the process. Every
subsequent request is 10 to 15 times slower and the only signal is a `print`. In a shared
service one oversized batch degrades the whole fleet. The load-time fallback (`agent.py:203`)
is a reasonable choice; the inference-time one is not.

### 3.4 Router is not thread-safe, and the default evicts on every language switch
`peewee_decide/router.py:169-198`. `load`, `_touch` and `_evict` mutate `_agents` and `_order` with no
lock. Two threads loading different checkpoints can each build a model, then evict each
other's. The default `max_loaded=1` (`router.py:149`) means alternating English and
non-English traffic reloads a model per request (the README itself measures 7 to 10 seconds
per reload). `preload=True` is documented but not the default.

### 3.5 Wrong dependency floor
`pyproject.toml:30` and `setup.py:15` declare `transformers>=4.45.0`. ModernBERT was added in
transformers 4.48.0 (verified by inspecting the 4.47.1 and 4.48.0 wheels). On 4.45 to 4.47
the English and typed-decisions checkpoints fail to load.

### 3.6 Inconsistent `confidence` semantics across question types
`peewee_decide/agent.py:309` and `agent.py:335`. For `choice` and `score`, confidence is normalised
entropy, `1 - H(p)/log(k)`. For `noul` it is `max(p, 1-p)`. These are not on the same scale:
a two-option `choice` at 0.90/0.10 has entropy-confidence 0.53, while the identical `noul` has
confidence 0.90. The README's gating example (`if conf >= 0.85`) will behave very differently
depending on which type the question happens to be. For gating decisions this needs one
definition, and it should be the calibrated top probability.

### 3.7 No calibration in the library
Temperatures are read from the checkpoint config (`agent.py:194-195`) and applied per
(type, option-count) bucket, but there is no function to fit them, no way to persist a fitted
set, and the multilingual checkpoint ships with none. The fitting code lives only in the
Kaggle notebook. Calibration is the whole reason to prefer a probability model over an LLM
for gating, so this belongs in `peewee_decide/`.

### 3.8 No batching across states
`Agent.system_one` (`agent.py:241`) takes one state and N questions. `collate_items` already
pads a batch of arbitrary items, so the model can score many (state, question) pairs in one
pass, but the public API never does. For a service, throughput is bounded by per-call latency
instead of GPU occupancy. The README's own numbers (39 ms for one question, 15.9 ms per
question at ten) show what batching buys.

### 3.9 No server, no CPU-optimised path
There is no HTTP or gRPC entry point, no Docker image, no ONNX export. The model is a plain
encoder plus a two-layer transformer head and MLPs, all standard ops, so ONNX Runtime with
int8 quantisation is feasible and would make the CPU path (currently 200 to 500 ms) viable for
a shared service.

### 3.10 Smaller defects
- `peewee_decide/router.py:266`: when `auto_task_detection` matches a workflow, `repo` is the raw
  `(repo, subfolder)` tuple rather than the string `_repo_str` produces elsewhere. Serialises
  as a list in JSON and breaks equality with the other branches.
- `peewee_decide/agent.py:21-46`, `_fix_tokenizer_config`: rewrites `tokenizer_config.json` inside the
  shared Hugging Face cache on every load and swallows every exception. Should patch the
  loaded dict in memory, or copy to a private directory, and log failures.
- `peewee_decide/email.py:56` and `peewee_decide/presets.py:45` both define `email_questions`; the package
  exports the `email.py` one. One should go.
- `agent.py:310`, `act_probability`: an undocumented output from an "action head" whose
  meaning (`act_costs` in the training config) is nowhere explained. Either document it or
  drop it from the public result.
- Warnings go through `print` (`agent.py:159,162,219,280`). Use `logging` or `warnings` so a
  service can route them.
- Version string is duplicated in `pyproject.toml`, `setup.py` and `peewee_decide/__init__.py`.
  `setup.py` is redundant with `pyproject.toml`.
- Tests are hand-rolled scripts with a global pass/fail list rather than pytest. One
  "test" (`tests/test_criteria.py`, CPU-fallback section) asserts on the source text of
  `Agent.__init__` via `inspect`, which breaks on any refactor. Nothing exercises
  `build_sequence` or the answer post-processing without downloading weights, so CI never runs
  a forward pass.
- Option text is hard-capped at 48 tokens per option (`common.py:69`) with no warning, and
  when the option budget is exceeded options are cut to as few as four tokens each
  (`common.py:71-74`). This is the mechanism behind the Banking77 collapse (0.425 vs Jev's
  0.870 on 77 labels), and it will bite tool routing over a large tool set.

---

## 4. Fit against each intended use

| Use | Fit today | What it needs |
|---|---|---|
| Model routing (Qwen vs reasoning vs Claude) | Good. `router_questions()` preset, short input, few options. | Calibrate on your own traffic; fine-tune once you have outcome labels (which route was right). |
| Tool routing | Fair for under ~20 tools. | Hierarchical choice (category then tool) above that; raise `head_max_len`; per-option token budget check. |
| Email/SMS spam, phishing, scam, abuse | Good. `email_questions()`, `clean_email_body`. Inputs usually fit. | Calibration on labelled mail; multilingual routing fix for French; fine-tune on your corpus. This is the best first production target. |
| Security gate (allow/review/quarantine/escalate) | Good on short prompts via `guard_questions()`. | Same as above; add a 4-way `choice` for the disposition; unified confidence for thresholds. |
| Agent gating (continue/stop/retry/escalate) | Poor. Transcripts do not fit; truncation drops the end. | `truncate_left`, truncation reporting, longer context (section 5, phase 3), a compact "state summary" format instead of raw transcript. |
| Tool-result verification | Poor. Task plus result rarely fits 400 tokens. | Same as agent gating. Feed (task, result excerpt, expected shape) rather than raw output. |
| Claude Code supervisor (route files/tests, validate progress) | Poor for diffs; fair for file-path or test-name routing. | Long context or chunk-and-aggregate; fine-tune on your own repos' history. |
| High-throughput central service | Not present. | Phase 2 below. |
| Specialised fine-tuned classifiers | Possible via notebook only. | `peewee_decide/train.py` with a documented dataset schema and a teacher-labelling script. |

Where Jev still wins outright: 64k context and 255 options in one call. Where Peewee wins:
latency (about 7x faster per question on a T4), zero marginal cost, data never leaves your
infrastructure, and you can fine-tune. The plan is to close the context and cardinality gaps
enough for the workloads above, not to match Jev generally.

---

## 5. Improvement plan

Phases are ordered so that each one leaves the library usable. Phase 0 and 1 are prerequisites
for trusting any decision the model makes. Phase 2 makes it a service. Phase 3 attacks the
context problem. Phase 4 is where the accuracy comes from.

### Phase 0: hygiene (about a day) — done, see `CHANGELOG.md`
- Fix the transformers floor to `>=4.48.0`; drop `setup.py`; single-source the version.
- Add a lock around `Router.load` / `_evict` / `attach` / `unload`.
- Default `max_loaded=2` (English and multilingual), and make `Router()` warn once when it
  evicts during `predict`.
- Remove the inference-time CPU fallback in `system_one`. Raise the error; let the caller (or
  the server) decide. Keep the load-time fallback but route it through `logging`.
- Return a string `repo` from the workflow branch of `Router.route`.
- Make `_fix_tokenizer_config` patch in memory; never write into the hub cache.
- Delete the duplicate `email_questions`.
- Move tests to pytest; delete the `inspect`-based assertions; add a fake tokenizer fixture so
  `build_sequence`, truncation and post-processing run in CI without weights.
- Replace `print` with `logging`.

### Phase 1: make it safe to gate on — done except shipping fitted multilingual temperatures (needs the weights), see `CHANGELOG.md`
- **Truncation reporting.** `build_sequence` returns how many state tokens were dropped;
  `usage` gains `state_tokens`, `state_tokens_dropped`, `truncated: bool`. Add a
  `truncate="right"|"left"` argument to `predict` and default agent-style states (a list of
  turns) to left truncation.
- **One confidence definition.** `confidence` becomes the calibrated top probability for every
  type. Keep entropy as `entropy` if anyone wants it. Document it.
- **Calibration API.** `peewee_decide.calibrate(agent, examples) -> temperatures` fitting one
  temperature per (type, option-count) bucket by NLL, `agent.save_calibration(path)`,
  `agent.load_calibration(path)`, and an `ece` report. Ship fitted temperatures for the
  multilingual checkpoint in this repo.
- **Fix the Latin language guess.** Extend stop-word lists with high-frequency short-message
  words per language, count diacritics on short inputs, and lower the bar: when a non-English
  language scores at least as high as English on an input under about 12 words, route
  multilingual. Add the French/Spanish/German cases in section 3.2 as regression tests. Offer
  an optional `lingua`/`fasttext` backend for anyone who wants a real detector.
- **Batch API.** `predict_many(requests: list[(state, questions)])` that collates everything
  into one forward pass and un-batches the answers. Keep `predict` as the one-request wrapper.
- **Option budget guard.** Warn (and expose in `usage`) when options are cut below a per-option
  token floor; document the hierarchical-choice pattern with a helper.

### Phase 2: the decision service — done, see `CHANGELOG.md`; verified on real weights on an RTX 5090 (2026-09-20, `reports/trinity-prime-20260920`): fp32 ONNX matches torch (argmax agreement 1.000, max probability difference 0.02), the original dynamic int8 did not (agreement 0.70 to 0.87, differences up to 0.91) and was replaced by weight-only int8, see section 7
- `peewee_decide/server.py`: FastAPI + uvicorn, `POST /v1/decide` (single) and `POST /v1/decide/batch`,
  `GET /healthz`, `GET /metrics` (Prometheus: latency, batch size, truncation rate, routing
  counts, per-question confidence histograms).
- **Dynamic batching.** A request queue that flushes on `max_batch` or `max_wait_ms` (start at
  32 and 5 ms) so many agents share one forward pass.
- **Preload by default** in the server; both English and multilingual resident.
- **ONNX export + ONNX Runtime backend** with int8 dynamic quantisation for CPU deployments.
  Benchmark against the torch path and publish the numbers in `BENCHMARKS.md`.
- Dockerfile (CPU and CUDA variants), a `docker compose` example, and a load test script.
- Thin clients: a Python client, and a LiteLLM guardrail/router hook so existing pipelines can
  call it without new code.

### Phase 3: the context problem — measured on 2026-09-20; do not raise the defaults, fine-tune
Status note. `scripts/length_sweep.py` (`reports/trinity-prime-20260920/length_sweep.md`) ran two
experiments per checkpoint with no retraining. The checkpoints are fine-tuned at `max_len` 1,024
(`english` at 512).

*Natural long reviews* (300 labelled IMDB reviews of 1,024 to 3,024 tokens): reading the whole
review at 2,048 or 4,096 is never worse than truncating and gains 0 to 3 points (`english` 0.843
to 0.867, `typed-decisions` 0.857 to 0.887, `multilingual` flat at 0.75). So longer inputs do
not break the encoders.

*Short review after neutral background* (200 short reviews behind AG News text, nothing
truncated): accuracy falls with the distance of the evidence from the start. `english` alone
0.925; at about 900 tokens 0.845; 1,900 tokens 0.700; 3,900 tokens 0.550; 7,900 tokens 0.485,
which is chance. `typed-decisions` 0.930, 0.840, 0.685, 0.540, 0.530; `multilingual` 0.890,
0.695, 0.585, 0.590, 0.530. The control that keeps only the last 1,024 tokens (what the service
does today for list states) holds at 0.86, 0.855, 0.83, 0.81 for `english`. Two effects: neutral
text in the state costs about 8 points even inside the trained length (distraction), and the
model cannot read evidence past the position it was trained to (extrapolation).

Decision. The defaults stay at the checkpoints' `max_len`, because raising them is harmful for
the workloads that matter: when the decisive part of a transcript sits at the end, a longer
window makes the model attend past its trained positions and lose it, while end-preserving
truncation at 1,024 keeps it. The 0 to 3 point gain on natural long reviews is not worth that.
Phase 3 is therefore the long-context fine-tune, and the training mix must contain long states
whose evidence sits deep in the input, not just long documents. Until then: keep the end of
transcripts, send structured state instead of raw context, and chunk-and-aggregate for inputs
that genuinely exceed 1,024 tokens.

The original plan, kept for reference:
- **Test the encoders past their training length.** ModernBERT-large and mmBERT-base both use
  RoPE and were pre-trained to 8,192 tokens; Peewee set `max_len` to 512/1,024 at fine-tuning
  time. Run the held-out suites at `max_len` 1,024, 2,048 and 4,096 with no retraining and
  record the accuracy curve. If it degrades gracefully, raise the defaults for the
  multilingual and typed-decisions checkpoints immediately.
- **Fine-tune at longer length.** Continue training the multilingual checkpoint at 4,096 with
  a mix that includes long states (transcripts, diffs, tool outputs) so the head learns to
  read them.
- **Structured state formats for agent workloads.** Define compact schemas so the model sees
  what matters: for gating, `{goal, last_n_turns, last_tool, last_result_excerpt, steps_taken,
  errors}`; for tool-result verification, `{task, expected, result_head, result_tail,
  exit_code}`; for diffs, `{files, hunks_summary, tests_touched}`. Most of the value of "64k
  context" is recoverable by not sending 64k of noise.
- **Chunk-and-aggregate** fallback for genuinely long inputs: score each chunk, combine with
  a learned or fixed rule (max for risk questions, last-chunk-weighted for state questions).

### Phase 4: per-workload classifiers — training CLI done (`peewee train`, `peewee eval`, `peewee prepare-data`), and mixing datasets beats training on either alone (2026-09-22); teacher labelling and the gating harness are next
- Mixing typed-decisions (upsampled 4x) with Open-Jev gave the best checkpoint on every split: 0.8005 on
  typed-decisions test against the published checkpoint's 0.7685, while matching the Open-Jev specialist on
  Open-Jev test and OOD. Calibration does not mix the same way — one temperature map fitted on a combined
  held-out set follows the larger dataset — so fit temperatures per workload (`--calib-data`, or
  `Agent.fit_temperatures` after training). `BENCHMARKS.md`, "Training on both datasets at once".
- `peewee_decide/train.py`: the RLCD loop lifted out of the notebook into a CLI (`peewee train
  --data ... --base multilingual --epochs 4`), with the dataset schema documented (JSONL of
  `{state, questions, targets}`).
- `peewee_decide/distill.py`: a teacher-labelling script that sends unlabelled states plus the question
  schema to Claude and writes soft targets, with a held-out split for calibration and
  evaluation. This is how each workload gets its data.
- Evaluation harness that reports accuracy, Brier, ECE and the confusion matrix per question,
  and a "coverage at precision" curve so you can pick gating thresholds honestly.
- Order of workloads, by how well they fit today and how easy labels are to get:
  1. Email/SMS spam and phishing (labels exist; inputs fit; multilingual matters).
  2. Model routing (labels come from outcome data you already collect).
  3. Security gate.
  4. Tool routing (after the hierarchical helper).
  5. Agent gating and tool-result verification (after Phase 3).
  6. Claude Code supervisor (after Phase 3, trained on your own repo history).

---

## 6. What was done in this commit

- Seeded `ultra_laya` with the full upstream history (50 commits) so upstream fixes can be
  merged later with `git merge`.
- Added this review. No library code was changed.

---

## 7. Open questions after the first GPU validation (2026-09-20)

Carried over from the first two development sessions; see `reports/trinity-prime-20260920/`.

- **int8 ONNX, resolved.** Dynamic int8 (activations quantised too) broke the GeGLU feed-forward
  blocks: probabilities moved by up to 0.91 and 13 to 30% of argmaxes flipped. Per-channel
  weights, unsigned weights, pre-processing and excluding the head all failed the same way;
  excluding the MLP alone recovered most of it. `--quantize` now does weight-only int8
  (`MatMulNBits`, block 128) on the encoder's linear weights, which tracks fp32 to within 0.06
  at about 40% of the size but is no faster on CPU. A faster CPU path needs activation
  quantisation that survives the MLP outliers (SmoothQuant-style scaling or static calibration
  with per-channel activation ranges), which is open.
- **Fitted temperatures are not wired as defaults.** `calibration/multilingual.json` (T = 2.07
  for the `choice:11+` bucket, fit on 438 MASSIVE examples, held-out ECE 0.37 to 0.13) and
  `calibration/english.json` (T = 2.44) are committed. They only cover the 11+ option bucket,
  so wiring them in `Agent._init_common` is a product decision for the owner.
- **Phase 3 is decided**: fine-tune for long context; do not raise `max_len`. See the Phase 3 status note.
- **The CPU Docker image builds and serves as written** (1.38 GB, `python:3.11-slim`, CPU torch
  plus the server and onnx extras); the CUDA image is still unbuilt. Throughput is in
  `BENCHMARKS.md` under "Decision service throughput": about 40 questions/s on 8 CPU cores
  against about 1,400 on the 5090, so CPU serving is for low-volume or edge use only.
- `act_probability` in answers comes from an "action head" whose meaning is undocumented
  upstream; it is passed through untouched. Decide whether to document or drop it.
- The language guess is a heuristic tuned on a small regression set in `tests/test_lang.py`.
  Quebec French traffic is the motivating case; extend the tests with real samples before
  tuning further, and prefer plugging in a real detector for production.
- `hierarchical_choice` multiplies the two stages' top probabilities as the path confidence;
  it is not calibrated as a whole.
- The service's `/v1/decide/batch` returns per-item errors inline with HTTP 200; a client that
  wants strict semantics should use `/v1/decide`.
- Upstream `NandhaKishorM/laya` has a single maintainer and a large unreviewed PR backlog.
  Work on the fork; mine upstream PRs for ideas (#19 calibration and #27 preload are ported,
  #18 head-budget and #3 server were references). The transformers-floor fix and the router
  tuple fix would be easy PRs to send upstream.
