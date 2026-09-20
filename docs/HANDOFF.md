# Hand-off: where this fork stands and what to do next

Written at the end of the first development session (sandboxed, no GPU, no Hugging Face hub
access). Read this first in a new session; then `REVIEW.md` for the plan and `CHANGELOG.md`
for the details.

## State of the code

| phase | status | where |
|---|---|---|
| 0 hygiene | done | `CHANGELOG.md` "Phase 0" |
| 1 safe to gate on (truncation reporting, unified confidence, calibration API, language guess, batch predict, patterns) | done; fitted temperatures are in `calibration/` but not yet wired as defaults | "Phase 1" |
| 2 decision service (dynamic batching, HTTP API, metrics, client, ONNX backend, Docker, load test, LiteLLM example) | done, verified on real weights (fp32 ONNX good, int8 broken) | "Phase 2", `reports/trinity-prime-20260920` |
| 3 context length | measured once, inconclusive: IMDB is too short to test past the trained 1,024 | `REVIEW.md` Phase 3 status note |
| 4 per-workload fine-tuning (`laya/train.py`, teacher labelling) | not started | `REVIEW.md` §5 |

180 weight-free tests pass. The full validation ran against the real checkpoints on an RTX
5090 on 2026-09-20 (`reports/trinity-prime-20260920/report.md`); the first run surfaced and
fixed three bugs (ONNX export left the model on CPU; `datasets` 5 dropped script datasets;
the `imdb` alias is gone), all in `CHANGELOG.md`.

## Repository facts

- GitHub: `roadius2/ultra_laya`, public. Branches: `claude/main` (the working branch, currently
  the default) and `main` (created from it, now several commits behind). The owner intends `main` to be
  the protected mainline with a ruleset named `protect-main`. On 2026-09-20 the ruleset existed
  but its branch include list was empty, so it protected nothing; CI only runs on pushes to
  `main` and on PRs. Fix both before opening PRs. Five Dependabot PRs for GitHub Actions bumps are open.
- Upstream `NandhaKishorM/laya` has a single maintainer and a large unreviewed PR backlog.
  Work on the fork; mine upstream PRs for ideas (#19 calibration and #27 preload are already
  ported, #18 head-budget and #3 server were references). The transformers-floor fix and the
  router tuple fix would be easy PRs to send upstream.
- Commit identity (set per clone):
  ```
  git config user.name roadius2
  git config user.email roadius2@users.noreply.github.com
  ```
  and end commit messages with `Co-Authored-By: Claude <noreply@anthropic.com>`.

## First things to do on a GPU box

1. `scripts/bootstrap.sh` clones into `~/dev_projects/ultra_laya`, sets the identity, makes a
   venv and installs extras. Or follow `docs/GPU_VALIDATION.md` by hand.
2. `python -m pytest` (must be green), then `python scripts/gpu_validate.py --out reports/$(hostname)`.
3. Commit `reports/<host>/report.md` and copy `reports/<host>/calibration/*.json` to
   `calibration/`. If the multilingual temperatures look sane, wire them as the default
   calibration for that checkpoint (`Agent._init_common` is the place) and note it in the changelog.
4. Read the length-sweep table. If accuracy holds past the trained `max_len`, raise the
   defaults for `multilingual` and `typed-decisions` in `Router` / docs and record the numbers
   in `REVIEW.md` Phase 3. If it collapses, Phase 3 is the long-context fine-tune.
5. If ONNX parity is good on ModernBERT and mmBERT, rebuild the CPU Docker image with
   `LAYA_BACKEND=onnx` and run `scripts/loadtest.py` for real throughput numbers.

## Known gaps and judgement calls

- **int8 ONNX is unusable as exported.** Dynamic weight-only quantisation of the whole graph
  moves probabilities by up to 0.91 and flips 13 to 30% of argmaxes on all three checkpoints,
  while fp32 ONNX matches torch exactly. Do not ship `--quantize` output; try excluding the
  decision head and embeddings from quantisation, or static quantisation with calibration data.
- **Fitted temperatures are not wired as defaults.** `calibration/multilingual.json` (T = 2.07
  for the `choice:11+` bucket, fit on 438 MASSIVE examples, held-out ECE 0.37 to 0.13) and
  `calibration/english.json` (T = 2.44) are committed. They only cover the 11+ option bucket,
  so wiring them in `Agent._init_common` is a product decision left to the owner.
- **The GPU box is shared.** `trinity-prime` runs an 18 GB embedding server and a TTS service
  that has been crash-looping on CUDA out-of-memory every 15 s for two weeks; its 10 to 13 GB
  grab collides with any run longer than a few seconds. Expect the odd OOM and rerun the step.
  The 5090 needs the cu128 torch wheels (`TORCH_INDEX=https://download.pytorch.org/whl/cu128`).
- `act_probability` in answers comes from an "action head" whose meaning is undocumented
  upstream; it is passed through untouched. Decide whether to document or drop it.
- The language guess is a heuristic tuned on a small regression set in `tests/test_lang.py`.
  Quebec French traffic is the motivating case; extend the tests with real samples before
  tuning further, and prefer plugging in a real detector for production.
- `hierarchical_choice` multiplies the two stages' top probabilities as the path confidence;
  it is not calibrated as a whole.
- The service's `/v1/decide/batch` returns per-item errors inline with HTTP 200; a client that
  wants strict semantics should use `/v1/decide`.
- Docker images and the CUDA base tag were written without being built here; expect a first
  build to need a tag or wheel-index adjustment.
