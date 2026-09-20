# Hand-off: where this fork stands and what to do next

Written at the end of the first development session (sandboxed, no GPU, no Hugging Face hub
access). Read this first in a new session; then `REVIEW.md` for the plan and `CHANGELOG.md`
for the details.

## State of the code

| phase | status | where |
|---|---|---|
| 0 hygiene | done | `CHANGELOG.md` "Phase 0" |
| 1 safe to gate on (truncation reporting, unified confidence, calibration API, language guess, batch predict, patterns) | done except shipping fitted multilingual temperatures | "Phase 1" |
| 2 decision service (dynamic batching, HTTP API, metrics, client, ONNX backend, Docker, load test, LiteLLM example) | done, unverified on real weights | "Phase 2" |
| 3 context length | not started; the measurement is scripted | `scripts/gpu_validate.py` step 5 |
| 4 per-workload fine-tuning (`laya/train.py`, teacher labelling) | not started | `REVIEW.md` §5 |

178 weight-free tests pass. Nothing has run against the real checkpoints since the fork.

## Repository facts

- GitHub: `roadius2/ultra_laya`, public. Branches: `claude/main` (the working branch, currently
  the default) and `main` (created from it, one commit behind). The owner intends `main` to be
  the protected mainline with a ruleset named `protect-main`; confirm the default branch and the
  ruleset exist before opening PRs. Five Dependabot PRs for GitHub Actions bumps are open.
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
