"""Phase 3 measurement: do the checkpoints keep working past the `max_len` they were fine-tuned at?

Two experiments per checkpoint, no retraining, run on a GPU box:

1. natural: IMDB reviews of at least `--min-tokens` tokens (874 labelled reviews reach 1,024; nine
   reach 2,048), answered at each `--lengths` setting with both truncation sides. Shows what the
   extra context buys on real long inputs, bucketed by review length.
2. padded: short IMDB reviews placed after neutral news text (AG News) so the state is about
   `--pad-targets` tokens long, answered with `max_len` large enough that nothing is cut. The label
   only depends on `review`, so this isolates "can the model still read position 4,000?" from
   "is the extra text useful?". A left-truncated run at the trained length is the control: it
   sees the review plus whatever background fits, exactly what the service does today.

    python scripts/length_sweep.py --out reports/$(hostname)-$(date +%Y%m%d)

Writes `<out>/length_sweep.md` and `length_sweep.json`.
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from typing import Any, Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

QUESTION = {"sentiment": {"type": "choice", "instructions": "Is the movie review in `review` positive or negative?",
                          "criteria": {"negative": None, "positive": None}}}
LABELS = ["negative", "positive"]


# ---------------------------------------------------------------------------- helpers (tested)
def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def select_long(rows: Sequence[Dict[str, Any]], tok, min_tokens: int, n: int, seed: int = 0) -> List[Dict[str, Any]]:
    """Up to `n` rows with at least `min_tokens` tokens, shuffled deterministically, with `tokens` added."""
    keep = []
    for r in rows:
        t = token_len(tok, r["text"])
        if t >= min_tokens:
            keep.append({"text": r["text"], "label": r["label"], "tokens": t})
    random.Random(seed).shuffle(keep)
    return keep[:n]


def pad_state(review: str, fillers: Sequence[str], tok, target_tokens: int, seed: int = 0):
    """`{"background": ..., "review": review}` whose token count approaches `target_tokens` from below,
    built from whole filler passages in a seeded order. Returns the state and its token count."""
    rng = random.Random(seed)
    order = list(range(len(fillers)))
    rng.shuffle(order)
    review_tokens = token_len(tok, review)
    budget = target_tokens - review_tokens
    parts, used = [], 0
    for i in order:
        t = token_len(tok, fillers[i])
        if used + t > budget:
            continue
        parts.append(fillers[i])
        used += t
        if budget - used < 8:
            break
    return {"background": " ".join(parts), "review": review}, used + review_tokens


# ---------------------------------------------------------------------------- experiments
def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def _score(agent, states, labels, max_len, side):
    orig = dict(agent.cfg)
    agent.cfg["max_len"] = max_len
    correct, conf, truncated, tokens = [], [], 0, []
    try:
        for st, lab in zip(states, labels):
            res = agent.predict(st, QUESTION, truncate=side)
            a = res["answers"]["sentiment"]
            correct.append(int(a["choice"] == LABELS[lab]))
            conf.append(a["confidence"])
            truncated += int(res["usage"]["truncated"])
            tokens.append(res["usage"]["input_tokens"])
    finally:
        agent.cfg.clear()
        agent.cfg.update(orig)
    return {"n": len(states), "accuracy": round(float(np.mean(correct)), 4), "mean_confidence": round(float(np.mean(conf)), 4),
            "ece": round(ece(conf, correct), 4), "truncated_fraction": round(truncated / max(1, len(states)), 3),
            "input_tokens_p50": int(np.median(tokens)), "correct": correct}


def natural_sweep(agent, rows, lengths, buckets=(1024, 1536, 2048)):
    states = [{"review": r["text"]} for r in rows]
    labels = [r["label"] for r in rows]
    toks = np.array([r["tokens"] for r in rows])
    out = {"n": len(rows), "review_tokens": {"min": int(toks.min()), "p50": int(np.median(toks)), "max": int(toks.max())},
           "settings": {}}
    for L in lengths:
        for side in ("right", "left"):
            r = _score(agent, states, labels, L, side)
            correct = np.array(r.pop("correct"))
            by = {}
            for lo, hi in zip(buckets, list(buckets[1:]) + [10**9]):
                m = (toks >= lo) & (toks < hi)
                if m.any():
                    by["%d-%s" % (lo, hi if hi < 10**9 else "")] = {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4)}
            r["by_review_length"] = by
            out["settings"]["max_len=%d,%s" % (L, side)] = r
            print("   natural max_len=%-5d %-5s acc=%.3f truncated=%3.0f%%  %s" % (
                L, side, r["accuracy"], 100 * r["truncated_fraction"],
                " ".join("%s:%.3f" % (k, v["accuracy"]) for k, v in by.items())), flush=True)
    return out


def padded_sweep(agent, rows, fillers, targets, trained_len, big_len):
    labels = [r["label"] for r in rows]
    base = _score(agent, [{"review": r["text"]} for r in rows], labels, big_len, "right")
    base.pop("correct")
    out = {"n": len(rows), "review_alone": base, "targets": {}}
    print("   padded  review alone            acc=%.3f" % base["accuracy"], flush=True)
    for T in targets:
        states, ntok = [], []
        for i, r in enumerate(rows):
            st, t = pad_state(r["text"], fillers, agent.tok, T, seed=i)
            states.append(st)
            ntok.append(t)
        full = _score(agent, states, labels, big_len, "right")
        full.pop("correct")
        ctrl = _score(agent, states, labels, trained_len, "left")
        ctrl.pop("correct")
        out["targets"][str(T)] = {"state_tokens_p50": int(np.median(ntok)),
                                  "full_context": full, "left_truncated_at_trained_len": ctrl}
        print("   padded  ~%-5d tokens: full-context acc=%.3f (trunc %.0f%%)  left-cut@%d acc=%.3f" % (
            T, full["accuracy"], 100 * full["truncated_fraction"], trained_len, ctrl["accuracy"]), flush=True)
    return out


# ---------------------------------------------------------------------------- report
def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# Length sweep (Phase 3)", "", "host: `%s`  date: %s  args: `%s`" % (rep["env"]["host"], rep["env"]["date"],
                                                                                  json.dumps(rep["args"])), ""]
    for name, r in rep["checkpoints"].items():
        L += ["## %s (trained max_len %d)" % (name, r["trained_max_len"]), ""]
        nat = r["natural"]
        L += ["### Natural long reviews (n=%d, tokens %d to %d, p50 %d)" % (
              nat["n"], nat["review_tokens"]["min"], nat["review_tokens"]["max"], nat["review_tokens"]["p50"]), "",
              "| setting | accuracy | ECE | mean conf | truncated | by review length |", "|---|---|---|---|---|---|"]
        for k, v in nat["settings"].items():
            L.append("| %s | %.3f | %.3f | %.3f | %.0f%% | %s |" % (k, v["accuracy"], v["ece"], v["mean_confidence"],
                     100 * v["truncated_fraction"], ", ".join("%s: %.3f (n=%d)" % (b, x["accuracy"], x["n"]) for b, x in v["by_review_length"].items())))
        pad = r["padded"]
        L += ["", "### Short review after neutral background (n=%d; review alone: %.3f)" % (pad["n"], pad["review_alone"]["accuracy"]), "",
              "| state tokens (p50) | full context, no truncation | ECE | left-truncated at trained length | ECE |", "|---|---|---|---|---|"]
        for T, v in pad["targets"].items():
            f, c = v["full_context"], v["left_truncated_at_trained_len"]
            L.append("| %d | %.3f (trunc %.0f%%) | %.3f | %.3f | %.3f |" % (v["state_tokens_p50"], f["accuracy"],
                     100 * f["truncated_fraction"], f["ece"], c["accuracy"], c["ece"]))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/length-sweep")
    ap.add_argument("--models", default="english,multilingual,typed-decisions")
    ap.add_argument("--device", default=None)
    ap.add_argument("--min-tokens", type=int, default=1024, help="natural experiment: reviews at least this long")
    ap.add_argument("--natural-samples", type=int, default=300)
    ap.add_argument("--padded-samples", type=int, default=200)
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    ap.add_argument("--pad-targets", default="900,1900,3900,7900")
    ap.add_argument("--big-len", type=int, default=8192, help="max_len used for the no-truncation padded runs")
    args = ap.parse_args()

    import laya
    import torch
    from datasets import concatenate_datasets, load_dataset
    from gpu_validate import CHECKPOINTS

    os.makedirs(args.out, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    targets = [int(x) for x in args.pad_targets.split(",")]
    imdb = concatenate_datasets([load_dataset("stanfordnlp/imdb", split=s) for s in ("train", "test")])
    fillers = list(load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=0).select(range(4000))["text"])
    short = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=1)
    rep: Dict[str, Any] = {"env": {"host": platform.node(), "torch": torch.__version__, "laya": laya.__version__,
                                   "date": time.strftime("%Y-%m-%d %H:%M")}, "args": vars(args), "checkpoints": {}}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        repo, sub = CHECKPOINTS[name]
        print("\n== %s" % name, flush=True)
        agent = laya.load(repo, subfolder=sub, device=args.device)
        trained = int(agent.cfg.get("max_len", 512))
        long_rows = select_long(imdb, agent.tok, args.min_tokens, args.natural_samples, seed=0)
        short_rows = [r for r in short.select(range(2000)) if token_len(agent.tok, r["text"]) <= 300][:args.padded_samples]
        rep["checkpoints"][name] = {
            "trained_max_len": trained,
            "natural": natural_sweep(agent, long_rows, lengths),
            "padded": padded_sweep(agent, short_rows, fillers, targets, trained, args.big_len),
        }
        with open(os.path.join(args.out, "length_sweep.json"), "w") as f:
            json.dump(rep, f, indent=2, default=str)
        del agent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, "length_sweep.md"), "w") as f:
        f.write(to_markdown(rep))
    print("\nwrote %s/length_sweep.md and length_sweep.json" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
