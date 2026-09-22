"""Validate this fork against the real checkpoints on a GPU box, in one run.

    pip install -e ".[dev,server,onnx]" datasets
    python scripts/gpu_validate.py --out reports/$(hostname)-$(date +%Y%m%d)

What it does, per checkpoint (english, multilingual, typed-decisions):
  1. load       real weights; load time, device, VRAM; a few preset predictions in EN / FR / HI
  2. latency    torch: 1 / 5 / 10 / 50 questions per call; predict_many over 32 requests
  3. onnx       export fp32 + int8; parity vs torch on 24 states; ONNX Runtime CPU latency
  4. calibrate  (multilingual, english) fit temperatures on MASSIVE validation (6 locales),
                evaluate on MASSIVE test; writes calibration/<name>.json ready to commit
  5. lengths    (multilingual, typed-decisions) accuracy on IMDB reviews at max_len
                512 / 1024 / 2048 / 4096 and left vs right truncation: does the encoder read
                past its training length?

Everything lands in --out: report.json, report.md, calibration/*.json, onnx/<name>/. Commit
report.md and calibration/*.json; the ONNX exports are large and stay local. Any step that
fails is recorded in the report and the run continues.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

import peewee_decide  # noqa: E402
from peewee_decide.calibrate import collect_records, fit_temperature_map  # noqa: E402

CHECKPOINTS = {
    "english": ("convaiinnovations/laya", None),
    "multilingual": ("convaiinnovations/laya", "multilingual"),
    "typed-decisions": ("convaiinnovations/laya", "typed-decisions"),
}
SAMPLE_STATES = [
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
    {"body": "The API has been returning 502s since 9am and our checkout is down."},
    {"body": "Je veux annuler mon forfait, votre service ne marche pas depuis hier."},
    {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"},
    {"body": "Can I get pricing for 50 seats on the enterprise tier?"},
    {"body": "Ich kann mich seit gestern nicht einloggen, bitte helfen Sie mir."},
]
MASSIVE_LOCALES = ["en-US", "fr-FR", "de-DE", "es-ES", "hi-IN", "ja-JP"]


def _sync(agent):
    try:
        import torch
        if agent.device.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def timed(fn, repeats=10, warmup=2, sync=None):
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        xs.append(time.perf_counter() - t0)
    return {"p50_ms": round(statistics.median(xs) * 1000, 2), "mean_ms": round(statistics.mean(xs) * 1000, 2),
            "min_ms": round(min(xs) * 1000, 2), "n": repeats}


def step(report: Dict[str, Any], name: str, fn):
    print("\n== %s" % name, flush=True)
    t0 = time.perf_counter()
    try:
        out = fn()
        report[name] = {"ok": True, "seconds": round(time.perf_counter() - t0, 1), "result": out}
        print("   ok (%.1fs)" % (time.perf_counter() - t0), flush=True)
    except Exception as e:  # keep going: a partial report beats none
        report[name] = {"ok": False, "seconds": round(time.perf_counter() - t0, 1),
                        "error": "%s: %s" % (type(e).__name__, e), "traceback": traceback.format_exc()}
        print("   FAILED: %s: %s" % (type(e).__name__, e), flush=True)


# ---------------------------------------------------------------------------- 1. load
def do_load(name, device):
    repo, sub = CHECKPOINTS[name]
    t0 = time.perf_counter()
    agent = peewee_decide.load(repo, subfolder=sub, device=device)
    load_s = time.perf_counter() - t0
    out = {"load_seconds": round(load_s, 1), "device": str(agent.device), "dtype": str(agent.dtype),
           "fell_back_to_cpu": agent.fell_back_to_cpu, "cfg": {k: agent.cfg.get(k) for k in
           ("encoder", "max_len", "head_max_len", "head_layers", "temperature", "temperature_by_options")},
           "n_params": sum(p.numel() for p in agent.model.parameters())}
    try:
        import torch
        if agent.device.type == "cuda":
            out["vram_mb"] = round(torch.cuda.memory_allocated() / 2**20)
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    qs = peewee_decide.triage_questions()
    out["samples"] = []
    for st in SAMPLE_STATES:
        r = agent.predict(st, qs)
        out["samples"].append({"state": st["body"][:60], "intent": r["answers"]["intent"]["choice"],
                               "intent_conf": r["answers"]["intent"]["confidence"],
                               "frustration": r["answers"]["frustration"]["score"],
                               "churn": r["answers"]["churn_risk"]["noul"], "usage": r["usage"]})
    return agent, out


# ---------------------------------------------------------------------------- 2. latency
def do_latency(agent):
    st = SAMPLE_STATES[0]
    base = peewee_decide.triage_questions()
    out = {}
    for n in (1, 5, 10, 50):
        qs = {}
        i = 0
        while len(qs) < n:
            for k, v in base.items():
                if len(qs) < n:
                    qs["%s_%d" % (k, i)] = v
            i += 1
        out["questions_%d" % n] = timed(lambda: agent.predict(st, qs), sync=lambda: _sync(agent))
    reqs = [(SAMPLE_STATES[i % len(SAMPLE_STATES)], base) for i in range(32)]
    out["predict_many_32x5"] = timed(lambda: agent.predict_many(reqs, max_batch=64), repeats=5, sync=lambda: _sync(agent))
    out["predict_many_32x5"]["per_question_ms"] = round(out["predict_many_32x5"]["p50_ms"] / 160, 3)
    return out


# ---------------------------------------------------------------------------- 3. onnx
def do_onnx(agent, name, out_dir):
    from peewee_decide.onnx_backend import OnnxAgent, export_onnx
    d = os.path.join(out_dir, "onnx", name)
    meta = export_onnx(agent, d, quantize=True)
    res = {"export": meta}
    qs = peewee_decide.triage_questions()
    states = SAMPLE_STATES * 4
    ref = [agent.predict(s, qs) for s in states]
    for variant, prefer in (("fp32", False), ("int8", True)):
        o = OnnxAgent(d, prefer_quantized=prefer, providers=["CPUExecutionProvider"])
        diffs, agree = [], 0
        for s, r in zip(states, ref):
            g = o.predict(s, qs)
            for qid in qs:
                ra, ga = r["answers"][qid], g["answers"][qid]
                if "probabilities" in ra:
                    diffs.extend(abs(ra["probabilities"][k] - ga["probabilities"][k]) for k in ra["probabilities"])
                    agree += int(ra.get("choice", ra.get("level")) == ga.get("choice", ga.get("level")))
                else:
                    diffs.append(abs(ra["noul"] - ga["noul"]))
                    agree += int((ra["noul"] >= 0.5) == (ga["noul"] >= 0.5))
        res[variant] = {"model": os.path.basename(o.onnx_path), "providers": o.providers,
                        "max_abs_prob_diff": round(max(diffs), 4), "mean_abs_prob_diff": round(float(np.mean(diffs)), 5),
                        "argmax_agreement": round(agree / (len(states) * len(qs)), 4),
                        "latency_1q": timed(lambda: o.predict(states[0], {"i": qs["intent"]}), repeats=10),
                        "latency_5q": timed(lambda: o.predict(states[0], qs), repeats=10)}
    return res


# ---------------------------------------------------------------------------- 4. calibrate
def massive_examples(locales, split, per_locale, seed=0):
    from datasets import load_dataset
    # `datasets` >= 4 no longer runs the repo's loading script; the hub's parquet conversion
    # holds every locale in one config with a `locale` column and the same intent labels.
    full = load_dataset("AmazonScience/massive", "default", split=split, revision="refs/convert/parquet")
    names = full.features["intent"].names
    examples = []
    labels_by_locale = {}
    for loc in locales:
        ds = full.filter(lambda row, loc=loc: row["locale"] == loc)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds), size=min(per_locale, len(ds)), replace=False)
        q = {"intent": {"type": "choice", "instructions": "What does the user want in `utt`?",
                        "criteria": {n: None for n in names[:20]}}}
        for i in idx:
            row = ds[int(i)]
            if row["intent"] >= 20:
                continue          # keep 20 options, as in the upstream benchmark
            examples.append(({"utt": row["utt"]}, q, {"intent": names[row["intent"]]}, loc))
        labels_by_locale[loc] = len([e for e in examples if e[3] == loc])
    return examples, labels_by_locale


def do_calibrate(agent, name, out_dir, per_locale):
    fit_ex, counts = massive_examples(MASSIVE_LOCALES, "validation", per_locale)
    test_ex, _ = massive_examples(MASSIVE_LOCALES, "test", per_locale, seed=1)
    recs = collect_records(agent, [(s, q, lab) for s, q, lab, _ in fit_ex], batch_size=32)
    fit = fit_temperature_map(recs)
    test_recs = collect_records(agent, [(s, q, lab) for s, q, lab, _ in test_ex], batch_size=32)
    from peewee_decide.calibrate import calibration_report
    held_out = calibration_report(test_recs, fit["temperature"], fit["temperature_by_options"])
    agent.fit_temperatures(recs)
    os.makedirs(os.path.join(out_dir, "calibration"), exist_ok=True)
    path = os.path.join(out_dir, "calibration", "%s.json" % name)
    agent.save_calibration(path, meta={"fit_on": "AmazonScience/massive validation", "locales": MASSIVE_LOCALES,
                                       "n_fit": len(recs), "n_test": len(test_recs)})
    # per-locale accuracy on the held-out set, after calibration
    by_loc: Dict[str, list] = {}
    for (_s, _q, _lab, loc), rec in zip(test_ex, test_recs):
        _qt, z, t, _k = rec
        by_loc.setdefault(loc, []).append(int(np.argmax(z) == np.argmax(t)))
    return {"fit": {k: v for k, v in fit.items() if k != "report"}, "fit_report": fit["report"]["all"],
            "held_out": held_out["all"], "per_locale_accuracy": {k: round(float(np.mean(v)), 3) for k, v in by_loc.items()},
            "n_by_locale": counts, "calibration_file": path}


# ---------------------------------------------------------------------------- 5. lengths
def do_lengths(agent, n_samples, lengths):
    from datasets import load_dataset
    # the bare "imdb" alias is gone from the hub
    ds = load_dataset("stanfordnlp/imdb", split="test").shuffle(seed=0).select(range(n_samples))
    q = {"sentiment": {"type": "choice", "instructions": "Is the movie review in `review` positive or negative?",
                       "criteria": {"negative": None, "positive": None}}}
    label = ["negative", "positive"]
    trained = int(agent.cfg.get("max_len", 512))
    out = {"trained_max_len": trained, "n": n_samples, "settings": {}}
    tok_lens = [len(agent.tok(r["text"], add_special_tokens=False)["input_ids"]) for r in ds]
    out["review_tokens"] = {"p50": int(np.median(tok_lens)), "p90": int(np.percentile(tok_lens, 90)), "max": int(max(tok_lens))}
    orig = dict(agent.cfg)
    try:
        for L in lengths:
            for side in ("right", "left"):
                agent.cfg["max_len"] = L
                correct, conf, truncated = [], [], 0
                for r in ds:
                    res = agent.predict({"review": r["text"]}, q, truncate=side)
                    a = res["answers"]["sentiment"]
                    correct.append(int(a["choice"] == label[r["label"]]))
                    conf.append(a["confidence"])
                    truncated += int(res["usage"]["truncated"])
                acc = float(np.mean(correct))
                out["settings"]["max_len=%d,%s" % (L, side)] = {
                    "accuracy": round(acc, 4), "mean_confidence": round(float(np.mean(conf)), 4),
                    "ece": round(peewee_decide.ece_score(np.asarray(conf), np.asarray(correct, dtype=float)), 4),
                    "truncated_fraction": round(truncated / n_samples, 3)}
                print("   max_len=%d %-5s acc=%.3f truncated=%.0f%%" % (L, side, acc, 100 * truncated / n_samples), flush=True)
    finally:
        agent.cfg.clear()
        agent.cfg.update(orig)
    return out


# ---------------------------------------------------------------------------- report
def to_markdown(rep: Dict[str, Any]) -> str:
    L = ["# GPU validation report", "", "host: `%s`  python: %s  torch: %s  peewee: %s  date: %s" % (
        rep["env"]["host"], rep["env"]["python"], rep["env"]["torch"], rep["env"]["peewee"], rep["env"]["date"]), ""]
    for name, r in rep["checkpoints"].items():
        L += ["## %s" % name, ""]
        for stepname, sr in r.items():
            if not sr["ok"]:
                L += ["- **%s: FAILED** `%s`" % (stepname, sr["error"]), ""]
                continue
            res = sr["result"]
            if stepname == "load":
                L += ["- load: %.1fs on %s (%s), %s params%s" % (
                    res["load_seconds"], res["device"], res["dtype"], "{:,}".format(res["n_params"]),
                    (", %d MB VRAM, %s" % (res["vram_mb"], res.get("gpu"))) if "vram_mb" in res else "")]
                L += ["", "| state | intent | conf | frustration | churn | truncated |", "|---|---|---|---|---|---|"]
                for s in res["samples"]:
                    L.append("| %s | %s | %.2f | %.2f | %.2f | %s |" % (s["state"], s["intent"], s["intent_conf"],
                                                                     s["frustration"], s["churn"], s["usage"]["truncated"]))
                L.append("")
            elif stepname == "latency":
                L += ["| questions per call | p50 ms |", "|---|---|"]
                for k, v in res.items():
                    L.append("| %s | %.1f%s |" % (k, v["p50_ms"], (" (%.2f/question)" % v["per_question_ms"]) if "per_question_ms" in v else ""))
                L.append("")
            elif stepname == "onnx":
                L += ["| onnx | max abs prob diff | argmax agreement | 1q p50 ms (CPU) | 5q p50 ms (CPU) |", "|---|---|---|---|---|"]
                for v in ("fp32", "int8"):
                    x = res[v]
                    L.append("| %s | %.4f | %.3f | %.1f | %.1f |" % (v, x["max_abs_prob_diff"], x["argmax_agreement"],
                                                                x["latency_1q"]["p50_ms"], x["latency_5q"]["p50_ms"]))
                L.append("")
            elif stepname == "calibrate":
                h = res["held_out"]
                L += ["- calibration fit on %s examples; held-out ECE %.3f -> %.3f, NLL %.3f -> %.3f, accuracy %.3f" % (
                    res["fit"]["n_by_bucket"], h["before"]["ece"], h["after"]["ece"], h["before"]["nll"], h["after"]["nll"], h["after"]["accuracy"]),
                      "- per-locale accuracy: %s" % json.dumps(res["per_locale_accuracy"]),
                      "- temperatures: %s / %s" % (res["fit"]["temperature"], res["fit"]["temperature_by_options"]), ""]
            elif stepname == "lengths":
                L += ["- IMDB review tokens p50 %d, p90 %d, max %d; trained max_len %d" % (
                    res["review_tokens"]["p50"], res["review_tokens"]["p90"], res["review_tokens"]["max"], res["trained_max_len"]),
                      "", "| setting | accuracy | ECE | mean conf | truncated |", "|---|---|---|---|---|"]
                for k, v in res["settings"].items():
                    L.append("| %s | %.3f | %.3f | %.3f | %.0f%% |" % (k, v["accuracy"], v["ece"], v["mean_confidence"], 100 * v["truncated_fraction"]))
                L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/gpu-validation")
    ap.add_argument("--models", default="english,multilingual,typed-decisions")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip", default="", help="comma-separated steps to skip: latency,onnx,calibrate,lengths")
    ap.add_argument("--per-locale", type=int, default=300, help="MASSIVE examples per locale for calibration")
    ap.add_argument("--length-samples", type=int, default=300, help="IMDB reviews for the length sweep")
    ap.add_argument("--lengths", default="512,1024,2048,4096")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    os.makedirs(args.out, exist_ok=True)
    import torch
    rep: Dict[str, Any] = {"env": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                                   "peewee": peewee_decide.__version__, "cuda": torch.cuda.is_available(),
                                   "date": time.strftime("%Y-%m-%d %H:%M")}, "args": vars(args), "checkpoints": {}}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        r: Dict[str, Any] = {}
        rep["checkpoints"][name] = r
        holder: Dict[str, Any] = {}

        def _load():
            agent, out = do_load(name, args.device)
            holder["agent"] = agent
            return out
        step(r, "load", _load)
        agent = holder.get("agent")
        if agent is None:
            continue
        if "latency" not in skip:
            step(r, "latency", lambda: do_latency(agent))
        if "onnx" not in skip:
            step(r, "onnx", lambda: do_onnx(agent, name, args.out))
        if "calibrate" not in skip and name in ("english", "multilingual"):
            step(r, "calibrate", lambda: do_calibrate(agent, name, args.out, args.per_locale))
        if "lengths" not in skip and name in ("multilingual", "typed-decisions"):
            step(r, "lengths", lambda: do_lengths(agent, args.length_samples, [int(x) for x in args.lengths.split(",")]))
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(rep, f, indent=2, default=str)
        holder.clear()
        agent = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(to_markdown(rep))
    print("\nwrote %s/report.md and report.json" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
