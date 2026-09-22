"""`peewee eval`: score a checkpoint on JSONL cases (see `peewee_decide.data`).

Accuracy is against each question's reference answer (the hard label when the case has one,
else the target argmax; noul counts p(true) >= 0.5 as true), which is how the upstream notebook
scored typed-decisions. Soft accuracy, Brier score and total variation compare the reported
distribution with the target distribution; ECE uses each answer's `confidence`.
"""
import argparse
import json
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .agent import Agent
from .common import ece_score
from .data import option_keys, read_jsonl, reference_index, target_vector, validate_record

QTYPE_ORDER = ("choice", "score", "noul")


def answer_probabilities(qdef: Dict[str, Any], answer: Dict[str, Any]) -> np.ndarray:
    """The reported distribution in option order."""
    if qdef["type"] == "noul":
        p1 = float(answer["noul"])
        return np.array([1.0 - p1, p1])
    if qdef["type"] == "choice":
        keys = list(Agent._to_internal(qdef)["crit"])        # the runtime keys choices as given, not as strings
    else:
        keys = option_keys(qdef)
    return np.array([float(answer["probabilities"][k]) for k in keys])


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    conf = np.array([r["confidence"] for r in rows], dtype=float)
    correct = np.array([r["correct"] for r in rows], dtype=float)
    out: Dict[str, Any] = {
        "n": len(rows), "accuracy": float(correct.mean()), "soft_accuracy": float(np.mean([r["soft"] for r in rows])),
        "brier": float(np.mean([r["brier"] for r in rows])), "total_variation": float(np.mean([r["tv"] for r in rows])),
        "ece": ece_score(conf, correct), "mean_confidence": float(conf.mean())}
    scored = [r for r in rows if "score_mae" in r]
    if scored:
        out["score_mae"] = float(np.mean([r["score_mae"] for r in scored]))
        out["score_within_one"] = float(np.mean([r["within_one"] for r in scored]))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def _percentiles(ms: List[float]) -> Dict[str, float]:
    return {"p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2)}


def evaluate(agent, records: Sequence[Dict[str, Any]], truncate: Optional[str] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []
    per_question: List[float] = []
    for rec in records:
        validate_record(rec)
        t0 = time.perf_counter()
        res = agent.predict(rec["state"], rec["questions"], truncate=truncate)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        per_question.append(latencies[-1] / len(rec["questions"]))
        workflow = (rec.get("meta") or {}).get("workflow")
        for qid, qdef in rec["questions"].items():
            tgt = rec["targets"][qid]
            answer = res["answers"][qid]
            p = answer_probabilities(qdef, answer)
            g = np.asarray(target_vector(qdef, tgt))
            pred = int(p[1] >= 0.5) if qdef["type"] == "noul" else int(p.argmax())
            row = {"type": qdef["type"], "workflow": workflow, "correct": float(pred == reference_index(qdef, tgt)),
                   "soft": float(p @ g), "brier": float(((p - g) ** 2).sum()), "tv": float(0.5 * np.abs(p - g).sum()),
                   "confidence": float(answer["confidence"])}
            if qdef["type"] == "score":
                levels = np.arange(len(p))
                err = abs(float(p @ levels) - float(g @ levels))
                row["score_mae"], row["within_one"] = err, float(err <= 1.0)
            rows.append(row)
    if not rows:
        raise ValueError("no questions to evaluate")
    report: Dict[str, Any] = {
        "n_cases": len(records), "n_questions": len(rows), "overall": _summarise(rows),
        "by_type": {t: _summarise([r for r in rows if r["type"] == t]) for t in QTYPE_ORDER
                    if any(r["type"] == t for r in rows)},
        "latency_ms": _percentiles(latencies), "latency_ms_per_question": _percentiles(per_question)}
    flows = sorted({r["workflow"] for r in rows if r["workflow"]})
    if flows:
        report["by_workflow"] = {w: _summarise([r for r in rows if r["workflow"] == w]) for w in flows}
    return report


def load_model_for_eval(spec: str, device: Optional[str] = None):
    """A checkpoint directory, a hub repo id, or a checkpoint name (english, multilingual, ...)."""
    from .train import resolve_base
    model_dir, _cfg = resolve_base(spec)
    return Agent(model_dir, device=device)


def format_report(rep: Dict[str, Any]) -> str:
    lines = ["| questions | n | accuracy | soft acc | Brier | ECE |", "|---|---|---|---|---|---|"]
    for name, s in [("all", rep["overall"])] + list(rep["by_type"].items()):
        lines.append("| %s | %d | %.4f | %.4f | %.4f | %.4f |" % (name, s["n"], s["accuracy"], s["soft_accuracy"],
                                                                s["brier"], s["ece"]))
    lines.append("latency per case: p50 %.1f ms, p95 %.1f ms" % (rep["latency_ms"]["p50"], rep["latency_ms"]["p95"]))
    lines.append("latency per question: p50 %.1f ms, p95 %.1f ms" % (rep["latency_ms_per_question"]["p50"],
                                                                     rep["latency_ms_per_question"]["p95"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="peewee eval", description="Score a checkpoint on JSONL cases.")
    ap.add_argument("model", help="checkpoint directory or name (english, multilingual, typed-decisions)")
    ap.add_argument("--data", required=True, help="cases, JSONL (schema in peewee_decide/data.py)")
    ap.add_argument("--device", help="cuda, cpu or mps (default: best available)")
    ap.add_argument("--truncate", choices=["left", "right"], help="state truncation side (default: runtime default)")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args(argv)
    rep = evaluate(load_model_for_eval(args.model, args.device), read_jsonl(args.data), truncate=args.truncate)
    rep["model"], rep["data"] = args.model, args.data
    print(format_report(rep))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
    return 0
