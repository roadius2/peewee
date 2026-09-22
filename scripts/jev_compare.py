"""Benchmark Peewee against TypeSafe's Jev on the same cases, scored the same way.

Three steps, each resumable, so the API calls and the GPU run can happen on different hosts:

    # Jev over the HTTP API (needs TYPESAFE_API_KEY; the first --sequential cases are sent one at a
    # time for clean latency, the rest --concurrency at a time)
    python scripts/jev_compare.py jev --data data/typed-decisions/test.jsonl --out-dir runs/jev

    # Peewee on a checkpoint, one case at a time, as `peewee eval` runs it
    python scripts/jev_compare.py peewee runs/mix-v1 --data data/typed-decisions/test.jsonl \\
        --out-dir runs/peewee-mix-v1 --device cuda

    # score both with peewee_decide.evaluate, plus agreement, latency and tokens
    python scripts/jev_compare.py report --data data/typed-decisions/test.jsonl \\
        --peewee runs/peewee-mix-v1/typed-decisions-test.jsonl --jev runs/jev/typed-decisions-test.jsonl \\
        --out reports/jev-vs-peewee/typed-decisions-test.json

Jev's outputs are a report card only: they are never training, calibration or data-selection
input (TypeSafe's customer agreement forbids using them to develop a competing model). Keep the
caches under `runs/`, away from `data/`.

Jev reports `confidence` as a concentration measure; Peewee's is the probability of the answer
it gives. So that ECE means the same thing for both, `from_jev` recomputes confidence Peewee's
way (and keeps Jev's own as `jev_confidence`). Jev rounds probabilities to two decimals; they
are renormalised.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Sequence

import numpy as np

from peewee_decide.data import read_jsonl, validate_record
from peewee_decide.evaluate import answer_probabilities, evaluate

logger = logging.getLogger("peewee.jev_compare")

JEV_URL = "https://api.typesafe.ai/v1/systemone"
RETRY_STATUS = (429, 500, 502, 503, 504, 529)


# ---------------------------------------------------------------------------- scoring (pure)
def from_jev(qdef: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    """One Jev answer in the shape `peewee_decide.evaluate` scores, with Peewee's confidence."""
    if qdef["type"] == "noul":
        p = float(answer["noul"])
        return {"noul": p, "confidence": max(p, 1.0 - p)}
    probs = {k: float(v) for k, v in answer["probabilities"].items()}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Jev answer has no probability mass: %s" % (answer,))
    probs = {k: v / total for k, v in probs.items()}
    return {"probabilities": probs, "confidence": max(probs.values()), "jev_confidence": answer.get("confidence")}


def _picked(qdef: Dict[str, Any], answer: Dict[str, Any]) -> int:
    p = answer_probabilities(qdef, answer)
    return int(p[1] >= 0.5) if qdef["type"] == "noul" else int(p.argmax())


def agreement(records: Sequence[Dict[str, Any]], a: Dict[str, Dict], b: Dict[str, Dict]) -> Dict[str, Any]:
    """Share of questions where both systems give the same answer, overall and per type."""
    rows = [(qdef["type"], _picked(qdef, a[r["id"]][qid]) == _picked(qdef, b[r["id"]][qid]))
            for r in records for qid, qdef in r["questions"].items()]
    types = sorted({t for t, _ in rows})
    return {"overall": float(np.mean([s for _, s in rows])),
            "by_type": {t: float(np.mean([s for u, s in rows if u == t])) for t in types}}


class _Replay:
    """An agent that answers from a cache, in record order, so `evaluate` can score it."""

    def __init__(self, records: Sequence[Dict[str, Any]], answers: Dict[str, Dict]):
        self._next = iter([answers[r["id"]] for r in records])

    def predict(self, state, questions, truncate=None):
        return {"answers": next(self._next)}


def _percentiles(ms: List[float]) -> Dict[str, float]:
    return {"p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2)}


def _latency(records: Sequence[Dict[str, Any]], cache: Dict[str, Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"latency_ms": _percentiles([cache[r["id"]]["latency_ms"] for r in records]),
                           "latency_ms_per_question": _percentiles(
                               [cache[r["id"]]["latency_ms"] / len(r["questions"]) for r in records])}
    seq = [cache[r["id"]]["latency_ms"] for r in records if cache[r["id"]].get("mode") == "sequential"]
    if seq:
        out["latency_ms_sequential"] = dict(_percentiles(seq), n=len(seq))
    return out


def report(records: Sequence[Dict[str, Any]], peewee: Dict[str, Dict], jev: Dict[str, Dict]) -> Dict[str, Any]:
    """Score both caches on the cases both answered; Jev errors exclude a case from both sides."""
    for name, cache in (("peewee", peewee), ("jev", jev)):
        missing = [r["id"] for r in records if r["id"] not in cache]
        if missing:
            raise ValueError("%s cache is missing %d case(s), e.g. %s" % (name, len(missing), missing[:5]))
    failed = sorted(r["id"] for r in records if "error" in jev[r["id"]] or "error" in peewee[r["id"]])
    kept = [r for r in records if r["id"] not in set(failed)]
    jev_answers = {r["id"]: {qid: from_jev(r["questions"][qid], a) for qid, a in jev[r["id"]]["answers"].items()}
                   for r in kept}
    peewee_answers = {r["id"]: peewee[r["id"]]["answers"] for r in kept}
    out: Dict[str, Any] = {"n_cases": len(kept), "excluded_cases": failed}
    for name, answers, cache in (("peewee", peewee_answers, peewee), ("jev", jev_answers, jev)):
        rep = evaluate(_Replay(kept, answers), kept)
        rep.pop("latency_ms_per_question", None)
        rep.update(_latency(kept, cache))
        models = sorted({cache[r["id"]]["model"] for r in kept if cache[r["id"]].get("model")})
        if models:
            rep["model"] = models
        tokens = [cache[r["id"]].get("usage", {}).get("input_tokens") for r in kept]
        if all(t is not None for t in tokens):
            rep["input_tokens"] = int(sum(tokens))
        truncated = [cache[r["id"]].get("usage", {}).get("truncated") for r in kept]
        if all(t is not None for t in truncated):
            rep["truncated_cases"] = int(sum(bool(t) for t in truncated))
        out[name] = rep
    out["agreement"] = agreement(kept, peewee_answers, jev_answers)
    return out


# ---------------------------------------------------------------------------- caches
def cache_path(out_dir: str, data: str) -> str:
    """runs/jev + data/open-jev/test.jsonl -> runs/jev/open-jev-test.jsonl"""
    parent = os.path.basename(os.path.dirname(os.path.abspath(data)))
    return os.path.join(out_dir, "%s-%s.jsonl" % (parent, os.path.splitext(os.path.basename(data))[0]))


def read_cache(path: str, skip_errors: bool = False) -> Dict[str, Dict]:
    """Rows by case id; a later row for the same id (a retry) replaces an earlier one."""
    out: Dict[str, Dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    out[row["id"]] = row
    if skip_errors:
        out = {k: v for k, v in out.items() if "error" not in v}
    return out


class _Appender:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "a")
        self._lock = threading.Lock()

    def write(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------- Jev
def call_jev(state: Any, questions: Dict[str, Any], key: str, model: str, timeout: float = 120.0,
             retries: int = 6) -> Dict[str, Any]:
    """One request. Returns the response plus `latency_ms` of the attempt that succeeded."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    for attempt in range(retries + 1):
        req = urllib.request.Request(JEV_URL, data=body, headers=headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.load(resp)
            out["latency_ms"] = (time.perf_counter() - t0) * 1000.0
            out["attempts"] = attempt + 1
            return out
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code not in RETRY_STATUS or attempt == retries:
                return {"error": "HTTP %d: %s" % (e.code, detail), "attempts": attempt + 1}
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries:
                return {"error": "%s: %s" % (type(e).__name__, e), "attempts": attempt + 1}
        time.sleep(min(60.0, 2.0 ** attempt))
    raise AssertionError("unreachable")


def question_chunks(questions: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
    """Split a case's questions into requests of at most `size`, in order. Jev answers each
    question independently, so a case too big for one request (64k tokens) can be sent in parts."""
    items = list(questions.items())
    return [dict(items[i:i + size]) for i in range(0, len(items), size)]


def ask_jev(state: Any, questions: Dict[str, Any], key: str, model: str, max_questions: int) -> Dict[str, Any]:
    """One case, in as many requests as `max_questions` needs, sent one after another; latency and
    tokens are summed over the parts."""
    out: Dict[str, Any] = {"answers": {}, "latency_ms": 0.0, "usage": {"input_tokens": 0}, "attempts": 0}
    chunks = question_chunks(questions, max_questions)
    for part in chunks:
        res = call_jev(state, part, key, model)
        out["attempts"] += res.get("attempts", 1)
        if "error" in res:
            return {"error": res["error"], "attempts": out["attempts"]}
        out["answers"].update(res["answers"])
        out["latency_ms"] += res["latency_ms"]
        out["usage"]["input_tokens"] += res["usage"]["input_tokens"]
        out["model"] = res.get("model")
    if len(chunks) > 1:
        out["requests"] = len(chunks)
    return out


def run_jev(records: Sequence[Dict[str, Any]], path: str, model: str, sequential: int, concurrency: int,
            max_questions: int = 50) -> None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("set TYPESAFE_API_KEY (e.g. `. ~/.config/typesafe/env`)")
    done = read_cache(path, skip_errors=True)
    todo = [r for r in records if r["id"] not in done]
    n_seq = max(0, sequential - sum(1 for r in done.values() if r.get("mode") == "sequential"))
    logger.info("%s: %d cached, %d to send (%d sequential)", path, len(done), len(todo), min(n_seq, len(todo)))
    out = _Appender(path)

    def one(rec, mode):
        res = ask_jev(rec["state"], rec["questions"], key, model, max_questions)
        res.update(id=rec["id"], mode=mode)
        out.write(res)
        if "error" in res:
            logger.warning("%s: %s", rec["id"], res["error"])
        return res

    try:
        for rec in todo[:n_seq]:
            one(rec, "sequential")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for i, _ in enumerate(pool.map(lambda r: one(r, "concurrent"), todo[n_seq:]), 1):
                if i % 200 == 0:
                    logger.info("%s: %d/%d", path, i, len(todo) - n_seq)
    finally:
        out.close()


# ---------------------------------------------------------------------------- Peewee
def run_peewee(records: Sequence[Dict[str, Any]], path: str, checkpoint: str, device: str) -> None:
    from peewee_decide.evaluate import load_model_for_eval
    done = read_cache(path)
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        return
    agent = load_model_for_eval(checkpoint, device)
    agent.predict(records[0]["state"], records[0]["questions"])            # warm-up, not timed
    out = _Appender(path)
    try:
        for i, rec in enumerate(todo, 1):
            t0 = time.perf_counter()
            res = agent.predict(rec["state"], rec["questions"])
            ms = (time.perf_counter() - t0) * 1000.0
            out.write({"id": rec["id"], "answers": res["answers"], "usage": res["usage"], "latency_ms": ms,
                       "model": checkpoint, "mode": "sequential"})
            if i % 500 == 0:
                logger.info("%s: %d/%d", path, i, len(todo))
    finally:
        out.close()


# ---------------------------------------------------------------------------- CLI
def _records(paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out = {}
    for p in paths:
        recs = read_jsonl(p)
        for r in recs:
            validate_record(r)
        out[p] = recs
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jev", help="answer the cases with Jev over the API")
    j.add_argument("--data", action="append", required=True)
    j.add_argument("--out-dir", required=True)
    j.add_argument("--model", default="jev-latest")
    j.add_argument("--sequential", type=int, default=200, help="cases sent one at a time for clean latency")
    j.add_argument("--concurrency", type=int, default=8)
    j.add_argument("--max-questions", type=int, default=50, help="questions per request; bigger cases are split")
    p = sub.add_parser("peewee", help="answer the cases with a Peewee checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--data", action="append", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device")
    r = sub.add_parser("report", help="score both caches")
    r.add_argument("--data", required=True)
    r.add_argument("--peewee", required=True)
    r.add_argument("--jev", required=True)
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cmd == "report":
        rep = report(read_jsonl(args.data), read_cache(args.peewee), read_cache(args.jev))
        rep["data"] = args.data
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=2)
        for name in ("peewee", "jev"):
            s, lat = rep[name]["overall"], rep[name]
            print("%-6s acc %.4f  ECE %.4f  Brier %.4f  p50 %.0f ms/case  %.1f ms/question" % (
                name, s["accuracy"], s["ece"], s["brier"], lat["latency_ms"]["p50"],
                lat["latency_ms_per_question"]["p50"]))
        print("agreement %.4f on %d cases (%d excluded)" % (rep["agreement"]["overall"], rep["n_cases"],
                                                              len(rep["excluded_cases"])))
        return 0
    for path, recs in _records(args.data).items():
        dest = cache_path(args.out_dir, path)
        if args.cmd == "jev":
            run_jev(recs, dest, args.model, args.sequential, args.concurrency, args.max_questions)
        else:
            run_peewee(recs, dest, args.checkpoint, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
