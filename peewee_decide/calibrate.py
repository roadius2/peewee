"""Fit, evaluate and persist temperature calibration for Peewee checkpoints.

The shipped checkpoints are over-confident (README: mean ECE 0.466 for `peewee_decide` before fitting)
and the multilingual checkpoint ships with no temperatures at all. This module is the missing
writer for the runtime's `temperature` / `temperature_by_options` lookup:

    import peewee_decide
    from peewee_decide.calibrate import collect_records, fit_temperature_map

    agent = peewee_decide.load("convaiinnovations/laya")
    examples = [(state, questions, {"dept": "billing", "urgent": True, "level": 2}), ...]
    records = collect_records(agent, examples)          # one forward pass per batch
    result = agent.fit_temperatures(records)            # stores the map on the agent
    print(result["report"])                              # ECE / NLL / Brier before and after
    agent.save_calibration("calibration.json")
    peewee_decide.load("convaiinnovations/laya", calibration="calibration.json")

Design follows the fitting loop in the upstream fine-tuning notebook and upstream PR #19 by
Matt Van Horn (mvanhorn): one temperature per question type, plus one per
`(type, option-count bucket)` where at least MIN_BUCKET_N labelled examples exist, each fitted
by minimising NLL over log T with LBFGS and clamped to [0.1, 10].
"""
import argparse
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .common import QTYPES, QTYPE_NAMES, ece_score, temp_bucket

MIN_BUCKET_N = 10
TEMP_LO, TEMP_HI = 0.1, 10.0
N_QTYPES = 3

# (qtype: int, logits: 1-d, target: 1-d distribution, k: int)
Record = Tuple[int, np.ndarray, np.ndarray, int]


def _vec(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------- targets
def target_from_label(qdef: Dict[str, Any], label, k: int) -> np.ndarray:
    """Turn a human label into a length-k target distribution.

    choice: the option key, its index, or a full distribution.
    score:  an integer level (one-hot), a float level (split between neighbours), or a distribution.
    noul:   a bool / 0 / 1 (one-hot over [false, true]), a float P(true), or a distribution.
    """
    t = qdef["type"]
    if isinstance(label, (list, tuple, np.ndarray)) and len(label) == k:
        v = np.asarray(label, dtype=np.float32)
        s = float(v.sum())
        if s <= 0:
            raise ValueError("target distribution must have positive mass")
        return v / s
    out = np.zeros(k, dtype=np.float32)
    if t == "choice":
        crit = qdef.get("criteria")
        keys = list(crit) if isinstance(crit, dict) else list(crit)
        if isinstance(label, str):
            if label not in keys:
                raise ValueError("label %r is not one of the options %r" % (label, keys))
            out[keys.index(label)] = 1.0
        else:
            out[int(label)] = 1.0
        return out
    if t == "score":
        x = float(label)
        lo = int(np.floor(x))
        if lo < 0 or lo > k - 1:
            raise ValueError("score label %r outside levels 0..%d" % (label, k - 1))
        hi = min(lo + 1, k - 1)
        frac = x - lo
        out[lo] += 1.0 - frac
        out[hi] += frac
        return out
    if t == "noul":
        if isinstance(label, (bool, np.bool_)) or label in (0, 1):
            y = 1.0 if bool(label) else 0.0
        else:
            y = float(label)
            if not 0.0 <= y <= 1.0:
                raise ValueError("noul label must be a bool or a probability, got %r" % (label,))
        out[0], out[1] = 1.0 - y, y
        return out
    raise ValueError("unknown question type %r" % t)


# ---------------------------------------------------------------------------- collection
def collect_records(agent, examples: Iterable, batch_size: int = 32, truncate: Optional[str] = None) -> List[Record]:
    """Run labelled examples through `agent` and keep the raw logits for fitting.

    `examples` yields `(state, questions, labels)` where `labels` maps question id to a label
    understood by `target_from_label`. Questions without a label are skipped. Forward passes
    are batched `batch_size` items at a time; nothing is written to disk.
    """
    records: List[Record] = []
    pending: List[Tuple[Dict, int, Any]] = []      # (qdef, k, label) aligned with items
    items: List[Dict] = []

    def flush():
        if not items:
            return
        logits, _act, _n = agent._forward_logits(items)
        for r, (qdef, k, label) in enumerate(pending):
            records.append((QTYPES[qdef["type"]], logits[r, :k].copy(), target_from_label(qdef, label, k), k))
        items.clear()
        pending.clear()

    for state, questions, labels in examples:
        built = agent._build_items(state, questions, truncate=truncate)
        for (qid, qdef), it in zip(questions.items(), built):
            if qid not in labels:
                continue
            items.append(it)
            pending.append((qdef, len(it["markers"]), labels[qid]))
            if len(items) >= batch_size:
                flush()
    flush()
    return records


# ---------------------------------------------------------------------------- fitting
def _pairs_to_tensors(pairs: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    kmax = max(len(z) for z, _ in pairs)
    z_mat = torch.full((len(pairs), kmax), -1e4)
    t_mat = torch.zeros((len(pairs), kmax))
    for i, (z, t) in enumerate(pairs):
        n = min(len(z), len(t))
        z_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(z[:n]))
        t_mat[i, :n] = torch.from_numpy(np.ascontiguousarray(t[:n]))
    return z_mat, t_mat


def fit_one_temperature(pairs: Sequence, min_n: int = MIN_BUCKET_N) -> float:
    """Fit one scalar T by NLL with LBFGS over log T. Returns 1.0 when there are fewer than `min_n` pairs."""
    sel = list(pairs)
    if len(sel) < min_n:
        return 1.0
    z_mat, t_mat = _pairs_to_tensors(sel)
    with torch.enable_grad():
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = -(t_mat * torch.log_softmax(z_mat / log_t.exp(), -1)).sum(-1).mean()
            loss.backward()
            return loss

        opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_LO, TEMP_HI).item())


def _softmax(z: np.ndarray, t_scale: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t_scale))
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def _lookup(temperature, temperature_by_options, qt: int, k: int) -> float:
    return float(temperature_by_options.get(temp_bucket(qt, k), temperature[qt]))


def calibration_report(records: Sequence[Record], temperature, temperature_by_options) -> Dict[str, Any]:
    """Accuracy, ECE, NLL and Brier score before and after applying a temperature map.

    Reported overall and per question type. "before" is T=1 everywhere.
    """
    groups: Dict[str, List[Record]] = {"all": list(records)}
    for rec in records:
        groups.setdefault(QTYPE_NAMES[rec[0]], []).append(rec)
    out: Dict[str, Any] = {}
    for name, recs in groups.items():
        if not recs:
            continue
        row = {"n": len(recs)}
        for tag, use_map in (("before", False), ("after", True)):
            conf, correct, nll, brier = [], [], [], []
            for qt, z, t, k in recs:
                t_scale = _lookup(temperature, temperature_by_options, qt, k) if use_map else 1.0
                p = _softmax(z[:k], t_scale)
                y = int(np.argmax(t[:k]))
                conf.append(float(p.max()))
                correct.append(float(int(p.argmax()) == y))
                nll.append(float(-(t[:k] * np.log(np.clip(p, 1e-12, 1.0))).sum()))
                brier.append(float(((p - t[:k]) ** 2).sum()))
            row[tag] = {
                "accuracy": float(np.mean(correct)),
                "ece": ece_score(np.asarray(conf), np.asarray(correct)),
                "nll": float(np.mean(nll)),
                "brier": float(np.mean(brier)),
                "mean_confidence": float(np.mean(conf)),
            }
        out[name] = row
    return out


def fit_temperature_map(records: Iterable[Record], min_bucket_n: int = MIN_BUCKET_N,
                        report: bool = True) -> Dict[str, Any]:
    """Fit type-level scalars and per-bucket temperatures from records.

    Returns `{"temperature": [3 floats], "temperature_by_options": {bucket: T},
    "n_by_bucket": {bucket: n}, "report": {...}}`. Buckets with fewer than `min_bucket_n`
    examples get no entry and fall back to the type-level scalar at runtime.
    """
    recs = [(int(qt), _vec(z)[:k], _vec(t)[:k], int(k)) for qt, z, t, k in records]
    temperature = [1.0] * N_QTYPES
    by_type: Dict[int, list] = {qt: [] for qt in range(N_QTYPES)}
    by_bucket: Dict[str, list] = {}
    for qt, z, t, k in recs:
        by_type[qt].append((z, t))
        by_bucket.setdefault(temp_bucket(qt, k), []).append((z, t))
    for qt in range(N_QTYPES):
        if by_type[qt]:
            temperature[qt] = fit_one_temperature(by_type[qt], min_bucket_n)
    temperature_by_options = {key: fit_one_temperature(pairs, min_bucket_n)
                              for key, pairs in by_bucket.items() if len(pairs) >= min_bucket_n}
    out: Dict[str, Any] = {
        "temperature": temperature,
        "temperature_by_options": temperature_by_options,
        "n_by_bucket": {key: len(pairs) for key, pairs in by_bucket.items()},
    }
    if report:
        out["report"] = calibration_report(recs, temperature, temperature_by_options)
    return out


def calibration_target(qdef: Dict[str, Any], target: Dict[str, Any], kind: str) -> List[float]:
    """What temperatures are fitted to for one question of a case record (`peewee_decide.data`):
    "probabilities" is its training distribution, "label" a one-hot of the reference answer."""
    from .data import reference_index, target_vector
    if kind == "probabilities":
        return target_vector(qdef, target)
    v = [0.0] * len(target_vector(qdef, target))
    v[reference_index(qdef, target)] = 1.0
    return v


def fit_on_cases(agent, cases: Sequence[Dict[str, Any]], target: str = "label", skip: Iterable[str] = (),
                 batch_size: int = 32) -> Optional[Dict[str, Any]]:
    """Fit a temperature map on labelled case records and apply it to `agent`.

    `skip` names "case/qid" questions to leave out; a case left with no questions is dropped.
    Returns `fit_temperature_map`'s result plus `n_records`, or None when nothing is left.
    """
    skip = set(skip)
    examples = []
    for r in cases:
        questions = {qid: qd for qid, qd in r["questions"].items() if "%s/%s" % (r["id"], qid) not in skip}
        if questions:
            targets = {qid: calibration_target(qd, r["targets"][qid], target) for qid, qd in questions.items()}
            examples.append((r["state"], questions, targets))
    recs = collect_records(agent, examples, batch_size=batch_size)
    if not recs:
        return None
    fit = fit_temperature_map(recs)
    apply_calibration_payload(agent, fit)
    fit["n_records"] = len(recs)
    return fit


# ---------------------------------------------------------------------------- persistence
def calibration_payload(temperature, temperature_by_options, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "temperature": [float(x) for x in temperature],
        "temperature_by_options": {str(k): float(v) for k, v in dict(temperature_by_options).items()},
    }
    if meta:
        payload["meta"] = meta
    return payload


def apply_calibration_payload(obj, payload: Dict[str, Any]) -> None:
    temps = payload.get("temperature")
    if temps is None or len(temps) != N_QTYPES:
        raise ValueError("calibration JSON must contain 'temperature': [3 floats]")
    obj.temperature = [float(x) for x in temps]
    obj.temperature_by_options = {str(k): float(v) for k, v in (payload.get("temperature_by_options") or {}).items()}


def save_calibration(path: str, temperature, temperature_by_options, meta: Optional[Dict[str, Any]] = None) -> None:
    with open(path, "w") as f:
        json.dump(calibration_payload(temperature, temperature_by_options, meta), f, indent=2)
        f.write("\n")


def load_calibration(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main(argv=None) -> int:
    """`peewee calibrate MODEL --data FILE --out calib.json`: fit temperatures for a workload."""
    from .data import read_jsonl, validate_record
    from .evaluate import load_model_for_eval
    from .train import _sha256
    ap = argparse.ArgumentParser(prog="peewee calibrate",
                                 description="Fit temperatures on labelled cases and write a calibration file.")
    ap.add_argument("model", help="checkpoint directory or name (english, multilingual, typed-decisions)")
    ap.add_argument("--data", required=True, help="labelled cases, JSONL (schema in peewee_decide/data.py)")
    ap.add_argument("--out", required=True, help="calibration JSON to write")
    ap.add_argument("--device", help="cuda, cpu or mps (default: best available)")
    ap.add_argument("--target", choices=["label", "probabilities"], default="label",
                    help="fit to each question's reference answer (what `peewee eval` scores) or to its target "
                         "distribution")
    args = ap.parse_args(argv)
    cases = read_jsonl(args.data)
    for r in cases:
        validate_record(r)
    agent = load_model_for_eval(args.model, args.device)
    fit = fit_on_cases(agent, cases, args.target)
    if fit is None:
        raise SystemExit("no questions to calibrate on")
    rep = fit["report"]["all"]
    agent.save_calibration(args.out, meta={
        "model": args.model, "data": args.data, "data_sha256": _sha256(args.data), "target": args.target,
        "n_records": fit["n_records"], "n_by_bucket": fit["n_by_bucket"],
        "ece_before": rep["before"]["ece"], "ece_after": rep["after"]["ece"]})
    print("%d questions: ECE %.4f -> %.4f (on the fitting data) -> %s" % (
        fit["n_records"], rep["before"]["ece"], rep["after"]["ece"], args.out))
    return 0
