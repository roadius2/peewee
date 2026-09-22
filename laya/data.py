"""Training and evaluation data for `laya train` and `laya eval`.

One JSONL record per case::

    {"id": "case-001",
     "state": "..." | {...} | [...],
     "questions": {"qid": {"type": "choice" | "score" | "noul", "instructions": "...", "criteria": ...}},
     "targets":   {"qid": {"probabilities": {...}, "label": ...}},
     "meta":      {"workflow": "..."}}

A target holds `probabilities` (keys: choice keys, score levels "0".."k-1", or "false"/"true"),
a hard `label`, or both. `probabilities` is the training signal when present; `label` is the
reference answer for accuracy when present. Option order is `laya.common.render_options` order.
"""
import json
from typing import Any, Dict, List

import numpy as np

from .agent import Agent
from .calibrate import target_from_label
from .common import QTYPES

_TRUE = ("true", "yes", "1")
_FALSE = ("false", "no", "0")


def option_keys(qdef: Dict[str, Any]) -> List[str]:
    """Option names in label-index order: choice keys, score levels "0".."k-1", or false/true."""
    q = Agent._to_internal(qdef)
    if q["t"] == "choice":
        return [str(k) for k in q["crit"]]
    if q["t"] == "score":
        return [str(i) for i in range(len(q["crit"]))]
    return ["false", "true"]


def _as_bool(label: Any) -> bool:
    if isinstance(label, str):
        low = label.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError("noul label must be true or false, got %r" % (label,))
    if isinstance(label, (bool, np.bool_)):
        return bool(label)
    return float(label) >= 0.5


def target_vector(qdef: Dict[str, Any], target: Dict[str, Any]) -> List[float]:
    """The training distribution over the options, normalised, in option order."""
    keys = option_keys(qdef)
    probs = target.get("probabilities")
    if probs is not None:
        if isinstance(probs, dict):
            unknown = sorted(set(map(str, probs)) - set(keys))
            if unknown:
                raise ValueError("unknown option(s) %s; options are %s" % (unknown, keys))
            v = np.array([float(probs.get(k, 0.0)) for k in keys])
        else:
            v = np.asarray(probs, dtype=float)
            if v.shape != (len(keys),):
                raise ValueError("expected %d probabilities, got %d" % (len(keys), v.size))
        if (v < 0).any():
            raise ValueError("negative probability in %s" % (probs,))
        if v.sum() <= 0:
            raise ValueError("probabilities have no mass")
        return (v / v.sum()).tolist()
    if "label" in target:
        label = target["label"]
        if qdef["type"] == "noul" and isinstance(label, str):
            label = _as_bool(label)
        return [float(x) for x in target_from_label(qdef, label, len(keys))]
    raise ValueError("target needs 'probabilities' or 'label'")


def reference_index(qdef: Dict[str, Any], target: Dict[str, Any]) -> int:
    """Index of the reference answer: the label when there is one, else the target's argmax
    (noul: p(true) >= 0.5)."""
    keys = option_keys(qdef)
    label = target.get("label")
    if label is not None:
        t = qdef["type"]
        if t == "noul":
            return int(_as_bool(label))
        if t == "score":
            i = int(round(float(label)))
        elif str(label) in keys:
            i = keys.index(str(label))
        else:
            i = int(label)
        if not 0 <= i < len(keys):
            raise ValueError("label %r outside the %d options" % (label, len(keys)))
        return i
    v = target_vector(qdef, target)
    if qdef["type"] == "noul":
        return int(v[1] >= 0.5)
    return int(np.argmax(v))


def validate_record(rec: Dict[str, Any]) -> None:
    """Raise `ValueError("<case>/<question>: ...")` for anything training or evaluation would trip on."""
    rid = rec.get("id")
    if not isinstance(rid, str) or not rid:
        raise ValueError("record without a string 'id': %s" % (json.dumps(rec, default=str)[:80],))
    if "state" not in rec:
        raise ValueError("%s: missing 'state'" % rid)
    qs, ts = rec.get("questions"), rec.get("targets")
    if not isinstance(qs, dict) or not qs:
        raise ValueError("%s: 'questions' must be a non-empty object" % rid)
    if not isinstance(ts, dict):
        raise ValueError("%s: 'targets' must be an object" % rid)
    for qid in ts:
        if qid not in qs:
            raise ValueError("%s/%s: target for a question that does not exist" % (rid, qid))
    for qid, qdef in qs.items():
        where = "%s/%s" % (rid, qid)
        t = qdef.get("type") if isinstance(qdef, dict) else None
        if t not in QTYPES:
            raise ValueError("%s: type must be one of %s, got %r" % (where, sorted(QTYPES), t))
        if not qdef.get("instructions"):
            raise ValueError("%s: missing 'instructions'" % where)
        crit = qdef.get("criteria")
        if t == "choice" and not (isinstance(crit, (dict, list)) and len(crit) >= 2):
            raise ValueError("%s: choice needs at least two criteria" % where)
        if t == "score" and not (isinstance(crit, list) and len(crit) >= 2):
            raise ValueError("%s: score needs a list of at least two levels" % where)
        if qid not in ts:
            raise ValueError("%s: no target" % where)
        try:
            target_vector(qdef, ts[qid])
            reference_index(qdef, ts[qid])
        except (ValueError, TypeError, IndexError) as e:
            raise ValueError("%s: %s" % (where, e)) from None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as e:
                raise ValueError("%s:%d: not valid JSON (%s)" % (path, n, e)) from None
    return out


def write_jsonl(path: str, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
