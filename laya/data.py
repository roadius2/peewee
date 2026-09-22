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
import argparse
import collections
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .agent import Agent
from .calibrate import target_from_label
from .common import QTYPES, build_sequence, render_options

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
    if "meta" in rec and not isinstance(rec["meta"], dict):
        raise ValueError("%s: 'meta' must be an object" % rid)
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


def record_to_items(rec: Dict[str, Any], tok, max_len: int, head_max_len: int,
                    truncate: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Tokenise every question of one case into training items for `collate_items`.

    Returns `(items, skipped)`; `skipped` names "case/qid" for questions whose option markers do
    not all fit in `max_len`/`head_max_len`. Truncation follows the runtime default (keep the start
    of strings and dicts, the end of lists) unless `truncate` is "left" or "right".
    """
    state = rec["state"]
    side = truncate or ("left" if isinstance(state, (list, tuple)) else "right")
    items, skipped = [], []
    for qid, qdef in rec["questions"].items():
        q = Agent._to_internal(qdef)
        ids, markers = build_sequence(tok, state, q, max_len, head_max_len, truncate_left=(side == "left"))
        if len(markers) != len(render_options(q)):
            skipped.append("%s/%s" % (rec["id"], qid))
            continue
        tgt = rec["targets"][qid]
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": target_vector(qdef, tgt),
                      "label": reference_index(qdef, tgt), "case": rec["id"], "qid": qid})
    return items, skipped


def _group_of(rec: Dict[str, Any]) -> str:
    return str((rec.get("meta") or {}).get("group") or rec["id"])


def split_cases(records: Sequence[Dict[str, Any]], fraction: float,
                seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split whole cases into `(train, held_out)`, keeping every case of a `meta.group` together.

    Deterministic for a given set of ids and seed, whatever order the records arrive in. A
    positive fraction holds out at least one group and always leaves at least one for training.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("held-out fraction must be in [0, 1), got %r" % (fraction,))
    ids = [r["id"] for r in records]
    dups = sorted(i for i, c in collections.Counter(ids).items() if c > 1)
    if dups:
        raise ValueError("duplicate case ids: %s" % dups[:5])
    groups = sorted({_group_of(r) for r in records})
    n = int(round(len(groups) * fraction))
    if fraction > 0 and len(groups) > 1:
        n = min(max(n, 1), len(groups) - 1)
    random.Random(seed).shuffle(groups)
    held = set(groups[:n])
    return [r for r in records if _group_of(r) not in held], [r for r in records if _group_of(r) in held]


TYPED_DECISIONS = "LocalLLaMA/typed-decisions"


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def convert_typed_decisions_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """One `LocalLLaMA/typed-decisions` row as a case record: teacher distribution plus hard label."""
    questions = _maybe_json(row["questions"])
    gold = _maybe_json(row["gold"])
    targets = {qid: {"probabilities": gold[qid]["probabilities"], "label": gold[qid]["label"]} for qid in questions}
    meta = {k: row[k] for k in ("workflow", "split", "label_agreement") if k in row}
    return {"id": str(row["id"]), "state": _maybe_json(row["state"]), "questions": questions,
            "targets": targets, "meta": meta}


def _load_dataset():
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("`laya prepare-data` needs the datasets package: pip install 'laya[train]'") from e
    return load_dataset


def convert_typed_decisions(split: str) -> List[Dict[str, Any]]:
    return [convert_typed_decisions_row(r) for r in _load_dataset()(TYPED_DECISIONS, "all", split=split)]


OPEN_JEV = "ZefanCai/Open-Jev"
OPEN_JEV_REVISION = "c67699e13d0ae25e35b77165a4b6b079bedc8aba"
OPEN_JEV_SPLITS = ("train", "calibration", "validation", "test", "ood")
OPEN_JEV_DEFAULT_CONFIG = "release-v2-redistributable"


def open_jev_question(row: Dict[str, Any]) -> Dict[str, Any]:
    """One Open-Jev row's question in laya's question format."""
    kind, options = row["kind"], [str(o) for o in row["options"]]
    if len(set(options)) != len(options):
        raise ValueError("%s: duplicate option strings %s" % (row["id"], options))
    if kind == "choice":
        return {"type": "choice", "instructions": row["question"], "criteria": {o: None for o in options}}
    if kind == "score":
        return {"type": "score", "instructions": row["question"], "criteria": options}
    if kind == "noul":
        if [o.lower() for o in options] != ["no", "yes"]:
            raise ValueError("%s: noul options must be ['no', 'yes'], got %s" % (row["id"], options))
        return {"type": "noul", "instructions": row["question"]}
    raise ValueError("%s: unknown kind %r" % (row["id"], kind))


def convert_open_jev_rows(rows) -> List[Dict[str, Any]]:
    """Open-Jev rows (one question each) as case records: one case per distinct state within a group."""
    cases: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (row["group_id"], row["state_json"])
        rec = cases.get(key)
        if rec is None:
            digest = hashlib.sha1(row["state_json"].encode("utf-8")).hexdigest()[:12]
            rec = {"id": "%s:%s" % (row["group_id"], digest), "state": json.loads(row["state_json"]),
                   "questions": {}, "targets": {},
                   "meta": {"group": row["group_id"], "workflow": row["source"], "split": row["split"]}}
            cases[key] = rec
        rec["questions"][row["id"]] = open_jev_question(row)
        rec["targets"][row["id"]] = {"probabilities": [float(x) for x in row["target"]]}
    return list(cases.values())


def convert_open_jev(config: str, split: str, revision: str = OPEN_JEV_REVISION) -> List[Dict[str, Any]]:
    return convert_open_jev_rows(_load_dataset()(OPEN_JEV, config, split=split, revision=revision))


def _write_split(out_dir: str, split: str, records: List[Dict[str, Any]]) -> None:
    for r in records:
        validate_record(r)
    path = os.path.join(out_dir, split + ".jsonl")
    write_jsonl(path, records)
    print("%s: %d cases, %d questions -> %s" % (split, len(records), sum(len(r["questions"]) for r in records), path))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="laya prepare-data", description="Write a public dataset as laya training JSONL.")
    ap.add_argument("dataset", choices=["typed-decisions", "open-jev"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--config", default=None, help="open-jev only: dataset config")
    ap.add_argument("--revision", default=None, help="open-jev only: pinned dataset revision")
    args = ap.parse_args(argv)
    if args.dataset == "typed-decisions":
        if args.config is not None or args.revision is not None:
            ap.error("--config and --revision only apply to open-jev")
    else:
        if args.config is None:
            args.config = OPEN_JEV_DEFAULT_CONFIG
        if args.revision is None:
            args.revision = OPEN_JEV_REVISION
    os.makedirs(args.out, exist_ok=True)
    if args.dataset == "typed-decisions":
        for split in ("train", "test"):
            _write_split(args.out, split, convert_typed_decisions(split))
    else:
        for split in OPEN_JEV_SPLITS:
            _write_split(args.out, split, convert_open_jev(args.config, split, args.revision))
    return 0
