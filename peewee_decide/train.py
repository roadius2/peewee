"""`peewee train`: fine-tune a Peewee checkpoint on JSONL cases (see `peewee_decide.data`) with the RLCD objective.

RLCD is the upstream recipe, lifted from `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`:
a Gaussian policy gradient over the option logits whose reward is a strictly proper scoring rule
against the target distribution (`peewee_decide.common.proper_reward`), with a group-relative (GRPO)
baseline, plus soft cross-entropy.
"""
import argparse
import collections
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .agent import _tokenizer_dir, resolve_checkpoint
from .common import build_model, collate_items, proper_reward
from .data import _group_of, read_jsonl, record_to_items, reference_index, split_cases, target_vector, validate_record

logger = logging.getLogger("peewee_decide.train")


def sigma_for_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    """Exploration noise, annealed linearly per epoch from `start` to `end` (the notebook's schedule)."""
    progress = epoch / max(1, epochs - 1)
    return start + (end - start) * progress


def rlcd_loss(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor, qtype: torch.Tensor, sigma: float,
              group_size: int = 4, generator: Optional[torch.Generator] = None, w_sph: float = 0.75,
              w_rps: float = 1.0, ce_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Policy-gradient term plus soft cross-entropy for one micro-batch (not divided by grad accumulation).

    Samples `group_size` noisy logit vectors around `logits` (zero-mean over the valid options),
    rewards each with `proper_reward` against `target`, uses the group mean as the baseline and
    the batch standard deviation as the scale, and pushes `logits` along the Gaussian score function.
    """
    m = mask.to(logits.dtype)
    k = m.sum(-1, keepdim=True)
    eps = torch.randn((group_size,) + tuple(logits.shape), device=logits.device, dtype=logits.dtype,
                      generator=generator) * sigma * m
    eps = (eps - eps.sum(-1, keepdim=True) / k) * m
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * m).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + ce_weight * loss_ce
    return loss, {"reward": float(r.mean()), "ce": float(loss_ce.detach()), "rl": float(loss_rl.detach())}


@dataclass
class TrainConfig:
    """Hyperparameters. Defaults reproduce the notebook on one GPU (effective batch 64 = 16 x 4)."""
    epochs: int = 4
    micro_batch: int = 16
    grad_accum: int = 4
    lr_encoder: float = 2.5e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    clip: float = 1.0
    group_size: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    ce_weight: float = 1.0
    calib_fraction: float = 0.1
    calibration_target: str = "probabilities"
    seed: int = 0
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    freeze_encoder: bool = False
    gradient_checkpointing: bool = True
    precision: str = "auto"
    max_skipped_fraction: float = 0.05
    log_every: int = 20


_PRECISIONS = ("auto", "bf16", "fp16", "fp32")
_CALIBRATION_TARGETS = ("probabilities", "label")
_CHOICES = {"precision": _PRECISIONS, "calibration_target": _CALIBRATION_TARGETS}
_OPTIONAL_INT_FIELDS = ("max_len", "head_max_len")


def pick_device(device: Optional[str]) -> torch.device:
    """`None` means CUDA when present, else CPU. Asking for CUDA without it is an error, never a CPU run."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; refusing to train on CPU")
    return d


def _precision(name: str, device: torch.device) -> Tuple[Optional[torch.dtype], bool]:
    """(autocast dtype or None, use a GradScaler)."""
    if name not in _PRECISIONS:
        raise ValueError("precision must be one of %s, got %r" % (_PRECISIONS, name))
    if device.type != "cuda" or name == "fp32":
        return None, False
    if name == "auto":
        name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return (torch.bfloat16, False) if name == "bf16" else (torch.float16, True)


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_base(base: str, token: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """A local checkpoint directory, a checkpoint name (`english`, `multilingual`, ...), or a hub repo id."""
    if os.path.isdir(base):
        return resolve_checkpoint(base)
    if "/" in base:
        return resolve_checkpoint(base, token=token)
    from .router import DEFAULT_MODELS, normalise_name
    repo, sub = DEFAULT_MODELS[normalise_name(base)]
    return resolve_checkpoint(repo, token=token, subfolder=sub)


def _load_model(model_dir: str, cfg: Dict[str, Any]):
    from safetensors.torch import load_file
    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.isdir(enc_dir) else None)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in weights.items()}, strict=True)
    try:
        model.encoder.config.reference_compile = False     # keep ModernBERT on the eager path
    except Exception:
        pass
    return model.float()


def _load_tokenizer(model_dir: str, cfg: Dict[str, Any]):
    from transformers import AutoTokenizer
    tok_dir = _tokenizer_dir(model_dir)
    return AutoTokenizer.from_pretrained(tok_dir if tok_dir else cfg["encoder"])


def _items(records: Sequence[Dict[str, Any]], tok, max_len: int, head_max_len: int):
    items, skipped = [], []
    for r in records:
        its, sk = record_to_items(r, tok, max_len, head_max_len)
        items.extend(its)
        skipped.extend(sk)
    return items, skipped


def _chunks(items: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward(model, b: Dict[str, Any], detach_encoder: bool = False) -> torch.Tensor:
    logits, _act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"],
                         detach_encoder=detach_encoder)
    return logits.float()


@torch.no_grad()
def _heldout_metrics(model, items, pad_id, device, amp, micro_batch) -> Dict[str, float]:
    if not items:
        return {}
    model.eval()
    ce, correct = 0.0, 0
    for chunk in _chunks(items, micro_batch):
        b = _to_device(collate_items([chunk], pad_id), device)
        with _autocast(device, amp):
            logits = _forward(model, b)
        logits = logits.float().masked_fill(~b["marker_mask"], -1e4)
        ce += float(-(b["target"] * torch.log_softmax(logits, -1)).sum(-1).sum())
        correct += int((logits.argmax(-1) == b["label"]).sum())
    model.train()
    return {"heldout_ce": ce / len(items), "heldout_accuracy": correct / len(items)}


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    """HEAD of the source checkout `peewee_decide` runs from, read from `.git` directly (the package never shells out)."""
    try:
        git = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".git")
        if os.path.isfile(git):                                 # worktree: ".git" names the real git dir
            with open(git) as f:
                git = os.path.join(os.path.dirname(git), f.read().split("gitdir:", 1)[1].strip())
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        common = git
        if os.path.isfile(os.path.join(git, "commondir")):     # worktrees keep branch refs in the main git dir
            with open(os.path.join(git, "commondir")) as f:
                common = os.path.join(git, f.read().strip())
        for d in (git, common):
            if os.path.isfile(os.path.join(d, ref)):
                with open(os.path.join(d, ref)) as f:
                    return f.read().strip()
        with open(os.path.join(common, "packed-refs")) as f:
            for line in f:
                sha, _, name = line.strip().partition(" ")
                if name == ref:
                    return sha
    except (OSError, IndexError):
        pass
    return None


def _save_checkpoint(model, tok, cfg: Dict[str, Any], out_dir: str) -> None:
    """Exactly what `peewee_decide.Agent` loads: fp16 safetensors, encoder config, tokenizer, rl_agent_config.json."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: (v.detach().half() if v.is_floating_point() else v.detach()).contiguous().cpu()
          for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    _write_json(os.path.join(out_dir, "rl_agent_config.json"), cfg)


def _calibration_target(qdef: Dict[str, Any], target: Dict[str, Any], kind: str) -> List[float]:
    """What temperatures are fitted to: the training distribution, or a one-hot of the reference answer."""
    if kind == "probabilities":
        return target_vector(qdef, target)
    v = [0.0] * len(target_vector(qdef, target))
    v[reference_index(qdef, target)] = 1.0
    return v


def _calibrate(out_dir: str, held: Sequence[Dict[str, Any]], device: torch.device,
               skipped: Sequence[str] = (), target: str = "probabilities") -> Optional[Dict[str, Any]]:
    """Load the saved checkpoint back through the runtime and fit temperatures on the held-out cases.

    `target` is `calibration_target`: "probabilities" fits NLL against each question's training
    distribution (a teacher's soft labels); "label" fits against the reference answer that
    `peewee eval` scores (the hard label, else the distribution's argmax).

    `skipped` names the "case/qid" questions `record_to_items` dropped during training (options
    that do not fit `max_len`/`head_max_len`). Those must not be sent through the runtime, which
    raises for exactly such a question instead of skipping it, so they are excluded here too; a
    record left with no questions at all is dropped.
    """
    from .agent import Agent
    from .calibrate import collect_records, fit_temperature_map
    skip = set(skipped)
    agent = Agent(out_dir, device=str(device))
    examples = []
    for r in held:
        questions = {qid: qd for qid, qd in r["questions"].items() if "%s/%s" % (r["id"], qid) not in skip}
        if not questions:
            continue
        targets = {qid: _calibration_target(qd, r["targets"][qid], target) for qid, qd in questions.items()}
        examples.append((r["state"], questions, targets))
    recs = collect_records(agent, examples, batch_size=32)
    del agent
    if not recs:
        return None
    fit = fit_temperature_map(recs)
    rep = fit["report"]["all"]
    return {"temperature": [float(x) for x in fit["temperature"]],
            "temperature_by_options": {k: float(v) for k, v in fit["temperature_by_options"].items()},
            "n_by_bucket": fit["n_by_bucket"], "n_records": len(recs),
            "heldout_ece_before": rep["before"]["ece"], "heldout_ece_after": rep["after"]["ece"]}


def _check_disjoint(records: Sequence[Dict[str, Any]], calib: Sequence[Dict[str, Any]]) -> None:
    ids = {r["id"] for r in records} & {r["id"] for r in calib}
    groups = {_group_of(r) for r in records} & {_group_of(r) for r in calib}
    if ids or groups:
        raise ValueError("calibration cases overlap the training data (ids %s, groups %s)"
                         % (sorted(ids)[:5], sorted(groups)[:5]))


def _sha256s(path: Union[str, Sequence[str], None]):
    if path is None or isinstance(path, str):
        return _sha256(path) if path else None
    return [_sha256(p) for p in path]


def train(base: str, records: Sequence[Dict[str, Any]], out_dir: str, cfg: Optional[TrainConfig] = None,
          device: Optional[str] = None, overwrite: bool = False, data_path: Union[str, Sequence[str], None] = None,
          token: Optional[str] = None, calib_records: Optional[Sequence[Dict[str, Any]]] = None,
          calib_path: Optional[str] = None, repeat: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Fine-tune `base` on `records`, calibrate on held-out cases, save to `out_dir`, return the run metadata.

    `calib_records`, when given, are the held-out cases (e.g. a dataset's official calibration
    split): training then uses every record and `calib_fraction` is ignored. `repeat` maps case
    ids to how many times their items appear per epoch; it upsamples training cases only, after
    the held-out split, so a repeated case never lands on both sides.
    """
    import transformers

    import peewee_decide
    cfg = cfg or TrainConfig()
    t_start = time.time()
    if not overwrite and any(os.path.exists(os.path.join(out_dir, n)) for n in ("model.safetensors",
                                                                                   "rl_agent_config.json")):
        raise FileExistsError("%s already holds a checkpoint; pass overwrite=True (--overwrite) to replace it" % out_dir)
    if cfg.calibration_target not in _CALIBRATION_TARGETS:
        raise ValueError("calibration_target must be one of %s, got %r"
                         % (_CALIBRATION_TARGETS, cfg.calibration_target))
    for r in list(records) + list(calib_records or ()):
        validate_record(r)
    if calib_records is not None:
        _check_disjoint(records, calib_records)
    dev = pick_device(device)
    amp, use_scaler = _precision(cfg.precision, dev)
    _seed_everything(cfg.seed)

    model_dir, base_cfg = resolve_base(base, token=token)
    tok = _load_tokenizer(model_dir, base_cfg)
    max_len = cfg.max_len or int(base_cfg.get("max_len", 512))
    head_max_len = cfg.head_max_len or int(base_cfg.get("head_max_len", 192))

    if calib_records is None:
        train_recs, held = split_cases(records, cfg.calib_fraction, cfg.seed)
    else:
        train_recs, held = list(records), list(calib_records)
    items, skipped = _items(train_recs, tok, max_len, head_max_len)
    held_items, held_skipped = _items(held, tok, max_len, head_max_len)
    all_skipped = skipped + held_skipped
    n_questions = sum(len(r["questions"]) for r in list(train_recs) + list(held))
    if n_questions and len(all_skipped) / n_questions > cfg.max_skipped_fraction:
        raise ValueError("%d of %d questions do not fit max_len=%d / head_max_len=%d (first: %s); raise the lengths "
                         "or shorten the options" % (len(all_skipped), n_questions, max_len, head_max_len,
                                                     all_skipped[:3]))
    if not items:
        raise ValueError("no training items")
    if all_skipped:
        logger.warning("peewee_decide.train: skipped %d questions whose options do not fit (first: %s)",
                       len(all_skipped), all_skipped[:3])
    n_unique = len(items)
    if repeat:
        if any(int(n) < 1 for n in repeat.values()):
            raise ValueError("repeat counts must be at least 1")
        items = [it for it in items for _ in range(int(repeat.get(it["case"], 1)))]

    model = _load_model(model_dir, base_cfg).to(dev)
    model.train()
    if cfg.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    elif cfg.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": head_params, "lr": cfg.lr_head}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": cfg.lr_encoder})
    trainable = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(items) / (cfg.micro_batch * cfg.grad_accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps_per_epoch * cfg.epochs), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    pad_id = tok.pad_token_id
    logger.info("peewee_decide.train: %d train items from %d cases, %d held-out cases, max_len %d, %s on %s",
                len(items), len(train_recs), len(held), max_len, amp or "fp32", dev)

    epochs_log: List[Dict[str, Any]] = []
    step = 0
    for epoch in range(cfg.epochs):
        sigma = sigma_for_epoch(epoch, cfg.epochs, cfg.sigma_start, cfg.sigma_end)
        order = list(items)
        random.Random(cfg.seed + epoch).shuffle(order)
        chunks = list(_chunks(order, cfg.micro_batch))
        sums = {"loss": 0.0, "reward": 0.0, "ce": 0.0}
        opt.zero_grad(set_to_none=True)
        for i, chunk in enumerate(chunks):
            b = _to_device(collate_items([chunk], pad_id), dev)
            with _autocast(dev, amp):
                logits = _forward(model, b, detach_encoder=cfg.freeze_encoder)
            loss, st = rlcd_loss(logits.float(), b["marker_mask"], b["target"], b["qtype"], sigma, cfg.group_size,
                                 gen, cfg.w_sph, cfg.w_rps, cfg.ce_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss at epoch %d, micro-batch %d; nothing was saved" % (epoch + 1, i + 1))
            scaler.scale(loss / cfg.grad_accum).backward()
            sums["loss"] += float(loss.detach())
            sums["reward"] += st["reward"]
            sums["ce"] += st["ce"]
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(chunks):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % cfg.log_every == 0:
                    logger.info("peewee_decide.train: epoch %d step %d loss %.4f reward %.4f lr %.2e", epoch + 1, step,
                                float(loss.detach()), st["reward"], sched.get_last_lr()[0])
        n = len(chunks)
        rec = {"epoch": epoch + 1, "sigma": sigma, "train_loss": sums["loss"] / n, "train_ce": sums["ce"] / n,
               "train_reward": sums["reward"] / n}
        rec.update(_heldout_metrics(model, held_items, pad_id, dev, amp, cfg.micro_batch))
        epochs_log.append(rec)
        logger.info("peewee_decide.train: epoch %d done: %s", epoch + 1, json.dumps(rec))

    if cfg.gradient_checkpointing and not cfg.freeze_encoder:
        model.encoder.gradient_checkpointing_disable()
    out_cfg = dict(base_cfg)
    out_cfg.update({"max_len": max_len, "head_max_len": head_max_len, "fine_tuned": True,
                    "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}})
    _save_checkpoint(model, tok, out_cfg, out_dir)
    del model, opt, sched, scaler
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    calibration = _calibrate(out_dir, held, dev, held_skipped, cfg.calibration_target) if held else None
    if calibration:
        out_cfg["temperature"] = calibration["temperature"]
        out_cfg["temperature_by_options"] = calibration["temperature_by_options"]
        _write_json(os.path.join(out_dir, "rl_agent_config.json"), out_cfg)
    else:
        logger.warning("peewee_decide.train: no held-out questions; temperatures left at 1.0")

    meta = {"base": base, "data": data_path, "data_sha256": _sha256s(data_path),
            "calib_data": calib_path, "calib_data_sha256": _sha256s(calib_path),
            "config": asdict(cfg), "device": str(dev),
            "precision": {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[amp],
            "max_len": max_len, "head_max_len": head_max_len,
            "cases": {"train": len(train_recs), "heldout": len(held)},
            "items": {"train": len(items), "train_unique": n_unique, "heldout": len(held_items)},
            "repeated_cases": {str(n): c for n, c in sorted(collections.Counter(repeat.values()).items()) if n > 1}
            if repeat else {},
            "skipped_questions": {"count": len(all_skipped), "first": all_skipped[:50]},
            "optimizer_steps": step, "epochs": epochs_log, "calibration": calibration,
            "git_commit": _git_commit(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "peewee": peewee_decide.__version__},
            "wall_seconds": round(time.time() - t_start, 1)}
    _write_json(os.path.join(out_dir, "train_meta.json"), meta)
    return meta


_CONFIG_HELP = {
    "epochs": "number of passes over the training data",
    "micro_batch": "number of items per forward/backward pass",
    "grad_accum": "number of micro-batches accumulated before an optimizer step",
    "lr_encoder": "learning rate for the encoder parameters",
    "lr_head": "learning rate for the head parameters",
    "weight_decay": "AdamW weight decay",
    "clip": "gradient norm clip value",
    "group_size": "number of noisy samples drawn per item for the policy gradient",
    "sigma_start": "exploration noise standard deviation at the first epoch",
    "sigma_end": "exploration noise standard deviation at the last epoch",
    "w_sph": "weight of the spherical scoring term in the reward",
    "w_rps": "weight of the ranked probability scoring term in the reward",
    "ce_weight": "weight of the soft cross-entropy term in the loss",
    "calib_fraction": "held-out share of cases used to fit temperatures (ignored with --calib-data)",
    "calibration_target": "fit temperatures to the training distribution or to the reference answer (hard label)",
    "seed": "random seed for shuffling, noise and initialisation",
    "max_len": "maximum input token length (default: from the base checkpoint)",
    "head_max_len": "maximum token length reserved for the question and options",
    "freeze_encoder": "keep the encoder weights fixed during training",
    "gradient_checkpointing": "trade compute for memory by recomputing encoder activations",
    "precision": "numeric precision to train in",
    "max_skipped_fraction": "largest share of skipped questions before training aborts",
    "log_every": "log training progress every this many optimizer steps",
}


def _add_config_args(ap: argparse.ArgumentParser) -> None:
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        help_ = _CONFIG_HELP.get(f.name)
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default, help=help_)
        elif f.name in _CHOICES:
            ap.add_argument(flag, type=str, default=f.default, choices=_CHOICES[f.name], help=help_)
        else:
            typ = int if f.name in _OPTIONAL_INT_FIELDS else type(f.default)
            ap.add_argument(flag, type=typ, default=f.default, help=help_)


def _data_spec(value: str) -> Tuple[str, int]:
    """`FILE` or `FILE:N`; a suffix that is not an integer stays part of the path."""
    path, sep, n = value.rpartition(":")
    if not sep or not n.lstrip("-").isdigit():
        return value, 1
    if int(n) < 1:
        raise argparse.ArgumentTypeError("repeat count must be at least 1: %r" % value)
    return path, int(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="peewee train", description="Fine-tune a Peewee checkpoint on JSONL cases (RLCD).")
    ap.add_argument("--data", required=True, action="append", type=_data_spec, metavar="FILE[:N]",
                    help="training cases, JSONL (schema in peewee_decide/data.py); repeat to mix files, and add :N to "
                         "repeat that file's training cases N times per epoch")
    ap.add_argument("--calib-data", metavar="FILE",
                    help="held-out cases for calibration and per-epoch metrics, instead of splitting --data")
    ap.add_argument("--base", required=True, help="checkpoint name (english, multilingual, ...), hub id, or directory")
    ap.add_argument("--out", required=True, help="output checkpoint directory")
    ap.add_argument("--device", help="cuda (default when present) or cpu")
    ap.add_argument("--overwrite", action="store_true", help="replace a checkpoint already in --out")
    ap.add_argument("--token", help="hub token for private bases (default: HF_TOKEN)")
    _add_config_args(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    records, repeat = [], {}
    for path, n in args.data:
        recs = read_jsonl(path)
        records.extend(recs)
        if n > 1:
            repeat.update((r["id"], n) for r in recs)
    paths = [p for p, _ in args.data]
    calib = read_jsonl(args.calib_data) if args.calib_data else None
    meta = train(args.base, records, args.out, cfg, device=args.device, overwrite=args.overwrite,
                 data_path=paths[0] if len(paths) == 1 else paths, token=args.token, calib_records=calib,
                 calib_path=args.calib_data, repeat=repeat or None)
    last = meta["epochs"][-1]
    cal = meta["calibration"] or {}
    print("saved %s: %d optimizer steps in %.0fs; last epoch train CE %.4f, held-out accuracy %s; "
          "held-out ECE %s -> %s" % (args.out, meta["optimizer_steps"], meta["wall_seconds"], last["train_ce"],
                                     last.get("heldout_accuracy", "n/a"), cal.get("heldout_ece_before", "n/a"),
                                     cal.get("heldout_ece_after", "n/a")))
    return 0
