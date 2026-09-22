"""Core model architecture, token sequence construction, and confidence estimation for Peewee."""
import json
import math
import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes compact JSON, so a
    rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values
        return [k if v is None or v == "" else "%s: %s" % (k, render_criterion(v)) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


# Option text is capped at this many tokens per option before any budget squeeze.
OPTION_TOKEN_CAP = 48
# Slack kept for the instructions when options overflow the head budget, and the per-option floor.
HEAD_OPTION_SLACK = 16
MIN_OPTION_TOKENS = 4


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
    return_info: bool = False,
):
    """Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP].

    Returns `(ids, markers)`, or `(ids, markers, info)` when `return_info` is set. `info`
    records everything that was cut so callers can report it instead of hiding it:

        state_tokens            tokens in the full serialised state
        state_tokens_kept       how many made it into the sequence
        state_tokens_dropped    how many did not
        truncated               state_tokens_dropped > 0
        truncation              "left" (kept the end) or "right" (kept the start)
        options                 number of options
        tokens_per_option       per-option token cap actually applied (None if not squeezed)
        options_squeezed        True when options were cut below OPTION_TOKEN_CAP to fit head_max_len
        options_over_cap        options whose text exceeded OPTION_TOKEN_CAP before any squeeze
        instructions_tokens_dropped   instruction tokens cut to make room for options
    """
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    head_full = tok("%s question: %s" % (q["t"], ins), add_special_tokens=False)["input_ids"]
    opt_ids = []
    over_cap = 0
    for i in order:
        raw = tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
        if len(raw) > OPTION_TOKEN_CAP:
            over_cap += 1
        opt_ids.append([tok.mask_token_id] + raw[:OPTION_TOKEN_CAP])
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    per = None
    if opt_budget < HEAD_OPTION_SLACK:
        per = max(MIN_OPTION_TOKENS, (head_max_len - HEAD_OPTION_SLACK) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_full[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st_full = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st_full[-room:] if truncate_left else st_full[:room]
    if room == 0:
        st = []
    ids = ids + st + [tok.sep_token_id]
    ids, markers = ids[:max_len], [m for m in markers if m < max_len]
    if not return_info:
        return ids, markers
    info = {
        "state_tokens": len(st_full),
        "state_tokens_kept": len(st),
        "state_tokens_dropped": len(st_full) - len(st),
        "truncated": len(st_full) > len(st),
        "truncation": "left" if truncate_left else "right",
        "options": len(opt_ids),
        "tokens_per_option": (per - 1) if per is not None else None,   # per counts the [MASK]
        "options_squeezed": per is not None,
        "options_over_cap": over_cap,
        "instructions_tokens_dropped": len(head_full) - len(head_ids),
    }
    return ids, markers, info


def _manual_self_attention(mha: nn.MultiheadAttention, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """`mha(x, x, x, key_padding_mask=pad)` written out in plain ops (batch_first, no bias_kv).

    PyTorch's fused fast path traces with the sample's sequence length baked into a reshape,
    which breaks ONNX export for any other length. This path is shape-dynamic and numerically
    the same; it is only used when `DecisionModel.manual_head_attention` is set (export).
    """
    B, L, D = x.shape
    H = mha.num_heads
    hd = D // H
    qkv = nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.reshape(B, L, H, hd).transpose(1, 2)
    k = k.reshape(B, L, H, hd).transpose(1, 2)
    v = v.reshape(B, L, H, hd).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)
    scores = scores.masked_fill(pad[:, None, None, :], -1e4)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, D)
    return mha.out_proj(out)


def _manual_encoder_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Pre-norm (`norm_first=True`) encoder layer in plain ops; see `_manual_self_attention`."""
    a = _manual_self_attention(layer.self_attn, layer.norm1(x), pad)
    x = x + layer.dropout1(a)
    f = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
    return x + layer.dropout2(f)


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False
        self.manual_head_attention = False     # set during ONNX export; see _manual_self_attention

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                if self.manual_head_attention:
                    h = _manual_encoder_layer(layer, h, pad)
                else:
                    h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    from transformers import AutoConfig, AutoModel

    if encoder_dir and os.path.exists(encoder_dir):
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def normalized_entropy(p: np.ndarray, k: int) -> float:
    """H(p) / log(k) in [0, 1]: 0 for a one-hot answer, 1 for a uniform one."""
    if k < 2:
        return 0.0
    p = np.asarray(p, dtype=np.float64)[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(ent / math.log(k), 0.0, 1.0))


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Entropy-based confidence, 1 - H(p) / log(k). Kept for callers of the old API.

    Since 0.4 the `confidence` field in answers is `top_probability`, which is the same
    quantity for every question type and is what calibration acts on.
    """
    return 1.0 - normalized_entropy(p, k)


def top_probability(p: np.ndarray, k: int) -> float:
    """Probability of the reported answer: max over the k options."""
    if k < 1:
        return 1.0
    return float(np.max(np.asarray(p, dtype=np.float64)[:k]))


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def collate_items(batch, pad_id: int):
    items = [it for group in batch for it in group]
    if not items:
        return None
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
    }
    if target is not None:
        res["target"] = target
    return res
