"""`laya train`: fine-tune a Laya checkpoint on JSONL cases (see `laya.data`) with the RLCD objective.

RLCD is the upstream recipe, lifted from `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`:
a Gaussian policy gradient over the option logits whose reward is a strictly proper scoring rule
against the target distribution (`laya.common.proper_reward`), with a group-relative (GRPO)
baseline, plus soft cross-entropy.
"""
import logging
from typing import Dict, Optional, Tuple

import torch

from .common import proper_reward

logger = logging.getLogger("laya.train")


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
