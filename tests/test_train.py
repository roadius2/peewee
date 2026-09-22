"""laya.train: the RLCD objective and the trainer, on a tiny random model. No weights, no network."""
import pytest

torch = pytest.importorskip("torch")

from laya.common import QTYPES, proper_reward  # noqa: E402
from laya.train import rlcd_loss, sigma_for_epoch  # noqa: E402


def _batch():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.3, -0.2, 0.0], [1.0, -1.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False], [True, True, True]])
    target = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.6, 0.0], [0.1, 0.3, 0.6]])
    qtype = torch.tensor([QTYPES["choice"], QTYPES["noul"], QTYPES["score"]])
    return logits, mask, target, qtype


def _notebook_loss(logits, mask, target, qtype, sigma, group_size, generator):
    """The upstream notebook's per-micro-batch loss, transcribed (without the / GRAD_ACCUM)."""
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, generator=generator) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    return loss_rl + 1.0 * loss_ce


def test_rlcd_loss_matches_the_notebook():
    logits, mask, target, qtype = _batch()
    ours, stats = rlcd_loss(logits, mask, target, qtype, sigma=0.3, group_size=4,
                            generator=torch.Generator().manual_seed(11))
    ref = _notebook_loss(logits, mask, target, qtype, 0.3, 4, torch.Generator().manual_seed(11))
    assert torch.allclose(ours, ref, atol=1e-6)
    assert set(stats) == {"reward", "ce", "rl"}


def test_masked_options_get_no_gradient():
    logits, mask, target, qtype = _batch()
    loss, _ = rlcd_loss(logits, mask, target, qtype, sigma=0.3, group_size=4, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert logits.grad[1, 2] == 0
    assert logits.grad[mask].abs().sum() > 0


def test_optimising_the_loss_moves_logits_towards_the_target():
    logits, mask, target, qtype = _batch()
    logits = logits.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    gen = torch.Generator().manual_seed(0)

    def ce():
        return float(-(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean())

    before = ce()
    for _ in range(200):
        opt.zero_grad()
        loss, _ = rlcd_loss(logits, mask, target, qtype, sigma=0.2, group_size=4, generator=gen)
        loss.backward()
        opt.step()
    assert ce() < before - 0.1


def test_sigma_anneals_per_epoch():
    assert sigma_for_epoch(0, 4, 0.4, 0.1) == pytest.approx(0.4)
    assert sigma_for_epoch(1, 4, 0.4, 0.1) == pytest.approx(0.3)
    assert sigma_for_epoch(3, 4, 0.4, 0.1) == pytest.approx(0.1)
    assert sigma_for_epoch(0, 1, 0.4, 0.1) == pytest.approx(0.4)
