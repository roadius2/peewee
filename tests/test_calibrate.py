"""Temperature fitting on synthetic logits: no weights."""
import numpy as np
import pytest

from peewee_decide.calibrate import (
    calibration_report,
    collect_records,
    fit_one_temperature,
    fit_temperature_map,
    target_from_label,
)

CHOICE = {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None, "c": None}}
SCORE = {"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "x"}


# ------------------------------------------------------------------ labels -> targets
@pytest.mark.parametrize("qdef, label, k, want", [
    (CHOICE, "b", 3, [0, 1, 0]),
    (CHOICE, 2, 3, [0, 0, 1]),
    (CHOICE, [0.2, 0.3, 0.5], 3, [0.2, 0.3, 0.5]),
    (SCORE, 1, 3, [0, 1, 0]),
    (SCORE, 1.25, 3, [0, 0.75, 0.25]),
    (SCORE, 2.0, 3, [0, 0, 1]),
    (NOUL, True, 2, [0, 1]),
    (NOUL, 0, 2, [1, 0]),
    (NOUL, 0.8, 2, [0.2, 0.8]),
])
def test_target_from_label(qdef, label, k, want):
    assert target_from_label(qdef, label, k) == pytest.approx(want)


def test_bad_labels_raise():
    with pytest.raises(ValueError):
        target_from_label(CHOICE, "zzz", 3)
    with pytest.raises(ValueError):
        target_from_label(SCORE, 5, 3)
    with pytest.raises(ValueError):
        target_from_label(NOUL, 1.5, 2)


# ------------------------------------------------------------------ fitting
def overconfident_records(n=200, k=4, scale=8.0, acc=0.6, seed=0, qtype=0):
    """Logits that are right `acc` of the time but always shout: the shipped-checkpoint shape."""
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        y = int(rng.integers(0, k))
        pred = y if rng.random() < acc else int(rng.integers(0, k))
        z = rng.normal(0, 0.3, k).astype(np.float32)
        z[pred] += scale
        recs.append((qtype, z, np.eye(k, dtype=np.float32)[y], k))
    return recs


def test_fit_one_temperature_cools_overconfidence():
    pairs = [(z, t) for _, z, t, _ in overconfident_records()]
    T = fit_one_temperature(pairs)
    assert T > 2.0


def test_fit_one_temperature_needs_enough_data():
    pairs = [(z, t) for _, z, t, _ in overconfident_records(n=5)]
    assert fit_one_temperature(pairs) == 1.0


def test_fit_temperature_map_improves_calibration():
    recs = overconfident_records(k=4, qtype=0) + overconfident_records(k=2, qtype=2, seed=1)
    out = fit_temperature_map(recs)
    assert set(out["temperature_by_options"]) == {"choice:3-5", "noul:2"}
    assert out["n_by_bucket"] == {"choice:3-5": 200, "noul:2": 200}
    assert out["temperature"][1] == 1.0                      # no score records: untouched
    rep = out["report"]
    for grp in ("all", "choice", "noul"):
        assert rep[grp]["after"]["ece"] < rep[grp]["before"]["ece"]
        assert rep[grp]["after"]["nll"] < rep[grp]["before"]["nll"]
        assert rep[grp]["after"]["accuracy"] == rep[grp]["before"]["accuracy"]   # T never changes argmax
    assert rep["all"]["n"] == 400


def test_small_bucket_falls_back_to_type_scalar():
    recs = overconfident_records(n=100, k=4) + overconfident_records(n=5, k=8, seed=2)
    out = fit_temperature_map(recs)
    assert "choice:6-10" not in out["temperature_by_options"]
    assert out["temperature"][0] > 1.0


def test_report_without_map_is_identity():
    recs = overconfident_records(n=50)
    rep = calibration_report(recs, [1.0, 1.0, 1.0], {})
    assert rep["all"]["before"] == rep["all"]["after"]


# ------------------------------------------------------------------ collection through an agent
def test_collect_records_uses_agent_pipeline(fake_tok):
    from tests.test_postprocess import bare_agent, fake_forward
    a = bare_agent(fake_tok)
    a._forward_logits = fake_forward
    qs = {"c": CHOICE, "s": SCORE, "n": NOUL}
    examples = [("one", qs, {"c": "a", "n": True}), ("two", qs, {"s": 2}), ("three", qs, {"c": "c", "s": 0, "n": 0.25})]
    recs = collect_records(a, examples, batch_size=2)
    assert [(qt, k) for qt, _, _, k in recs] == [(0, 3), (2, 2), (1, 3), (0, 3), (1, 3), (2, 2)]
    assert recs[0][1].tolist() == [0.0, 1.0, 2.0]                       # raw logits kept
    assert recs[-1][2].tolist() == pytest.approx([0.75, 0.25])
