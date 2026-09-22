"""peewee_decide.train: the RLCD objective and the trainer, on a tiny random model. No weights, no network."""
import pytest

torch = pytest.importorskip("torch")

from peewee_decide.common import QTYPES, proper_reward  # noqa: E402
from peewee_decide.train import TrainConfig, main as train_main, pick_device, rlcd_loss, sigma_for_epoch, train  # noqa: E402
from tests.conftest import toy_records  # noqa: E402


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

    @torch.no_grad()
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


FAST = dict(epochs=2, micro_batch=8, grad_accum=1, lr_encoder=1e-3, lr_head=1e-2, gradient_checkpointing=False,
            calib_fraction=0.25, log_every=1)


def _read(path):
    import json
    with open(path) as f:
        return json.load(f)


def test_train_writes_a_checkpoint_the_runtime_loads(tiny_base, tmp_path):
    from safetensors import safe_open

    from peewee_decide.agent import Agent
    out = tmp_path / "run"
    meta = train(tiny_base, toy_records(), str(out), TrainConfig(**FAST), device="cpu")
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json",
                 "train_meta.json"):
        assert (out / name).exists(), name
    cfg = _read(str(out / "rl_agent_config.json"))
    assert cfg["fine_tuned"] is True and cfg["max_len"] == 128 and cfg["head_max_len"] == 64
    assert cfg["amp_dtype"] == "bf16"
    assert len(cfg["temperature"]) == 3 and isinstance(cfg["temperature_by_options"], dict)
    assert meta["cases"] == {"train": 18, "heldout": 6} and meta["skipped_questions"]["count"] == 0
    assert meta["calibration"]["n_records"] == 18 and len(meta["epochs"]) == 2
    assert _read(str(out / "train_meta.json"))["optimizer_steps"] == meta["optimizer_steps"] > 0
    with safe_open(str(out / "model.safetensors"), "pt") as f:
        floats = {f.get_tensor(k).dtype for k in f.keys() if f.get_tensor(k).is_floating_point()}
    assert floats == {torch.float16}
    agent = Agent(str(out), device="cpu")
    rec = toy_records()[0]
    assert set(agent.predict(rec["state"], rec["questions"])["answers"]) == {"sentiment", "happy", "stars"}


def test_calibration_only_uses_the_held_out_questions_training_kept(tiny_base, tmp_path):
    """A question record_to_items skips (options too long for head_max_len) must not reach the
    runtime during calibration: Agent._build_items raises for exactly the questions that were
    dropped everywhere else, so `_calibrate` has to drop them from the held-out records too."""
    from tests.conftest import TINY_WORDS
    recs = toy_records()
    long_criteria = {w: "the a is or not statement" for w in TINY_WORDS}   # too long for head_max_len=64
    for r in recs:
        r["questions"]["toolong"] = {"type": "choice", "instructions": "pick one", "criteria": long_criteria}
        r["targets"]["toolong"] = {"probabilities": {TINY_WORDS[0]: 1.0}}
    out = tmp_path / "run"
    meta = train(tiny_base, recs, str(out), TrainConfig(**dict(FAST, max_skipped_fraction=0.3)), device="cpu")
    assert (out / "train_meta.json").exists()
    assert meta["skipped_questions"]["count"] > 0
    assert meta["calibration"] is not None


def test_calibrate_leaves_out_skipped_questions_and_returns_none_when_nothing_is_left(tiny_base):
    from peewee_decide.train import _calibrate
    held = toy_records(2)
    first = ["%s/%s" % (held[0]["id"], q) for q in held[0]["questions"]]
    assert _calibrate(tiny_base, held, torch.device("cpu"), skipped=first)["n_records"] == 3
    everything = first + ["%s/%s" % (held[1]["id"], q) for q in held[1]["questions"]]
    assert _calibrate(tiny_base, held, torch.device("cpu"), skipped=everything) is None


def test_training_lowers_the_soft_cross_entropy(tiny_base, tmp_path):
    meta = train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**dict(FAST, epochs=4)), device="cpu")
    ce = [e["train_ce"] for e in meta["epochs"]]
    assert ce[-1] < ce[0]


def test_freeze_encoder_leaves_the_encoder_untouched(tiny_base, tmp_path):
    from safetensors.torch import load_file
    train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**dict(FAST, freeze_encoder=True)),
          device="cpu")
    before = load_file(tiny_base + "/model.safetensors")
    after = load_file(str(tmp_path / "run" / "model.safetensors"))
    enc = [k for k in before if k.startswith("encoder.")]
    assert enc and all(torch.equal(before[k].half(), after[k]) for k in enc)
    assert any(not torch.equal(before[k].half(), after[k]) for k in before if k.startswith("scorer."))


def test_an_existing_checkpoint_is_not_overwritten(tiny_base, tmp_path):
    (tmp_path / "rl_agent_config.json").write_text("{}")
    with pytest.raises(FileExistsError, match="overwrite"):
        train(tiny_base, toy_records(), str(tmp_path), TrainConfig(**FAST), device="cpu")


def test_too_many_skipped_questions_abort(tiny_base, tmp_path):
    with pytest.raises(ValueError, match="do not fit"):
        train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**dict(FAST, max_len=10, head_max_len=8)),
              device="cpu")


def test_invalid_records_fail_before_any_work(tiny_base, tmp_path):
    recs = toy_records()
    recs[3]["targets"]["stars"] = {"label": 9}
    with pytest.raises(ValueError, match="case-003/stars"):
        train(tiny_base, recs, str(tmp_path / "run"), TrainConfig(**FAST), device="cpu")
    assert not (tmp_path / "run").exists()


def test_non_finite_loss_aborts_without_saving(tiny_base, tmp_path, monkeypatch):
    import peewee_decide.train as lt

    def nan_loss(*args, **kwargs):
        return torch.tensor(float("nan"), requires_grad=True), {"reward": 0.0, "ce": 0.0, "rl": 0.0}

    monkeypatch.setattr(lt, "rlcd_loss", nan_loss)
    with pytest.raises(RuntimeError, match="non-finite loss"):
        train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**FAST), device="cpu")
    assert not (tmp_path / "run" / "model.safetensors").exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_cuda_without_cuda_is_an_error():
    with pytest.raises(RuntimeError, match="CUDA requested"):
        pick_device("cuda")


def test_train_cli(tiny_base, tmp_path):
    from peewee_decide.data import write_jsonl
    data = str(tmp_path / "train.jsonl")
    write_jsonl(data, toy_records())
    out = str(tmp_path / "run")
    rc = train_main(["--data", data, "--base", tiny_base, "--out", out, "--device", "cpu", "--epochs", "1",
                     "--micro-batch", "8", "--grad-accum", "1", "--no-gradient-checkpointing",
                     "--calib-fraction", "0.25"])
    assert rc == 0 and _read(out + "/train_meta.json")["config"]["epochs"] == 1
    with pytest.raises(SystemExit) as e:
        train_main(["--help"])
    assert e.value.code == 0


def test_train_cli_rejects_an_unknown_precision(tiny_base, tmp_path):
    from peewee_decide.data import write_jsonl
    data = str(tmp_path / "train.jsonl")
    write_jsonl(data, toy_records())
    with pytest.raises(SystemExit) as e:
        train_main(["--data", data, "--base", tiny_base, "--out", str(tmp_path / "run"), "--device", "cpu",
                    "--precision", "fp8"])
    assert e.value.code == 2


def test_git_commit_is_read_without_a_subprocess():
    import os
    import subprocess

    from peewee_decide.train import _git_commit
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(root, ".git")):
        pytest.skip("not a git checkout")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    assert _git_commit() == head


def _capture_calibration_targets(monkeypatch):
    import peewee_decide.calibrate as lc
    seen = []
    real = lc.fit_temperature_map

    def spy(records, *args, **kwargs):
        records = list(records)
        seen.extend(t for _qt, _z, t, _k in records)
        return real(records, *args, **kwargs)

    monkeypatch.setattr(lc, "fit_temperature_map", spy)
    return seen


@pytest.mark.parametrize("target, one_hot", [("probabilities", False), ("label", True)])
def test_calibration_target_picks_distribution_or_reference_answer(tiny_base, tmp_path, monkeypatch, target, one_hot):
    seen = _capture_calibration_targets(monkeypatch)
    meta = train(tiny_base, toy_records(), str(tmp_path / "run"),
                 TrainConfig(**dict(FAST, epochs=1, calibration_target=target)), device="cpu")
    assert seen and meta["config"]["calibration_target"] == target
    assert all(sorted(t.tolist()) == [0.0] * (len(t) - 1) + [1.0] for t in seen) is one_hot


def test_unknown_calibration_target_fails_before_any_work(tiny_base, tmp_path):
    with pytest.raises(ValueError, match="calibration_target"):
        train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**dict(FAST, calibration_target="soft")),
              device="cpu")
    assert not (tmp_path / "run").exists()


def test_calib_records_replace_the_held_out_split(tiny_base, tmp_path):
    calib = toy_records(6, seed=1)
    for r in calib:
        r["id"] = "calib-" + r["id"]
    meta = train(tiny_base, toy_records(), str(tmp_path / "run"), TrainConfig(**dict(FAST, epochs=1)), device="cpu",
                 calib_records=calib)
    assert meta["cases"] == {"train": 24, "heldout": 6}
    assert meta["calibration"]["n_records"] == 18


def test_calib_records_that_overlap_training_are_rejected(tiny_base, tmp_path):
    recs = toy_records()
    with pytest.raises(ValueError, match="case-000"):
        train(tiny_base, recs, str(tmp_path / "run"), TrainConfig(**FAST), device="cpu", calib_records=recs[:2])
    grouped = toy_records(4, seed=1)
    for i, r in enumerate(grouped):
        r["id"], r["meta"]["group"] = "calib-%d" % i, "case-001"
    with pytest.raises(ValueError, match="case-001"):
        train(tiny_base, recs, str(tmp_path / "run"), TrainConfig(**FAST), device="cpu", calib_records=grouped)


def test_repeat_upsamples_training_cases_only(tiny_base, tmp_path):
    from peewee_decide.data import split_cases
    recs = toy_records()
    repeat = {r["id"]: 3 for r in recs[:8]}
    meta = train(tiny_base, recs, str(tmp_path / "run"), TrainConfig(**dict(FAST, epochs=1)), device="cpu",
                 repeat=repeat)
    train_recs, held = split_cases(recs, FAST["calib_fraction"], 0)
    assert meta["items"]["train_unique"] == len(train_recs) * 3
    assert meta["items"]["train"] == sum(3 * repeat.get(r["id"], 1) for r in train_recs)
    assert meta["items"]["heldout"] == len(held) * 3


def test_train_cli_mixes_files_and_takes_a_calibration_file(tiny_base, tmp_path):
    from peewee_decide.data import write_jsonl
    a, b, c = (str(tmp_path / n) for n in ("a.jsonl", "b.jsonl", "c.jsonl"))
    recs = toy_records()
    calib = toy_records(4, seed=2)
    for r in calib:
        r["id"] = "calib-" + r["id"]
    write_jsonl(a, recs[:8])
    write_jsonl(b, recs[8:])
    write_jsonl(c, calib)
    out = str(tmp_path / "run")
    rc = train_main(["--data", a + ":4", "--data", b, "--calib-data", c, "--calibration-target", "label",
                     "--base", tiny_base, "--out", out, "--device", "cpu", "--epochs", "1", "--micro-batch", "8",
                     "--grad-accum", "1", "--no-gradient-checkpointing"])
    meta = _read(out + "/train_meta.json")
    assert rc == 0 and meta["data"] == [a, b] and len(meta["data_sha256"]) == 2
    assert meta["calib_data"] == c and meta["cases"] == {"train": 24, "heldout": 4}
    assert meta["items"] == {"train": 24 * 3 + 8 * 3 * 3, "train_unique": 72, "heldout": 12}
    assert meta["config"]["calibration_target"] == "label"


@pytest.mark.parametrize("argv", [["--calibration-target", "soft"], ["--data", "x.jsonl:0"]])
def test_train_cli_rejects_bad_mix_arguments(tiny_base, tmp_path, argv):
    from peewee_decide.data import write_jsonl
    data = str(tmp_path / "train.jsonl")
    write_jsonl(data, toy_records())
    with pytest.raises(SystemExit) as e:
        train_main(["--data", data, "--base", tiny_base, "--out", str(tmp_path / "run"), "--device", "cpu"] + argv)
    assert e.value.code == 2
