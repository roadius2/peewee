"""`Agent` pipeline pieces on synthetic logits: no weights, no download."""
import json
import logging
import os

import numpy as np
import pytest

from laya.agent import Agent, _check_truncate, patch_tokenizer_config
from laya.common import normalized_entropy

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "churn": {"type": "noul", "instructions": "Will they leave?"},
}
LOGITS = np.array([[2.0, 0.0, -1e4], [0.0, 0.0, 3.0], [0.0, 2.0, -1e4]], dtype=np.float32)
ACT = np.tile([0.25, 0.75], (3, 1)).astype(np.float32)


def bare_agent(fake_tok, truncate=None, **cfg):
    """An Agent with no model: enough state for tokenisation and post-processing."""
    a = Agent.__new__(Agent)
    a.tok = fake_tok
    a.cfg = {"max_len": 512, "head_max_len": 192, **cfg}
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.calibration_source = "checkpoint"
    a.truncate = truncate
    a._warned = set()
    return a


def fake_forward(items, scale=1.0):
    n = len(items)
    kmax = max(len(it["markers"]) for it in items)
    logits = np.full((n, kmax), -1e4, dtype=np.float32)
    for r, it in enumerate(items):
        k = len(it["markers"])
        logits[r, :k] = scale * np.arange(k, dtype=np.float32)
    return logits, np.tile([0.25, 0.75], (n, 1)).astype(np.float32), sum(len(it["ids"]) for it in items)


# ------------------------------------------------------------------ build items
def test_build_items_shapes(fake_tok):
    items = bare_agent(fake_tok)._build_items({"body": "hello"}, QUESTIONS)
    assert [len(it["markers"]) for it in items] == [2, 3, 2]
    assert [it["qtype"] for it in items] == [0, 1, 2]
    assert all("info" in it for it in items)


def test_build_items_raises_when_options_do_not_fit(fake_tok):
    q = {"big": {"type": "choice", "instructions": "x",
                 "criteria": {"opt%d" % i: "one two three four five six" for i in range(77)}}}
    with pytest.raises(ValueError, match="exceed head_max_len"):
        bare_agent(fake_tok, max_len=128)._build_items("s", q)


def test_truncation_side_defaults(fake_tok):
    a = bare_agent(fake_tok)
    long_words = " ".join("w%d" % i for i in range(1000))
    assert a._build_items(long_words, QUESTIONS)[0]["info"]["truncation"] == "right"
    assert a._build_items({"body": long_words}, QUESTIONS)[0]["info"]["truncation"] == "right"
    assert a._build_items([{"role": "user", "content": long_words}], QUESTIONS)[0]["info"]["truncation"] == "left"
    assert a._build_items(long_words, QUESTIONS, truncate="left")[0]["info"]["truncation"] == "left"
    assert bare_agent(fake_tok, truncate="left")._build_items(long_words, QUESTIONS)[0]["info"]["truncation"] == "left"
    # an explicit call-time side beats the agent default
    assert bare_agent(fake_tok, truncate="left")._build_items(long_words, QUESTIONS, truncate="right")[0]["info"]["truncation"] == "right"


def test_check_truncate():
    assert _check_truncate(None) is None
    assert _check_truncate("LEFT") == "left"
    with pytest.raises(ValueError):
        _check_truncate("middle")


# ------------------------------------------------------------------ post-processing
def test_postprocess_answers(fake_tok):
    a = bare_agent(fake_tok)
    items = a._build_items("hello", QUESTIONS)
    out = a._postprocess(QUESTIONS, items, LOGITS, ACT, n_tokens=42)

    dept = out["answers"]["dept"]
    assert dept["type"] == "choice" and dept["choice"] == "billing"
    assert abs(sum(dept["probabilities"].values()) - 1.0) < 1e-3
    assert dept["probabilities"]["billing"] == pytest.approx(0.8808, abs=1e-3)
    # confidence is the probability of the reported answer, for every type
    assert dept["confidence"] == pytest.approx(0.8808, abs=1e-3)
    assert dept["entropy"] == pytest.approx(normalized_entropy(np.array([0.8808, 0.1192]), 2), abs=1e-3)

    urg = out["answers"]["urgency"]
    p = np.exp(LOGITS[1]); p /= p.sum()
    assert urg["score"] == pytest.approx(float((np.arange(3) * p).sum()), abs=1e-3)
    assert urg["level"] == 2
    assert urg["confidence"] == pytest.approx(float(p.max()), abs=1e-3)
    assert urg["legend"] == {"0": "low", "1": "mid", "2": "high"}

    churn = out["answers"]["churn"]
    assert churn["noul"] == pytest.approx(0.8808, abs=1e-3)
    assert churn["confidence"] == pytest.approx(0.8808, abs=1e-3)
    assert dept["action"] == {"act_probability": 0.25}

    usage = out["usage"]
    assert usage["input_tokens"] == 42 and usage["output_tokens"] == 0
    assert usage["truncated"] is False and usage["state_tokens_dropped"] == 0
    assert usage["state_tokens"] == 1
    assert "option_budget" not in usage


def test_confidence_is_comparable_across_types(fake_tok):
    """A 90/10 two-way choice and a 90% noul must report the same confidence."""
    a = bare_agent(fake_tok)
    qs = {"c": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}},
          "n": {"type": "noul", "instructions": "x"}}
    items = a._build_items("s", qs)
    z = float(np.log(9.0))
    logits = np.array([[z, 0.0], [0.0, z]], dtype=np.float32)
    out = a._postprocess(qs, items, logits, np.zeros((2, 2), dtype=np.float32), 1)["answers"]
    assert out["c"]["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert out["n"]["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert out["c"]["entropy"] == pytest.approx(out["n"]["entropy"], abs=1e-4)


def test_temperature_bucket_is_applied(fake_tok):
    a = bare_agent(fake_tok)
    items = a._build_items("hello", QUESTIONS)
    act = np.zeros((3, 2), dtype=np.float32)
    hot = a._postprocess(QUESTIONS, items, LOGITS, act, 1)["answers"]["dept"]["probabilities"]["billing"]
    a.temperature_by_options = {"choice:2": 4.0}
    cool = a._postprocess(QUESTIONS, items, LOGITS, act, 1)["answers"]["dept"]["probabilities"]["billing"]
    assert cool < hot
    assert cool == pytest.approx(1 / (1 + np.exp(-0.5)), abs=1e-3)


def test_truncation_is_reported_and_warned_once(fake_tok, caplog):
    a = bare_agent(fake_tok)
    state = " ".join("w%d" % i for i in range(3000))
    with caplog.at_level(logging.DEBUG, logger="laya"):
        items = a._build_items(state, QUESTIONS)
        out = a._postprocess(QUESTIONS, items, LOGITS, ACT, 1)
        out2 = a._postprocess(QUESTIONS, items, LOGITS, ACT, 1)
    u = out["usage"]
    assert u["truncated"] is True and u["state_tokens"] == 3000 and u["state_tokens_dropped"] > 2000
    assert u["truncation"] == "right"
    assert out2["usage"]["truncated"] is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "truncated" in r.getMessage()]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "truncated" in r.getMessage()]
    assert len(warnings) == 1 and len(debugs) == 1


def test_option_budget_is_reported(fake_tok, caplog):
    a = bare_agent(fake_tok)
    qs = {"many": {"type": "choice", "instructions": "x",
                   "criteria": {"opt%d" % i: "one two three four five six" for i in range(77)}},
          "long": {"type": "choice", "instructions": "x",
                   "criteria": {"a": " ".join(["z"] * 100), "b": None}},
          "fine": {"type": "noul", "instructions": "x"}}
    items = a._build_items("s", qs)
    n = len(items)
    logits = np.full((n, 77), -1e4, dtype=np.float32)
    logits[:, :2] = 0
    with caplog.at_level(logging.WARNING, logger="laya"):
        out = a._postprocess(qs, items, logits, np.zeros((n, 2), dtype=np.float32), 1)
    budget = out["usage"]["option_budget"]
    assert set(budget) == {"many", "long"}
    assert budget["many"]["options_squeezed"] is True and budget["many"]["tokens_per_option"] == 3
    assert budget["long"]["options_over_cap"] == 1 and budget["long"]["options_squeezed"] is False
    assert any("hierarchical_choice" in r.getMessage() for r in caplog.records)


def test_system_one_composes(fake_tok, monkeypatch):
    a = bare_agent(fake_tok)
    monkeypatch.setattr(a, "_forward_logits", fake_forward)
    out = a.system_one("hello", QUESTIONS)
    assert set(out["answers"]) == set(QUESTIONS)
    assert out["answers"]["dept"]["choice"] == "tech"
    assert out["usage"]["input_tokens"] > 0


# ------------------------------------------------------------------ batch
def test_predict_many_matches_single(fake_tok, monkeypatch):
    a = bare_agent(fake_tok)
    monkeypatch.setattr(a, "_forward_logits", fake_forward)
    reqs = [("hello", QUESTIONS), {"state": {"body": "again"}, "questions": {"churn": QUESTIONS["churn"]}},
            ("third", {}), ([{"role": "user", "content": "x"}], QUESTIONS)]
    many = a.predict_many(reqs, max_batch=2)
    assert len(many) == 4
    singles = [a.system_one("hello", QUESTIONS), a.system_one({"body": "again"}, {"churn": QUESTIONS["churn"]})]
    for m, s in zip(many, singles):
        assert m["answers"] == s["answers"]
        assert m["usage"]["input_tokens"] == s["usage"]["input_tokens"]
    assert many[2]["answers"] == {}
    assert many[3]["usage"]["truncation"] == "left"


def test_predict_many_batches_calls(fake_tok, monkeypatch):
    a = bare_agent(fake_tok)
    calls = []

    def counting(items):
        calls.append(len(items))
        return fake_forward(items)

    monkeypatch.setattr(a, "_forward_logits", counting)
    a.predict_many([("s%d" % i, QUESTIONS) for i in range(5)], max_batch=8)
    assert calls == [8, 7]


# ------------------------------------------------------------------ calibration on the agent
def test_fit_save_load_calibration(fake_tok, tmp_path):
    a = bare_agent(fake_tok)
    a.cfg["encoder"] = "enc"
    rng = np.random.default_rng(0)
    records = []
    for _ in range(60):
        y = rng.integers(0, 2)
        z = np.zeros(2, dtype=np.float32)
        z[y] = 6.0 if rng.random() < 0.7 else 0.0
        z[1 - y] = 6.0 if rng.random() < 0.3 else 0.0
        records.append((2, z, np.eye(2, dtype=np.float32)[y], 2))
    result = a.fit_temperatures(records)
    assert a.calibration_source == "fitted"
    assert result["temperature"][2] > 1.0                       # over-confident inputs get cooled
    assert "noul:2" in result["temperature_by_options"]
    assert result["report"]["all"]["after"]["nll"] <= result["report"]["all"]["before"]["nll"]

    path = tmp_path / "cal.json"
    a.save_calibration(str(path), meta={"note": "test"})
    payload = json.loads(path.read_text())
    assert payload["meta"] == {"note": "test", "encoder": "enc"}

    b = bare_agent(fake_tok)
    b.load_calibration(str(path))
    assert b.temperature == a.temperature and b.temperature_by_options == a.temperature_by_options
    assert b.calibration_source == str(path)


# ------------------------------------------------------------------ tokenizer config patching
def test_patch_tokenizer_config():
    cfg = {"tokenizer_class": "TokenizersBackend", "backend": "x", "extra_special_tokens": ["<a>", "<b>"]}
    assert patch_tokenizer_config(cfg) is True
    assert cfg["tokenizer_class"] == "PreTrainedTokenizerFast" and "backend" not in cfg
    assert cfg["extra_special_tokens"] == {"extra_0": "<a>", "extra_1": "<b>"}
    ok = {"tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {"x": "<x>"}}
    assert patch_tokenizer_config(ok) is False


def test_tokenizer_dir_never_writes_into_the_checkpoint(tmp_path):
    from laya.agent import _tokenizer_dir
    tok = tmp_path / "tokenizer"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    (tok / "tokenizer_config.json").write_text('{"tokenizer_class": "TokenizersBackend"}')
    before = (tok / "tokenizer_config.json").read_text()
    out = _tokenizer_dir(str(tmp_path))
    assert out != str(tok)
    assert (tok / "tokenizer_config.json").read_text() == before
    assert json.load(open(os.path.join(out, "tokenizer_config.json")))["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert os.path.exists(os.path.join(out, "tokenizer.json"))
    (tok / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')
    assert _tokenizer_dir(str(tmp_path)) == str(tok)
    assert _tokenizer_dir(str(tmp_path / "missing")) is None
