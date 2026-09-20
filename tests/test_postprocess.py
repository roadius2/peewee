"""`Agent._postprocess` and `_build_items` on synthetic logits: no weights, no download."""
import numpy as np
import pytest

from laya.agent import Agent, patch_tokenizer_config
from laya.common import confidence_from_probs

QUESTIONS = {
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "churn": {"type": "noul", "instructions": "Will they leave?"},
}


def bare_agent(fake_tok, **cfg):
    """An Agent with no model: enough state for tokenisation and post-processing."""
    a = Agent.__new__(Agent)
    a.tok = fake_tok
    a.cfg = {"max_len": 512, "head_max_len": 192, **cfg}
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    return a


def test_build_items_shapes(fake_tok):
    items = bare_agent(fake_tok)._build_items({"body": "hello"}, QUESTIONS)
    assert [len(it["markers"]) for it in items] == [2, 3, 2]
    assert [it["qtype"] for it in items] == [0, 1, 2]


def test_build_items_raises_when_options_do_not_fit(fake_tok):
    q = {"big": {"type": "choice", "instructions": "x",
                 "criteria": {"opt%d" % i: "one two three four five six" for i in range(77)}}}
    with pytest.raises(ValueError, match="exceed head_max_len"):
        bare_agent(fake_tok, max_len=128)._build_items("s", q)


def test_postprocess_answers(fake_tok):
    a = bare_agent(fake_tok)
    items = a._build_items("hello", QUESTIONS)
    logits = np.array([
        [2.0, 0.0, -1e4],
        [0.0, 0.0, 3.0],
        [0.0, 2.0, -1e4],
    ], dtype=np.float32)
    act = np.tile([0.25, 0.75], (3, 1)).astype(np.float32)
    out = a._postprocess(QUESTIONS, items, logits, act, n_tokens=42)

    dept = out["answers"]["dept"]
    assert dept["type"] == "choice" and dept["choice"] == "billing"
    assert abs(sum(dept["probabilities"].values()) - 1.0) < 1e-3
    assert dept["probabilities"]["billing"] == pytest.approx(0.8808, abs=1e-3)
    assert dept["confidence"] == pytest.approx(confidence_from_probs(np.array([0.8808, 0.1192]), 2), abs=1e-3)

    urg = out["answers"]["urgency"]
    p = np.exp(logits[1]); p /= p.sum()
    assert urg["score"] == pytest.approx(float((np.arange(3) * p).sum()), abs=1e-3)
    assert urg["legend"] == {"0": "low", "1": "mid", "2": "high"}

    churn = out["answers"]["churn"]
    assert churn["noul"] == pytest.approx(0.8808, abs=1e-3)
    assert churn["confidence"] == pytest.approx(0.8808, abs=1e-3)

    assert out["usage"] == {"input_tokens": 42, "output_tokens": 0}
    assert dept["action"] == {"act_probability": 0.25}


def test_temperature_bucket_is_applied(fake_tok):
    a = bare_agent(fake_tok)
    items = a._build_items("hello", QUESTIONS)
    logits = np.array([[2.0, 0.0, -1e4], [0.0, 0.0, 3.0], [0.0, 2.0, -1e4]], dtype=np.float32)
    act = np.zeros((3, 2), dtype=np.float32)
    hot = a._postprocess(QUESTIONS, items, logits, act, 1)["answers"]["dept"]["probabilities"]["billing"]
    a.temperature_by_options = {"choice:2": 4.0}
    cool = a._postprocess(QUESTIONS, items, logits, act, 1)["answers"]["dept"]["probabilities"]["billing"]
    assert cool < hot
    assert cool == pytest.approx(1 / (1 + np.exp(-0.5)), abs=1e-3)


def test_system_one_composes(fake_tok, monkeypatch):
    a = bare_agent(fake_tok)
    seen = {}

    def fake_forward(items):
        seen["n"] = len(items)
        return np.zeros((3, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.float32), 7

    monkeypatch.setattr(a, "_forward_logits", fake_forward)
    out = a.system_one("hello", QUESTIONS)
    assert seen["n"] == 3 and set(out["answers"]) == set(QUESTIONS)
    assert out["answers"]["dept"]["probabilities"] == {"billing": 0.5, "tech": 0.5}


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
    import json, os
    assert json.load(open(os.path.join(out, "tokenizer_config.json")))["tokenizer_class"] == "PreTrainedTokenizerFast"
    assert os.path.exists(os.path.join(out, "tokenizer.json"))

    # a config that needs no patching is loaded in place
    (tok / "tokenizer_config.json").write_text('{"tokenizer_class": "PreTrainedTokenizerFast"}')
    assert _tokenizer_dir(str(tmp_path)) == str(tok)
    assert _tokenizer_dir(str(tmp_path / "missing")) is None
