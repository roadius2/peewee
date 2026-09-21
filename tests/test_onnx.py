"""ONNX export and ONNX Runtime parity on a tiny random decision model: no weights, no network."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel  # noqa: E402
from tests.conftest import FakeTokenizer  # noqa: E402

VOCAB = 100


class TinyTokenizer(FakeTokenizer):
    @staticmethod
    def word_id(word):
        return 10 + sum(ord(c) for c in word) % (VOCAB - 10)


def tiny_agent(tmp_path):
    from transformers import BertConfig, BertModel
    torch.manual_seed(0)
    enc = BertModel(BertConfig(vocab_size=VOCAB, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, max_position_embeddings=128), add_pooling_layer=False)
    model = DecisionModel(enc, head_layers=1, n_act=2).eval()
    a = Agent.__new__(Agent)
    a.tok = TinyTokenizer()
    a.cfg = {"encoder": "tiny", "head_layers": 1, "max_len": 96, "head_max_len": 40}
    a.model = model
    a.device = torch.device("cpu")
    a.dtype = torch.float32
    a.model_dir = None
    a.temperature = [1.0, 1.5, 1.0]
    a.temperature_by_options = {"choice:2": 2.0}
    a.calibration_source = "checkpoint"
    a.truncate = None
    a._warned = set()
    return a


QS = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": "bugs", "ops": None}},
      "urg": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]},
      "angry": {"type": "noul", "instructions": "Angry?"}}


def test_export_and_parity(tmp_path):
    from laya.onnx_backend import OnnxAgent, export_onnx
    a = tiny_agent(tmp_path)
    out = str(tmp_path / "export")
    meta = export_onnx(a, out, quantize=False)
    assert meta["quantized"] is False and os.path.exists(os.path.join(out, "model.onnx"))
    assert json.load(open(os.path.join(out, "rl_agent_config.json")))["max_len"] == 96
    cal = json.load(open(os.path.join(out, "calibration.json")))
    assert cal["temperature"] == [1.0, 1.5, 1.0] and cal["temperature_by_options"] == {"choice:2": 2.0}

    o = OnnxAgent(out, tokenizer=TinyTokenizer())
    assert o.quantized is False and o.calibration_source == "export"
    assert o.temperature_by_options == {"choice:2": 2.0}

    for state in ["hello world", {"body": "charged twice again and again"}, [{"role": "u", "content": "x y z"}]]:
        ref = a.system_one(state, QS)
        got = o.system_one(state, QS)
        for qid in QS:
            for key in ("probabilities",):
                if key in ref["answers"][qid]:
                    for k, v in ref["answers"][qid][key].items():
                        assert abs(got["answers"][qid][key][k] - v) < 2e-3, (qid, k)
            assert abs(got["answers"][qid]["confidence"] - ref["answers"][qid]["confidence"]) < 2e-3
        assert got["usage"]["input_tokens"] == ref["usage"]["input_tokens"]

    # batch path and different sequence lengths / option counts in one session
    many = o.predict_many([("a", QS), ("b c d e f g h", {"angry": QS["angry"]})])
    assert [set(m["answers"]) for m in many] == [set(QS), {"angry"}]


def test_export_quantized(tmp_path):
    from laya.onnx_backend import OnnxAgent, export_onnx
    a = tiny_agent(tmp_path)
    out = str(tmp_path / "export-q")
    meta = export_onnx(a, out, quantize=True)
    assert meta["quantized"] is True and os.path.exists(os.path.join(out, "model.int8.onnx"))
    o = OnnxAgent(out, tokenizer=TinyTokenizer())
    assert o.quantized is True
    res = o.system_one("hello there", QS)
    p = res["answers"]["dept"]["probabilities"]
    assert abs(sum(p.values()) - 1.0) < 1e-3
    full = OnnxAgent(out, tokenizer=TinyTokenizer(), prefer_quantized=False)
    assert full.quantized is False
    # weight-only int8: encoder weights are MatMulNBits, activations and the head stay fp32,
    # so probabilities track the fp32 graph closely (dynamic int8 broke the MLP on real weights)
    import onnx
    ops = {n.op_type for n in onnx.load(o.onnx_path).graph.node}
    assert "MatMulNBits" in ops and "DynamicQuantizeLinear" not in ops
    assert meta["quantization"].startswith("weight-only int8")
    ref = full.system_one("hello there", QS)["answers"]
    for qid in QS:
        if "probabilities" in ref[qid]:
            for k, v in ref[qid]["probabilities"].items():
                assert abs(v - res["answers"][qid]["probabilities"][k]) < 0.05


def test_missing_export_dir_raises(tmp_path):
    from laya.onnx_backend import OnnxAgent
    (tmp_path / "rl_agent_config.json").write_text('{"encoder": "x", "head_layers": 1}')
    with pytest.raises(FileNotFoundError, match="export-onnx"):
        OnnxAgent(str(tmp_path), tokenizer=TinyTokenizer())


def _accelerator():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_export_restores_model_dtype(tmp_path):
    """Export needs an fp32 CPU copy; the caller's agent must come back unchanged."""
    from laya.onnx_backend import export_onnx
    a = tiny_agent(tmp_path)
    a.model = a.model.to(torch.bfloat16)
    a.dtype = torch.bfloat16
    export_onnx(a, str(tmp_path / "export"), quantize=False)
    p = next(a.model.parameters())
    assert p.dtype == torch.bfloat16 and p.device == a.device


@pytest.mark.skipif(_accelerator() is None, reason="needs a CUDA or MPS device")
def test_export_restores_model_device(tmp_path):
    from laya.onnx_backend import export_onnx
    a = tiny_agent(tmp_path)
    a.device = _accelerator()
    a.model = a.model.to(a.device)
    export_onnx(a, str(tmp_path / "export"), quantize=False)
    assert next(a.model.parameters()).device.type == a.device.type
    assert a.predict("the customer was charged twice", QS)["answers"]["dept"]["choice"] in QS["dept"]["criteria"]


def _exported(tmp_path):
    from laya.onnx_backend import export_onnx
    out = str(tmp_path / "export")
    export_onnx(tiny_agent(tmp_path), out, quantize=False)
    return out


def test_requested_provider_missing_raises(tmp_path, monkeypatch):
    """An explicit provider that the runtime lacks is a startup error, not a silent CPU run."""
    import onnxruntime as ort
    from laya.onnx_backend import OnnxAgent
    out = _exported(tmp_path)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        OnnxAgent(out, tokenizer=TinyTokenizer(), providers=["CUDAExecutionProvider"])


def test_auto_provider_warns_when_cuda_is_only_missing_from_onnxruntime(tmp_path, monkeypatch, caplog):
    import logging
    import onnxruntime as ort
    from laya.onnx_backend import OnnxAgent
    out = _exported(tmp_path)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with caplog.at_level(logging.WARNING, logger="laya.onnx"):
        o = OnnxAgent(out, tokenizer=TinyTokenizer())
    assert o.providers == ["CPUExecutionProvider"]
    assert any("onnxruntime-gpu" in r.getMessage() for r in caplog.records)


def test_providers_for_device():
    from laya.onnx_backend import providers_for_device
    assert providers_for_device("cuda") == ["CUDAExecutionProvider"]
    assert providers_for_device("cuda:0") == ["CUDAExecutionProvider"]
    assert providers_for_device("cpu") == ["CPUExecutionProvider"]
    assert providers_for_device(None) is None
