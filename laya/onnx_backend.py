"""ONNX export and an ONNX Runtime backend for CPU (and CUDA) serving.

    laya export-onnx convaiinnovations/laya ./laya-english-onnx --quantize
    python -c "from laya.onnx_backend import OnnxAgent; a = OnnxAgent('./laya-english-onnx'); print(a.predict(...))"
    LAYA_BACKEND=onnx LAYA_MODELS=english=./laya-english-onnx laya serve

The decision model is an encoder plus a small transformer head and two MLPs, all standard ops,
so it exports cleanly. Dynamic int8 quantisation of the linear layers roughly halves CPU
latency at a small accuracy cost; measure both on your own labelled data before choosing.

An export directory holds `model.onnx` (and `model.int8.onnx` when quantised), the checkpoint's
`rl_agent_config.json`, its `tokenizer/`, a `calibration.json` with the temperatures that were
active at export time, and `onnx_export.json` describing the export.
"""
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent import Agent, _tokenizer_dir, resolve_checkpoint
from .calibrate import calibration_payload
from .common import collate_items

logger = logging.getLogger("laya.onnx")

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]


# ------------------------------------------------------------------------------ export
def export_onnx(agent: Agent, out_dir: str, quantize: bool = False, opset: int = 17,
                sample_items: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Export `agent.model` to `out_dir/model.onnx` with dynamic batch, sequence and option axes.

    Everything the runtime needs besides weights (config, tokenizer, temperatures) is copied
    next to it so `OnnxAgent(out_dir)` is self-contained.
    """
    import torch

    if quantize:
        _weight_only_quantizer()                  # fail before the (slow) fp32 export, not after it
    os.makedirs(out_dir, exist_ok=True)
    first = next(agent.model.parameters())
    orig_device, orig_dtype = first.device, first.dtype
    model = agent.model.eval().float().cpu()      # the tracer wants fp32 on CPU; restored below
    if sample_items is None:
        sample_items = agent._build_items(
            "The customer was charged twice and wants a refund.",
            {"q1": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None, "other": None}},
             "q2": {"type": "noul", "instructions": "Is the user angry?"}})
    b = collate_items([sample_items], agent.tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "marker_pos": {0: "batch", 1: "options"},
        "marker_mask": {0: "batch", 1: "options"},
        "qtype": {0: "batch"},
        "logits": {0: "batch", 1: "options"},
        "act_logits": {0: "batch"},
    }
    onnx_path = os.path.join(out_dir, "model.onnx")
    t0 = time.perf_counter()
    kw: Dict[str, Any] = dict(input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_axes=dynamic_axes,
                              opset_version=opset, do_constant_folding=True)
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False          # the TorchScript exporter handles the padding-mask head reliably
    was_manual = getattr(model, "manual_head_attention", False)
    model.manual_head_attention = True        # shape-dynamic head attention for the tracer
    try:
        with torch.no_grad():
            torch.onnx.export(model, args, onnx_path, **kw)
    finally:
        model.manual_head_attention = was_manual
        model.to(device=orig_device, dtype=orig_dtype)   # leave the caller's agent usable
    export_seconds = time.perf_counter() - t0
    logger.info("laya.onnx: exported %s in %.1fs", onnx_path, export_seconds)

    # runtime companions
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(agent.cfg, f, indent=2)
    src_tok = _tokenizer_dir(getattr(agent, "model_dir", "")) if getattr(agent, "model_dir", None) else None
    if src_tok and os.path.isdir(src_tok):
        dst = os.path.join(out_dir, "tokenizer")
        if os.path.abspath(src_tok) != os.path.abspath(dst):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src_tok, dst)
    elif hasattr(agent.tok, "save_pretrained"):
        agent.tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump(calibration_payload(agent.temperature, agent.temperature_by_options,
                                      {"source": agent.calibration_source}), f, indent=2)

    meta = {"opset": opset, "quantized": False, "export_seconds": round(export_seconds, 2),
            "encoder": agent.cfg.get("encoder"), "torch": torch.__version__,
            "inputs": INPUT_NAMES, "outputs": OUTPUT_NAMES}
    if quantize:
        q_path = os.path.join(out_dir, "model.int8.onnx")
        t0 = time.perf_counter()
        _quantize_weights_int8(onnx_path, q_path)
        meta.update({"quantized": True, "quantize_seconds": round(time.perf_counter() - t0, 2),
                     "quantization": "weight-only int8 (MatMulNBits, block 128, symmetric); encoder MatMul weights "
                                     "only, decision head and attention products fp32"})
        logger.info("laya.onnx: wrote %s", q_path)
    with open(os.path.join(out_dir, "onnx_export.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _weight_only_quantizer():
    """The n-bit weight-only quantiser, or a clear error. It ships in onnxruntime>=1.22 (Python >= 3.10)
    and needs `onnx-ir`; older runtimes only have a 4-bit variant."""
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
    except ImportError as e:
        raise RuntimeError("--quantize needs onnxruntime>=1.22 (Python >= 3.10) and onnx-ir: pip install 'laya[onnx]' "
                           "(%s)" % e) from None
    return DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer


def _quantize_weights_int8(src: str, dst: str, block_size: int = 128) -> None:
    """Weight-only int8 for the encoder's linear layers, activations left in fp32.

    Dynamic quantisation (int8 activations too) was measured on the real checkpoints and broke the
    GeGLU feed-forward blocks: probabilities moved by up to 0.9 and 13 to 30% of argmaxes flipped.
    Quantising only the constant MatMul weights with `MatMulNBits` keeps the fp32 graph's answers
    (max probability difference about 0.05, argmax agreement 0.97 to 1.0) at about 40% of the
    file size. It does not make CPU inference faster on its own; the gain is memory.
    The decision head is left in fp32 because it is small and it is where calibration lives.
    """
    import onnx
    DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer = _weight_only_quantizer()

    model = onnx.load(src)
    head = [n.name for n in model.graph.node if n.op_type in ("MatMul", "Gemm") and not n.name.startswith("/encoder")]
    q = MatMulNBitsQuantizer(model, algo_config=DefaultWeightOnlyQuantConfig(block_size=block_size, is_symmetric=True,
                                                                              bits=8), nodes_to_exclude=head)
    q.process()
    onnx.save(q.model.model, dst)


# ------------------------------------------------------------------------------ runtime
def providers_for_device(device: Optional[str]) -> Optional[List[str]]:
    """Map a torch-style device string to an explicit ONNX Runtime provider list.

    `None` keeps the automatic choice (CUDA when the runtime has it, else CPU).
    """
    if device is None:
        return None
    d = str(device).lower()
    if d.startswith("cuda"):
        return ["CUDAExecutionProvider"]
    if d == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError("unsupported ONNX device %r; use cuda, cpu or None" % device)


class OnnxAgent(Agent):
    """`Agent` whose forward pass runs in ONNX Runtime. Same `predict` / `predict_many` API.

    `path` is an export directory from `export_onnx`. Prefers `model.int8.onnx` when present
    unless `prefer_quantized=False`. `providers` defaults to CUDA when the runtime has it, then
    CPU; that automatic choice logs a warning when torch can see a GPU but the ONNX runtime
    cannot (the `onnxruntime` wheel instead of `onnxruntime-gpu`). An explicit `providers`
    list that names a provider the runtime lacks raises, so a service configured for CUDA
    never quietly runs on CPU. `threads` sets intra-op parallelism (0 = runtime default).
    """

    def __init__(self, path: str, calibration: Optional[str] = None, truncate: Optional[str] = None,
                 prefer_quantized: bool = True, providers: Optional[List[str]] = None, threads: int = 0,
                 tokenizer: Any = None):
        import onnxruntime as ort
        import torch

        model_dir, self.cfg = resolve_checkpoint(path, require_weights=False)
        self.model_dir = model_dir
        self._init_common(model_dir, calibration, truncate, tokenizer=tokenizer)
        if calibration is None and os.path.exists(os.path.join(model_dir, "calibration.json")):
            self.load_calibration(os.path.join(model_dir, "calibration.json"))
            self.calibration_source = "export"

        q_path, f_path = os.path.join(model_dir, "model.int8.onnx"), os.path.join(model_dir, "model.onnx")
        self.onnx_path = q_path if (prefer_quantized and os.path.exists(q_path)) else f_path
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError("no model.onnx in %r; run `laya export-onnx` first" % path)
        self.quantized = self.onnx_path.endswith(".int8.onnx")

        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
            if "CUDAExecutionProvider" not in available and torch.cuda.is_available():
                logger.warning("laya.onnx: torch sees a CUDA device but this onnxruntime build has no "
                               "CUDAExecutionProvider; running %s on CPU. Install onnxruntime-gpu, or pass "
                               "providers=['CPUExecutionProvider'] to silence this.", os.path.basename(self.onnx_path))
        else:
            missing = [p for p in providers if p not in available]
            if missing:
                raise RuntimeError("ONNX Runtime provider(s) %s not available in this build (have %s); "
                                   "install onnxruntime-gpu for CUDAExecutionProvider" % (missing, available))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.onnx_path, so, providers=providers)
        self.providers = self.session.get_providers()
        self.device = torch.device("cuda" if "CUDAExecutionProvider" in self.providers else "cpu")
        self.dtype = torch.float32
        self.fell_back_to_cpu = False
        self.model = None
        logger.info("laya.onnx: loaded %s on %s", self.onnx_path, self.providers)

    def _forward_logits(self, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray, int]:
        b = collate_items([items], self.tok.pad_token_id)
        feeds = {
            "input_ids": b["input_ids"].numpy().astype(np.int64),
            "attention_mask": b["attention_mask"].numpy().astype(np.int64),
            "marker_pos": b["marker_pos"].numpy().astype(np.int64),
            "marker_mask": b["marker_mask"].numpy().astype(np.bool_),
            "qtype": b["qtype"].numpy().astype(np.int64),
        }
        logits, act_logits = self.session.run(OUTPUT_NAMES, feeds)
        logits = np.asarray(logits, dtype=np.float32)
        act = np.exp(act_logits - act_logits.max(-1, keepdims=True))
        act = (act / act.sum(-1, keepdims=True)).astype(np.float32)
        return logits, act, int(b["attention_mask"].sum())


# ------------------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from .agent import load

    p = argparse.ArgumentParser(prog="laya export-onnx", description="Export a Laya checkpoint to ONNX")
    p.add_argument("checkpoint", help="hub id or local path, e.g. convaiinnovations/laya")
    p.add_argument("out_dir")
    p.add_argument("--subfolder", help="checkpoint subfolder in a bundle repo, e.g. multilingual")
    p.add_argument("--calibration", help="calibration JSON to bake into the export")
    p.add_argument("--quantize", action="store_true", help="also write a weight-only int8 model (about 40% of the size, same answers)")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    agent = load(args.checkpoint, device="cpu", subfolder=args.subfolder, calibration=args.calibration)
    meta = export_onnx(agent, args.out_dir, quantize=args.quantize, opset=args.opset)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
