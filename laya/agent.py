"""High-level inference runtime for laya System 1 decision models."""
import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .calibrate import (
    apply_calibration_payload,
    fit_temperature_map,
    load_calibration as _load_calibration_file,
    save_calibration as _save_calibration_file,
)
from .common import (
    QTYPES,
    amp_dtype,
    build_model,
    build_sequence,
    collate_items,
    normalized_entropy,
    render_options,
    temp_bucket,
    top_probability,
)

logger = logging.getLogger("laya")


def _check_truncate(side: Optional[str]) -> Optional[str]:
    if side is None:
        return None
    side = str(side).lower()
    if side not in ("left", "right"):
        raise ValueError("truncate must be 'left', 'right' or None, got %r" % (side,))
    return side


def patch_tokenizer_config(tcfg: Dict[str, Any]) -> bool:
    """Make a checkpoint's tokenizer_config loadable across transformers versions.

    Mutates `tcfg` in place and returns True when anything changed. Two known problems:

    * `tokenizer_class` missing or `TokenizersBackend` (written by newer tokenizers) which older
      transformers cannot resolve; `PreTrainedTokenizerFast` loads the same tokenizer.json.
    * Checkpoints built on the mmBERT/Gemma tokenizer store `extra_special_tokens` as a list;
      transformers expects a mapping and raises "'list' object has no attribute 'keys'".
    """
    changed = False
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
        changed = True
    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
        changed = True
    return changed


def _tokenizer_dir(model_dir: str) -> Optional[str]:
    """Directory to load the tokenizer from: the checkpoint's own, or a patched private copy.

    The Hugging Face cache is shared by every process on the machine, so it is never written
    to. When the shipped tokenizer_config.json needs patching, the tokenizer folder is copied
    into a per-process temporary directory and patched there.
    """
    tok_dir = os.path.join(model_dir, "tokenizer")
    if not os.path.isdir(tok_dir):
        return None
    cfg_file = os.path.join(tok_dir, "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return tok_dir
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("laya: could not read %s (%s); loading tokenizer as-is", cfg_file, e)
        return tok_dir
    if not patch_tokenizer_config(tcfg):
        return tok_dir
    try:
        patched = tempfile.mkdtemp(prefix="laya-tokenizer-")
        shutil.copytree(tok_dir, patched, dirs_exist_ok=True)
        with open(os.path.join(patched, "tokenizer_config.json"), "w") as f:
            json.dump(tcfg, f, indent=2)
        logger.debug("laya: patched tokenizer_config.json into %s", patched)
        return patched
    except OSError as e:
        logger.warning("laya: could not create a patched tokenizer copy (%s); loading as-is", e)
        return tok_dir


def _verify_compatibility(model: torch.nn.Module, cfg: Dict, weights: Dict[str, torch.Tensor], model_id: str):
    """Verify that the loaded checkpoint weights and config strictly match the expected architecture."""
    # 1. Verify required configuration attributes
    required_cfg = ["encoder", "head_layers"]
    missing_cfg = [k for k in required_cfg if k not in cfg]
    if missing_cfg:
        raise ValueError(
            f"Incompatible model config for {model_id!r}: missing configuration keys {missing_cfg}. "
            f"Ensure this is a valid RL Agent decision model."
        )

    # 2. Check for required component prefixes
    required_prefixes = ("encoder.", "type_emb.", "scorer.", "act_head.")
    for prefix in required_prefixes:
        if not any(k.startswith(prefix) for k in weights.keys()):
            raise ValueError(
                f"Incompatible model weights for {model_id!r}: checkpoint is missing '{prefix}' parameters. "
                f"Expected an RL Agent decision model with encoder and decision heads."
            )

    # 3. Check for parameter shape mismatches
    model_sd = model.state_dict()
    shape_mismatches = []
    missing_keys = []

    for name, param in model.named_parameters():
        if name not in weights:
            missing_keys.append(name)
        elif tuple(weights[name].shape) != tuple(param.shape):
            shape_mismatches.append(f"  - {name}: expected {tuple(param.shape)}, found {tuple(weights[name].shape)}")

    if shape_mismatches:
        err_details = "\n".join(shape_mismatches[:5])
        if len(shape_mismatches) > 5:
            err_details += f"\n  ... and {len(shape_mismatches) - 5} more mismatched layers."
        raise ValueError(
            f"Model architecture mismatch for {model_id!r}:\n{err_details}\n"
            f"The checkpoint weights do not match the configured model architecture."
        )

    if missing_keys:
        raise ValueError(
            f"Model weights incomplete for {model_id!r}: missing {len(missing_keys)} parameter tensors "
            f"(e.g. {missing_keys[:3]})."
        )


class Agent:
    """System 1 decision model runtime: fast, non-autoregressive, calibrated decisions."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        calibration: Optional[str] = None,
        truncate: Optional[str] = None,
    ):
        """Load a Laya checkpoint.

        `subfolder` selects one checkpoint from a repo that bundles several, e.g.
        `Agent("convaiinnovations/laya", subfolder="multilingual")`. Only that subfolder is
        downloaded, so bundling does not cost every user the whole family.

        `calibration` is a JSON file written by `save_calibration`; its temperatures override
        the ones shipped in the checkpoint config.

        `truncate` sets the default side to cut an over-long state from: "right" keeps the
        start, "left" keeps the end. Leave it None to keep the start of strings and dicts and
        the end of lists (conversation transcripts), which is what a gate over an agent's
        transcript needs.
        """
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(
                    f"Local model path not found: {model_id_or_path!r}. "
                    f"Check that the directory exists and that training saved the model successfully."
                )
            from huggingface_hub import snapshot_download

            kw = {"token": token or os.environ.get("HF_TOKEN")}
            if subfolder:
                # fetch only the requested checkpoint, not every checkpoint in the repo
                kw["allow_patterns"] = [f"{subfolder}/*"]
            model_dir = snapshot_download(model_id_or_path, **kw)

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(
                    f"Subfolder {subfolder!r} not found in {model_id_or_path!r}."
                )

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Incompatible model: {model_id_or_path!r} does not contain 'rl_agent_config.json'. "
                f"Make sure you are loading a compatible RL Agent model (e.g. 'convaiinnovations/rl-agent')."
            )

        with open(cfg_path) as f:
            self.cfg = json.load(f)

        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Incompatible model: 'model.safetensors' not found in {model_id_or_path!r}."
            )

        # 1. Device resolution with automatic fallback
        if device is not None:
            target_device = torch.device(device)
            if target_device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("laya: CUDA requested but not available; falling back to CPU.")
                self.device = torch.device("cpu")
            elif target_device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("laya: MPS requested but not available; falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = target_device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        tok_dir = _tokenizer_dir(model_dir)
        self.tok = AutoTokenizer.from_pretrained(tok_dir if tok_dir else self.cfg.get("encoder"))

        enc_dir = os.path.join(model_dir, "encoder")
        self.model = build_model(self.cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)

        # Load weights and verify architectural compatibility
        weights = load_file(weights_path)
        _verify_compatibility(self.model, self.cfg, weights, model_id_or_path)

        self.model.load_state_dict(weights, strict=True)

        # ModernBERT's reference_compile defaults to "auto" and will torch.compile the encoder.
        # That is a loss for the batch sizes Laya runs (a handful of questions per call) and can
        # hang on some platforms, so keep the eager path.
        try:
            self.model.encoder.config.reference_compile = False
        except Exception:
            pass

        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.calibration_source = "checkpoint"
        if calibration:
            self.load_calibration(calibration)
        self.truncate = _check_truncate(truncate)
        self._warned = set()
        self.dtype = amp_dtype(self.cfg.get("amp_dtype", "fp16"))

        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.dtype = torch.float16
        elif self.device.type in ("cpu", "mps"):
            self.dtype = torch.float32

        # 2. Place on device with graceful fallback to CPU on memory error
        fell_back_from = fell_back_why = None
        try:
            self.model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if self.device.type != "cpu":
                # Record what actually went wrong: the reason matters more than the symptom,
                # and it is the only place the underlying exception is ever surfaced.
                fell_back_from, fell_back_why = self.device, e
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.model.to(self.device).eval()
            else:
                raise e

        self.fell_back_to_cpu = fell_back_from is not None
        if self.fell_back_to_cpu:
            logger.warning(
                "laya: could not place the model on %s, so it is running on CPU. Reason: %s. "
                "Inference will be roughly 10-15x slower (~200-500 ms rather than ~35 ms). "
                "If this is a newer NVIDIA GPU (Blackwell / RTX 50-series), your PyTorch build may "
                "not support its CUDA architecture: "
                "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 "
                "(see https://pytorch.org/get-started/locally/).",
                fell_back_from, fell_back_why)

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            ins = json.dumps(ins)
        return {"t": t, "ins": ins, "crit": crit}

    # ------------------------------------------------------------------ pipeline pieces
    def _warn_once(self, key: str, msg: str, *args):
        """Log a WARNING the first time `key` happens on this agent, DEBUG afterwards."""
        if key in self._warned:
            logger.debug(msg, *args)
        else:
            self._warned.add(key)
            logger.warning(msg + " (further occurrences logged at DEBUG)", *args)

    def _resolve_truncate(self, state, truncate: Optional[str]) -> str:
        side = _check_truncate(truncate) or self.truncate
        if side is None:
            side = "left" if isinstance(state, (list, tuple)) else "right"
        return side

    def _build_items(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
                     truncate: Optional[str] = None) -> List[Dict]:
        """Tokenise every question against `state` into collatable items (no torch, no weights).

        Each item carries an `info` dict from `build_sequence` describing what was cut.
        """
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        side = self._resolve_truncate(state, truncate)
        items = []
        for qid, qdef in questions.items():
            q = self._to_internal(qdef)
            seq, markers, info = build_sequence(self.tok, state, q, max_len, head_max_len,
                                                truncate_left=(side == "left"), return_info=True)
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "info": info})
        return items

    @torch.no_grad()
    def _forward_logits(self, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run one forward pass over collated items.

        Returns raw (un-tempered) option logits `[n, kmax]`, action-head probabilities
        `[n, n_act]`, and the number of non-pad input tokens. Errors propagate: a GPU failure
        is the caller's to handle (retry, shed load, or restart), never a reason to silently
        move a shared model to CPU for the rest of the process.
        """
        b = collate_items([items], self.tok.pad_token_id)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=use_amp):
            logits, act = self.model(
                b["input_ids"].to(self.device),
                b["attention_mask"].to(self.device),
                b["marker_pos"].to(self.device),
                b["marker_mask"].to(self.device),
                b["qtype"].to(self.device),
            )
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        return logits, act, int(b["attention_mask"].sum())

    def _usage(self, questions: Dict[str, Any], items: List[Dict], n_tokens: int) -> Dict[str, Any]:
        """Token accounting plus everything that was cut, so callers can act on it."""
        infos = [it.get("info") or {} for it in items]
        state_tokens = max((i.get("state_tokens", 0) for i in infos), default=0)
        dropped = max((i.get("state_tokens_dropped", 0) for i in infos), default=0)
        side = next((i.get("truncation") for i in infos if i.get("truncation")), "right")
        usage: Dict[str, Any] = {
            "input_tokens": n_tokens,
            "output_tokens": 0,
            "state_tokens": state_tokens,
            "state_tokens_dropped": dropped,
            "truncated": dropped > 0,
            "truncation": side,
        }
        if dropped > 0:
            self._warn_once(
                "state_truncated",
                "laya: state of %d tokens was truncated (%s side kept); up to %d tokens dropped. "
                "The result is computed on a partial state.", state_tokens, side, dropped)
        budget = {}
        for (qid, _qdef), info in zip(questions.items(), infos):
            if info.get("options_squeezed") or info.get("options_over_cap"):
                budget[qid] = {k: info[k] for k in ("options", "tokens_per_option", "options_squeezed",
                                                     "options_over_cap", "instructions_tokens_dropped")}
        if budget:
            usage["option_budget"] = budget
            squeezed = [qid for qid, b in budget.items() if b["options_squeezed"]]
            if squeezed:
                self._warn_once(
                    "options_squeezed",
                    "laya: options of question(s) %s were cut to as few as %d tokens each to fit "
                    "head_max_len=%d. Accuracy degrades sharply here; raise head_max_len or use "
                    "laya.patterns.hierarchical_choice.",
                    squeezed, min(budget[q]["tokens_per_option"] for q in squeezed),
                    self.cfg.get("head_max_len", 192))
        return usage

    def _postprocess(
        self,
        questions: Dict[str, Dict[str, Any]],
        items: List[Dict],
        logits: np.ndarray,
        act: np.ndarray,
        n_tokens: int,
    ) -> Dict[str, Any]:
        """Turn raw logits into the public answer payload (temperature, confidence, labels).

        `confidence` is the calibrated probability of the reported answer for every question
        type (the top option for `choice` and `score`, `max(p, 1-p)` for `noul`), so one
        threshold means the same thing everywhere. `entropy` is the normalised entropy of the
        full distribution (0 = certain, 1 = uniform) for callers who want spread as well.
        """
        answers = {}
        for r, (qid, qdef) in enumerate(questions.items()):
            q = self._to_internal(qdef)
            k = len(items[r]["markers"])
            qt = QTYPES[q["t"]]
            t_scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = logits[r, :k] / max(1e-3, float(t_scale))
            p = np.exp(z - z.max())
            p = p / p.sum()

            conf = round(top_probability(p, k), 4)
            ent = round(normalized_entropy(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}

            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf,
                    "entropy": ent,
                    "action": ext,
                }
            elif q["t"] == "score":
                exp_score = float((np.arange(k) * p).sum())
                answers[qid] = {
                    "type": "score",
                    "score": round(exp_score, 4),
                    "level": int(p.argmax()),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf,
                    "entropy": ent,
                    "action": ext,
                }
            else:
                answers[qid] = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": conf,
                    "entropy": ent,
                    "action": ext,
                }

        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": self._usage(questions, items, n_tokens),
        }

    # ------------------------------------------------------------------ public API
    def system_one(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]],
                   truncate: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate typed questions across state in a single, parallel forward pass.

        Args:
            state: Text string, JSON dict, or conversation turn list.
            questions: Dictionary mapping question_id -> question definition.
                - choice: {"type": "choice", "instructions": "...", "criteria": {"optA": "...", ...}}
                - score:  {"type": "score",  "instructions": "...", "criteria": ["lvl0", "lvl1", ...]}
                - noul:   {"type": "noul",   "instructions": "..."}
            truncate: "right" keeps the start of an over-long state, "left" keeps the end.
                Default: the agent's setting, else start for strings/dicts and end for lists.

        Returns:
            Dictionary with `answers`, and `usage` that reports token counts and whatever was
            truncated (`usage["truncated"]`, `usage["state_tokens_dropped"]`,
            `usage["option_budget"]`).
        """
        items = self._build_items(state, questions, truncate=truncate)
        logits, act, n_tokens = self._forward_logits(items)
        return self._postprocess(questions, items, logits, act, n_tokens)

    def predict_many(self, requests: Sequence[Any], truncate: Optional[str] = None,
                     max_batch: int = 64) -> List[Dict[str, Any]]:
        """Answer many (state, questions) requests in as few forward passes as possible.

        `requests` is a sequence of `(state, questions)` tuples or `{"state", "questions"}`
        dicts. Every question of every request is collated into one batch (chunked at
        `max_batch` items to bound memory), so a GPU sees one large pass instead of many small
        ones: the README measures 39 ms for one question and 7 ms per question at ten.
        Results come back in request order with the same payload as `predict`.
        """
        parsed = []
        for req in requests:
            if isinstance(req, dict) and "state" in req and "questions" in req:
                parsed.append((req["state"], req["questions"]))
            else:
                state, questions = req
                parsed.append((state, questions))
        per_request_items = [self._build_items(st, qs, truncate=truncate) for st, qs in parsed]
        flat = [it for items in per_request_items for it in items]
        rows: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(0, len(flat), max(1, int(max_batch))):
            chunk = flat[i:i + max(1, int(max_batch))]
            logits, act, _ = self._forward_logits(chunk)
            rows.extend((logits[r], act[r]) for r in range(len(chunk)))
        results = []
        pos = 0
        for (st, qs), items in zip(parsed, per_request_items):
            n = len(items)
            sel = rows[pos:pos + n]
            pos += n
            if n == 0:
                results.append({"model": "laya-rl-agent", "answers": {}, "usage": self._usage(qs, items, 0)})
                continue
            kmax = max(len(z) for z, _ in sel)
            logits = np.full((n, kmax), -1e4, dtype=np.float32)
            act = np.zeros((n, sel[0][1].shape[0]), dtype=np.float32)
            n_tok = 0
            for r, (z, a) in enumerate(sel):
                logits[r, :len(z)] = z
                act[r] = a
                n_tok += len(items[r]["ids"])
            results.append(self._postprocess(qs, items, logits, act, n_tok))
        return results

    # ------------------------------------------------------------------ calibration
    def fit_temperatures(self, records, min_bucket_n: int = 10, report: bool = True) -> Dict[str, Any]:
        """Fit per-type and per-bucket temperatures from `laya.calibrate.collect_records` output
        and apply them to this agent. Returns the fitted map plus a before/after report."""
        result = fit_temperature_map(records, min_bucket_n=min_bucket_n, report=report)
        self.temperature = list(result["temperature"])
        self.temperature_by_options = dict(result["temperature_by_options"])
        self.calibration_source = "fitted"
        return result

    def save_calibration(self, path: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Write the current temperatures to JSON (no weights)."""
        meta = dict(meta or {})
        meta.setdefault("encoder", self.cfg.get("encoder"))
        _save_calibration_file(path, self.temperature, self.temperature_by_options, meta)

    def load_calibration(self, path: str) -> None:
        """Replace the current temperatures with a JSON file written by `save_calibration`."""
        apply_calibration_payload(self, _load_calibration_file(path))
        self.calibration_source = path

    predict = system_one


RLAgent = Agent


def load(model_id_or_path: str = "convaiinnovations/laya", device: Optional[str] = None,
         token: Optional[str] = None, subfolder: Optional[str] = None,
         calibration: Optional[str] = None, truncate: Optional[str] = None) -> Agent:
    """Load a Laya agent.

    `subfolder` picks one checkpoint out of a repo that bundles several:

        laya.load("convaiinnovations/laya")                           # English (repo root)
        laya.load("convaiinnovations/laya", subfolder="multilingual")

    `calibration` is a JSON file from `Agent.save_calibration`; `truncate` is the default
    side to cut over-long states from ("left" keeps the end).
    """
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder,
                 calibration=calibration, truncate=truncate)
