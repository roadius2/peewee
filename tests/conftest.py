"""Shared fixtures. Nothing here downloads weights: every fixture is a stub or a fake tokenizer."""
import os

import pytest

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Needs real weights on disk; run it by hand: python tests/test_local_e2e.py [model_root]
collect_ignore = ["test_local_e2e.py"]


class FakeTokenizer:
    """Whitespace tokenizer with the attributes `build_sequence` and `collate_items` use.

    One token per word, ids >= 10 so they never collide with the special ids. Deterministic
    across processes (no `hash()`), so tests can assert on exact ids.
    """

    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2
    mask_token_id = 3
    mask_token = "[MASK]"

    @staticmethod
    def word_id(word: str) -> int:
        return 10 + sum(ord(c) for c in word) % 10000

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.word_id(w) for w in text.split()]}


@pytest.fixture
def fake_tok():
    return FakeTokenizer()


POS_WORDS = ["good", "great", "happy", "love", "fine"]
NEG_WORDS = ["bad", "awful", "sad", "hate", "poor"]
TINY_WORDS = sorted(set(POS_WORDS + NEG_WORDS + (
    "review reviewer is the a positive or negative how many stars one three five star level question choice "
    "score noul false true no yes statement does not hold holds sentiment").split()))


@pytest.fixture(scope="session")
def tiny_base(tmp_path_factory):
    """A complete, loadable Laya checkpoint: tiny random BERT encoder, real word-level tokenizer.

    No weights are downloaded. `laya.Agent(tiny_base, device="cpu")` loads it end to end.
    """
    import json

    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import BertConfig, PreTrainedTokenizerFast

    from laya.common import build_model

    d = tmp_path_factory.mktemp("tiny-base")
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + TINY_WORDS)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                            sep_token="[SEP]", mask_token="[MASK]").save_pretrained(str(d / "tokenizer"))
    BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
               intermediate_size=64, max_position_embeddings=256).save_pretrained(str(d / "encoder"))
    cfg = {"encoder": "tiny-bert", "head_layers": 1, "max_len": 128, "head_max_len": 64,
           "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0], "amp_dtype": "bf16"}
    torch.manual_seed(0)
    model = build_model(cfg, encoder_dir=str(d / "encoder"))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    (d / "rl_agent_config.json").write_text(json.dumps(cfg))
    return str(d)


def toy_records(n: int = 24, seed: int = 0):
    """Learnable toy cases for trainer tests: the review's words decide every answer."""
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        pos = i % 2 == 0
        words = [rng.choice(POS_WORDS if pos else NEG_WORDS) for _ in range(6)]
        out.append({
            "id": "case-%03d" % i,
            "state": {"review": " ".join(words)},
            "questions": {
                "sentiment": {"type": "choice", "instructions": "Is the review positive or negative?",
                              "criteria": {"positive": None, "negative": None}},
                "happy": {"type": "noul", "instructions": "Is the reviewer happy?"},
                "stars": {"type": "score", "instructions": "How many stars?",
                          "criteria": ["one star", "three stars", "five stars"]},
            },
            "targets": {
                "sentiment": {"probabilities": {"positive": 0.9, "negative": 0.1} if pos
                              else {"positive": 0.1, "negative": 0.9}},
                "happy": {"label": pos},
                "stars": {"probabilities": [0.05, 0.15, 0.8] if pos else [0.8, 0.15, 0.05]},
            },
            "meta": {"workflow": "reviews"},
        })
    return out
