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
