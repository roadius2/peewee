"""Helpers of scripts/length_sweep.py on the fake whitespace tokenizer: no weights, no network."""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from length_sweep import pad_state, select_long, token_len  # noqa: E402
from tests.conftest import FakeTokenizer  # noqa: E402

TOK = FakeTokenizer()


def test_token_len_counts_with_the_agent_tokenizer():
    assert token_len(TOK, "one two three") == 3


def test_select_long_keeps_only_rows_at_or_above_the_threshold():
    rows = [{"text": " ".join(["w"] * n), "label": n % 2} for n in (5, 50, 120, 7, 200, 99)]
    picked = select_long(rows, TOK, min_tokens=100, n=10, seed=0)
    assert sorted(p["tokens"] for p in picked) == [120, 200]
    assert all(p["label"] == p["tokens"] % 2 for p in picked)


def test_select_long_caps_the_sample_and_shuffles_deterministically():
    rows = [{"text": " ".join(["w"] * (100 + i)), "label": 0} for i in range(40)]
    a = select_long(rows, TOK, min_tokens=100, n=8, seed=3)
    b = select_long(rows, TOK, min_tokens=100, n=8, seed=3)
    assert len(a) == 8 and [p["tokens"] for p in a] == [p["tokens"] for p in b]
    assert [p["tokens"] for p in a] != sorted(p["tokens"] for p in a)


def test_pad_state_fills_background_up_to_the_target_and_keeps_the_review():
    fillers = [" ".join(["f%d" % i] * 37) for i in range(50)]      # 37 tokens each
    review = "great film ten out of ten"
    state, tokens = pad_state(review, fillers, TOK, target_tokens=400, seed=1)
    assert state["review"] == review
    assert 400 - 40 <= tokens <= 400 + 5                                  # whole fillers, so a little short
    assert token_len(TOK, state["background"]) + token_len(TOK, review) == tokens


def test_pad_state_with_a_target_below_the_review_adds_no_background():
    state, tokens = pad_state("a b c d", ["x y z"] * 3, TOK, target_tokens=2, seed=0)
    assert state["background"] == "" and tokens == 4


def test_module_has_no_import_side_effects():
    import length_sweep
    assert hasattr(length_sweep, "main") and pytest is not None
