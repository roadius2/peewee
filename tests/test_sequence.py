"""`build_sequence` and `collate_items` with a fake tokenizer: layout, budgets, truncation."""

from laya.common import build_sequence, collate_items

CHOICE = {"t": "choice", "ins": "Which team?", "crit": {"billing": "invoices", "tech": "bugs", "sales": None}}


def test_layout(fake_tok):
    ids, markers = build_sequence(fake_tok, "hello world", CHOICE, max_len=512, head_max_len=192)
    assert ids[0] == fake_tok.cls_token_id and ids[-1] == fake_tok.sep_token_id
    assert len(markers) == 3
    assert all(ids[m] == fake_tok.mask_token_id for m in markers)
    # the state comes after the second [SEP]
    seps = [i for i, t in enumerate(ids) if t == fake_tok.sep_token_id]
    assert ids[seps[1] + 1:seps[2]] == [fake_tok.word_id("hello"), fake_tok.word_id("world")]


def test_state_is_serialised_json_for_dicts(fake_tok):
    ids, _ = build_sequence(fake_tok, {"body": "hello"}, CHOICE)
    assert fake_tok.word_id('{"body":') in ids


def test_never_exceeds_max_len(fake_tok):
    state = " ".join("w%d" % i for i in range(2000))
    ids, markers = build_sequence(fake_tok, state, CHOICE, max_len=128, head_max_len=48)
    assert len(ids) == 128
    assert all(m < 128 for m in markers)


def test_right_truncation_keeps_the_start(fake_tok):
    state = " ".join("w%d" % i for i in range(300))
    ids, _ = build_sequence(fake_tok, state, CHOICE, max_len=64, head_max_len=32)
    assert fake_tok.word_id("w0") in ids
    assert fake_tok.word_id("w299") not in ids


def test_left_truncation_keeps_the_end(fake_tok):
    state = " ".join("w%d" % i for i in range(300))
    ids, _ = build_sequence(fake_tok, state, CHOICE, max_len=64, head_max_len=32, truncate_left=True)
    assert fake_tok.word_id("w299") in ids
    assert fake_tok.word_id("w0") not in ids


def test_option_text_is_capped_at_48_tokens(fake_tok):
    long_opt = " ".join("o%d" % i for i in range(100))
    q = {"t": "choice", "ins": "x", "crit": {"a": long_opt, "b": None}}
    ids, markers = build_sequence(fake_tok, "s", q, max_len=512, head_max_len=192)
    assert markers[1] - markers[0] == 1 + 48     # [MASK] + 48 option tokens


def test_many_options_are_squeezed_not_dropped(fake_tok):
    q = {"t": "choice", "ins": "x", "crit": {"opt%d" % i: "one two three four five six" for i in range(77)}}
    ids, markers = build_sequence(fake_tok, "s", q, max_len=512, head_max_len=192)
    assert len(markers) == 77
    gaps = [b - a for a, b in zip(markers, markers[1:])]
    assert set(gaps) == {4}                       # [MASK] + 3 tokens each


def test_markers_past_max_len_are_dropped(fake_tok):
    q = {"t": "choice", "ins": "x", "crit": {"opt%d" % i: "one two three four five six" for i in range(77)}}
    _, markers = build_sequence(fake_tok, "s", q, max_len=128, head_max_len=192)
    assert len(markers) < 77


def test_mask_token_in_input_is_neutralised(fake_tok):
    q = {"t": "noul", "ins": "Is it [MASK] bad?", "crit": None}
    ids, markers = build_sequence(fake_tok, "text with [MASK] inside", q)
    assert [i for i, t in enumerate(ids) if t == fake_tok.mask_token_id] == markers


def test_collate_pads_and_masks(fake_tok):
    a, ma = build_sequence(fake_tok, "short", CHOICE)
    b, mb = build_sequence(fake_tok, " ".join(["long"] * 40), {"t": "noul", "ins": "q", "crit": None})
    batch = collate_items([[{"ids": a, "markers": ma, "qtype": 0}, {"ids": b, "markers": mb, "qtype": 2}]],
                          fake_tok.pad_token_id)
    assert batch["input_ids"].shape == (2, max(len(a), len(b)))
    assert batch["attention_mask"].sum(1).tolist() == [len(a), len(b)]
    assert batch["marker_mask"].sum(1).tolist() == [3, 2]
    assert batch["qtype"].tolist() == [0, 2]
    assert collate_items([[]], fake_tok.pad_token_id) is None


def test_return_info_reports_cuts(fake_tok):
    state = " ".join("w%d" % i for i in range(300))
    ids, markers, info = build_sequence(fake_tok, state, CHOICE, max_len=64, head_max_len=32, return_info=True)
    assert info["state_tokens"] == 300
    assert 0 < info["state_tokens_kept"] < 64
    assert info["state_tokens_dropped"] == 300 - info["state_tokens_kept"]
    assert info["truncated"] is True and info["truncation"] == "right"
    assert info["options"] == 3 and info["options_squeezed"] is False and info["options_over_cap"] == 0

    _, _, info = build_sequence(fake_tok, "short", CHOICE, return_info=True)
    assert info == {"state_tokens": 1, "state_tokens_kept": 1, "state_tokens_dropped": 0, "truncated": False,
                    "truncation": "right", "options": 3, "tokens_per_option": None, "options_squeezed": False,
                    "options_over_cap": 0, "instructions_tokens_dropped": 0}

    q = {"t": "choice", "ins": "x", "crit": {"opt%d" % i: "one two three four five six" for i in range(77)}}
    _, _, info = build_sequence(fake_tok, "s", q, max_len=512, head_max_len=192, return_info=True)
    assert info["options_squeezed"] is True and info["tokens_per_option"] == 3

    q = {"t": "choice", "ins": "x", "crit": {"a": " ".join(["z"] * 100), "b": None}}
    _, _, info = build_sequence(fake_tok, "s", q, return_info=True)
    assert info["options_over_cap"] == 1 and info["options_squeezed"] is False
