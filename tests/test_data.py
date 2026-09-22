"""laya.data: JSONL schema, target vectors and reference answers. No weights, no network."""
import pytest

from laya.data import (option_keys, read_jsonl, record_to_items, reference_index, split_cases, target_vector,
                       validate_record, write_jsonl)
from tests.conftest import FakeTokenizer

TOK = FakeTokenizer()

CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "tech": None, "other": None}}
CHOICE_LIST = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "tech"]}
SCORE = {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high", "critical"]}
NOUL = {"type": "noul", "instructions": "Is the user angry?"}


def test_option_keys_follow_render_order():
    assert option_keys(CHOICE) == ["billing", "tech", "other"]
    assert option_keys(CHOICE_LIST) == ["billing", "tech"]
    assert option_keys(SCORE) == ["0", "1", "2", "3"]
    assert option_keys(NOUL) == ["false", "true"]


def test_soft_targets_are_ordered_normalised_and_default_missing_keys_to_zero():
    assert target_vector(CHOICE, {"probabilities": {"other": 1.0, "billing": 3.0}}) == [0.75, 0.0, 0.25]
    assert target_vector(SCORE, {"probabilities": {"3": 0.5, "0": 0.5}}) == [0.5, 0.0, 0.0, 0.5]
    assert target_vector(NOUL, {"probabilities": {"true": 0.2, "false": 0.8}}) == pytest.approx([0.8, 0.2])
    assert target_vector(SCORE, {"probabilities": [1, 1, 1, 1]}) == [0.25] * 4


@pytest.mark.parametrize("target, message", [
    ({"probabilities": {"billing": 1.0, "sales": 1.0}}, "unknown option"),
    ({"probabilities": {"billing": 0.0}}, "no mass"),
    ({"probabilities": {"billing": -0.5, "tech": 1.0}}, "negative"),
    ({"probabilities": [0.5, 0.5]}, "expected 3"),
    ({}, "needs 'probabilities' or 'label'"),
])
def test_bad_soft_targets_raise(target, message):
    with pytest.raises(ValueError, match=message):
        target_vector(CHOICE, target)


def test_hard_labels_become_one_hot():
    assert target_vector(CHOICE, {"label": "tech"}) == [0.0, 1.0, 0.0]
    assert target_vector(SCORE, {"label": "2"}) == [0.0, 0.0, 1.0, 0.0]
    assert target_vector(SCORE, {"label": 1}) == [0.0, 1.0, 0.0, 0.0]
    assert target_vector(NOUL, {"label": "true"}) == [0.0, 1.0]
    assert target_vector(NOUL, {"label": False}) == [1.0, 0.0]


def test_reference_answer_prefers_the_label_over_the_argmax():
    both = {"probabilities": {"billing": 0.6, "tech": 0.4}, "label": "tech"}
    assert reference_index(CHOICE, both) == 1
    assert reference_index(CHOICE, {"probabilities": {"billing": 0.6, "tech": 0.4}}) == 0
    assert reference_index(SCORE, {"label": "3"}) == 3
    assert reference_index(NOUL, {"label": "false"}) == 0
    assert reference_index(NOUL, {"probabilities": {"true": 0.5, "false": 0.5}}) == 1


def _record(**over):
    rec = {"id": "case-1", "state": {"body": "charged twice"},
           "questions": {"team": CHOICE, "angry": NOUL},
           "targets": {"team": {"label": "billing"}, "angry": {"probabilities": {"true": 0.7, "false": 0.3}}}}
    rec.update(over)
    return rec


def test_a_valid_record_passes():
    validate_record(_record())


@pytest.mark.parametrize("over, message", [
    ({"id": ""}, "string 'id'"),
    ({"questions": {}}, "non-empty"),
    ({"questions": {"team": dict(CHOICE, type="rank")}, "targets": {"team": {"label": "billing"}}}, "case-1/team: type"),
    ({"questions": {"team": dict(CHOICE, criteria={"only": None})}, "targets": {"team": {"label": "only"}}},
     "at least two"),
    ({"targets": {"team": {"label": "billing"}}}, "case-1/angry: no target"),
    ({"targets": {"team": {"label": "billing"}, "angry": {"label": "true"}, "ghost": {"label": 1}}}, "case-1/ghost"),
    ({"targets": {"team": {"label": "sales"}, "angry": {"label": "true"}}}, "case-1/team"),
])
def test_invalid_records_name_the_case_and_question(over, message):
    with pytest.raises(ValueError, match=message):
        validate_record(_record(**over))


def test_jsonl_round_trip(tmp_path):
    path = str(tmp_path / "cases.jsonl")
    write_jsonl(path, [_record(), _record(id="case-2")])
    assert [r["id"] for r in read_jsonl(path)] == ["case-1", "case-2"]
    assert read_jsonl(path)[0]["state"] == {"body": "charged twice"}


def test_record_to_items_builds_one_item_per_question():
    items, skipped = record_to_items(_record(), TOK, max_len=128, head_max_len=64)
    assert skipped == []
    by_q = {it["qid"]: it for it in items}
    assert set(by_q) == {"team", "angry"}
    team = by_q["team"]
    assert team["qtype"] == 0 and team["target"] == [1.0, 0.0, 0.0] and team["label"] == 0
    assert len(team["markers"]) == 3 and team["case"] == "case-1"
    angry = by_q["angry"]
    assert angry["qtype"] == 2 and angry["target"] == pytest.approx([0.3, 0.7]) and angry["label"] == 1


def test_questions_whose_options_do_not_fit_are_skipped_and_named():
    items, skipped = record_to_items(_record(), TOK, max_len=8, head_max_len=6)
    assert skipped and all(s.startswith("case-1/") for s in skipped)
    assert len(items) + len(skipped) == 2


def test_list_states_keep_their_end_by_default():
    from laya.common import serialize_state
    state = ["opening " + "filler " * 60, "closing words here"]
    toks = TOK(serialize_state(state))["input_ids"]
    rec = _record(state=state)
    left, _ = record_to_items(rec, TOK, max_len=48, head_max_len=24)
    right, _ = record_to_items(rec, TOK, max_len=48, head_max_len=24, truncate="right")
    assert toks[-1] in left[0]["ids"] and toks[0] not in left[0]["ids"]
    assert toks[0] in right[0]["ids"] and toks[-1] not in right[0]["ids"]


def _cases(n):
    return [_record(id="case-%02d" % i) for i in range(n)]


def test_split_is_by_case_deterministic_and_order_independent():
    recs = _cases(20)
    train, held = split_cases(recs, 0.25, seed=3)
    assert len(held) == 5 and len(train) == 15
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    _, held_again = split_cases(list(reversed(recs)), 0.25, seed=3)
    assert {r["id"] for r in held_again} == {r["id"] for r in held}


def test_split_edge_cases():
    assert split_cases(_cases(5), 0.0, seed=0)[1] == []
    assert len(split_cases(_cases(3), 0.01, seed=0)[1]) == 1
    with pytest.raises(ValueError, match="duplicate case ids"):
        split_cases(_cases(2) + _cases(1), 0.5, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        split_cases(_cases(4), 1.0, seed=0)
