"""laya.data: JSONL schema, target vectors and reference answers. No weights, no network."""
import pytest

from laya.data import option_keys, read_jsonl, reference_index, target_vector, validate_record, write_jsonl

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
