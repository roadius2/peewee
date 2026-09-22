"""laya.data: JSONL schema, target vectors and reference answers. No weights, no network."""
import json
import pytest

from laya.data import (convert_open_jev_rows, convert_typed_decisions_row, main as prepare_main, option_keys,
                       read_jsonl, record_to_items, reference_index, split_cases, target_vector, validate_record,
                       write_jsonl)
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
    ({"meta": ["not", "a", "dict"]}, "case-1: 'meta' must be an object"),
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


def _td_row():
    return {"id": 7, "workflow": "customer_service", "split": "test", "label_agreement": 0.8,
            "state": json.dumps({"ticket": "refund please"}),
            "questions": json.dumps({"team": CHOICE, "urgent": SCORE, "angry": NOUL}),
            "gold": json.dumps({
                "team": {"label": "tech", "probabilities": {"billing": 0.5, "tech": 0.3, "other": 0.2}, "type": "choice"},
                "urgent": {"label": "2", "probabilities": {"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}, "type": "score"},
                "angry": {"label": "false", "probabilities": {"false": 0.9, "true": 0.1}, "type": "noul"}})}


def test_typed_decisions_rows_convert_to_valid_records():
    rec = convert_typed_decisions_row(_td_row())
    validate_record(rec)
    assert rec["id"] == "7" and rec["state"] == {"ticket": "refund please"}
    assert rec["targets"]["team"] == {"probabilities": {"billing": 0.5, "tech": 0.3, "other": 0.2}, "label": "tech"}
    assert rec["meta"] == {"workflow": "customer_service", "split": "test", "label_agreement": 0.8}
    assert reference_index(rec["questions"]["team"], rec["targets"]["team"]) == 1


def test_plain_text_states_are_kept_as_text():
    assert convert_typed_decisions_row(dict(_td_row(), state="just a sentence"))["state"] == "just a sentence"


def test_prepare_data_writes_train_and_test(tmp_path, monkeypatch):
    import laya.data as data
    monkeypatch.setattr(data, "convert_typed_decisions",
                        lambda split: [convert_typed_decisions_row(dict(_td_row(), id=split))])
    assert prepare_main(["typed-decisions", "--out", str(tmp_path)]) == 0
    assert [r["id"] for r in read_jsonl(str(tmp_path / "train.jsonl"))] == ["train"]
    assert [r["id"] for r in read_jsonl(str(tmp_path / "test.jsonl"))] == ["test"]


def test_prepare_data_rejects_open_jev_options_for_typed_decisions(tmp_path, monkeypatch):
    import laya.data as data
    monkeypatch.setattr(data, "convert_typed_decisions",
                        lambda split: [convert_typed_decisions_row(dict(_td_row(), id=split))])
    with pytest.raises(SystemExit) as e:
        prepare_main(["typed-decisions", "--out", str(tmp_path), "--config", "some-config"])
    assert e.value.code == 2


def _oj_rows():
    base = {"group_id": "g1", "split": "train", "source": "customer-control-v1"}
    s1 = json.dumps({"ticket": "refund please"})
    s2 = json.dumps({"ticket": "app crashes"})
    return [
        dict(base, id="g1:cat", kind="choice", question="Which category?", options=["billing: money", "bug: defects"],
             target=[0.9, 0.1], state_json=s1),
        dict(base, id="g1:angry", kind="noul", question="Is the user angry?", options=["no", "yes"],
             target=[0.0, 1.0], state_json=s1),
        dict(base, id="g1:sev", kind="score", question="Severity?", options=["low", "mid", "high"],
             target=[0.2, 0.5, 0.3], state_json=s1),
        dict(base, id="g1:cat2", kind="choice", question="Which category?", options=["bug: defects", "billing: money"],
             target=[1.0, 0.0], state_json=s2),
    ]


def test_open_jev_rows_group_into_one_case_per_state():
    recs = convert_open_jev_rows(_oj_rows())
    assert len(recs) == 2 and len({r["id"] for r in recs}) == 2
    for r in recs:
        validate_record(r)
    first = next(r for r in recs if "g1:cat" in r["questions"])
    assert set(first["questions"]) == {"g1:cat", "g1:angry", "g1:sev"}
    assert first["state"] == {"ticket": "refund please"}
    assert first["meta"] == {"group": "g1", "workflow": "customer-control-v1", "split": "train"}
    assert option_keys(first["questions"]["g1:cat"]) == ["billing: money", "bug: defects"]
    assert target_vector(first["questions"]["g1:cat"], first["targets"]["g1:cat"]) == pytest.approx([0.9, 0.1])
    assert target_vector(first["questions"]["g1:angry"], first["targets"]["g1:angry"]) == [0.0, 1.0]
    assert first["questions"]["g1:sev"]["criteria"] == ["low", "mid", "high"]


@pytest.mark.parametrize("change, message", [
    ({"kind": "rank"}, "unknown kind"),
    ({"kind": "noul", "options": ["false", "true"]}, "noul options"),
    ({"options": ["same", "same"]}, "duplicate option"),
])
def test_bad_open_jev_rows_raise(change, message):
    with pytest.raises(ValueError, match=message):
        convert_open_jev_rows([dict(_oj_rows()[0], **change)])


def test_split_keeps_a_group_together():
    recs = [_record(id="g%d-v%d" % (g, v), meta={"group": "g%d" % g}) for g in range(10) for v in range(3)]
    train, held = split_cases(recs, 0.3, seed=0)
    held_groups = {r["meta"]["group"] for r in held}
    assert len(held_groups) == 3 and len(held) == 9
    assert not held_groups & {r["meta"]["group"] for r in train}


def test_prepare_data_open_jev_writes_every_split(tmp_path, monkeypatch):
    import laya.data as data
    calls = []

    def fake(config, split, revision):
        calls.append((config, split, revision))
        return convert_open_jev_rows([dict(r, split=split) for r in _oj_rows()])

    monkeypatch.setattr(data, "convert_open_jev", fake)
    assert prepare_main(["open-jev", "--out", str(tmp_path), "--config", "context-retention-control-v1"]) == 0
    assert [c[1] for c in calls] == ["train", "calibration", "validation", "test", "ood"]
    assert {c[0] for c in calls} == {"context-retention-control-v1"} and calls[0][2] == data.OPEN_JEV_REVISION
    assert len(read_jsonl(str(tmp_path / "ood.jsonl"))) == 2
