"""Hierarchical choice and tool selection over a scripted model."""
import pytest

from laya.patterns import hierarchical_choice, select_tool


class Scripted:
    """Answers each `choice` question with a preset option and probability, recording the calls."""

    def __init__(self, picks):
        self.picks = list(picks)
        self.calls = []

    def predict(self, state, questions, **kw):
        (qid, q), = questions.items()
        self.calls.append((state, q, kw))
        keys = list(q["criteria"])
        pick, conf = self.picks.pop(0)
        probs = {k: (conf if k == pick else round((1 - conf) / max(1, len(keys) - 1), 4)) for k in keys}
        return {"answers": {qid: {"type": "choice", "choice": pick, "probabilities": probs, "confidence": conf}},
                "usage": {}}


GROUPS = {
    "billing": {"refund": "money back", "invoice": "copy of a bill"},
    "account": ["login", "password", "delete_account"],
    "shipping": {"track": "where is my parcel"},
}


def test_hierarchical_choice_two_stages():
    m = Scripted([("account", 0.8), ("password", 0.5)])
    out = hierarchical_choice(m, {"body": "forgot my pw"}, GROUPS, "What does the user need?",
                              group_descriptions={"billing": "money"}, truncate="left")
    assert out["choice"] == "password" and out["group"] == "account"
    assert out["confidence"] == pytest.approx(0.4)
    assert out["group_probabilities"]["account"] == 0.8
    assert len(m.calls) == 2
    stage1_q, stage2_q = m.calls[0][1], m.calls[1][1]
    assert list(stage1_q["criteria"]) == ["billing", "account", "shipping"]
    assert stage1_q["criteria"]["billing"] == "money"                   # explicit description used
    assert stage1_q["criteria"]["account"] == "login, password, delete_account"   # else member list
    assert stage2_q["criteria"] == {"login": None, "password": None, "delete_account": None}
    assert m.calls[0][2] == {"truncate": "left"}                        # predict kwargs pass through


def test_hierarchical_choice_single_member_group_skips_stage_two():
    m = Scripted([("shipping", 0.9)])
    out = hierarchical_choice(m, "where is it", GROUPS, "?")
    assert out["choice"] == "track" and out["confidence"] == 0.9 and out["stages"][1] is None
    assert len(m.calls) == 1


def test_hierarchical_choice_rejects_empty():
    with pytest.raises(ValueError):
        hierarchical_choice(Scripted([]), "s", {}, "?")


def test_select_tool_flat_below_threshold():
    tools = {"read": {"description": "read a file"}, "write": {"description": "write a file"}}
    m = Scripted([("write", 0.7)])
    out = select_tool(m, {"task": "save it"}, tools)
    assert out["choice"] == "write" and out["group"] is None and out["confidence"] == 0.7
    assert out["probabilities"] == {"read": 0.3, "write": 0.7}
    assert len(out["stages"]) == 1 and out["stages"][0]["answers"]["tool"]["choice"] == "write"
    assert len(m.calls) == 1 and list(m.calls[0][1]["criteria"]) == ["read", "write"]


def test_select_tool_goes_hierarchical_above_threshold():
    tools = {"t%d" % i: {"description": "tool %d" % i, "group": "g%d" % (i % 3)} for i in range(20)}
    tools["lonely"] = {"description": "no group"}
    m = Scripted([("g1", 0.6), ("t4", 0.5)])
    out = select_tool(m, {"task": "x"}, tools, max_flat=16)
    assert out["choice"] == "t4" and out["group"] == "g1"
    assert set(m.calls[0][1]["criteria"]) == {"g0", "g1", "g2", "other"}
    assert m.calls[0][1]["criteria"]["other"] == "lonely"


class Recording:
    """Answers every question in one predict call, so the test can assert on the batch."""

    def __init__(self, picks):
        self.picks = picks
        self.calls = []

    def predict(self, state, questions, **kw):
        self.calls.append((state, dict(questions), kw))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                pick = self.picks.get(qid, keys[0])
                answers[qid] = {"type": "choice", "choice": pick, "confidence": 0.7,
                                "probabilities": {k: (0.7 if k == pick else 0.1) for k in keys}}
            else:
                answers[qid] = {"type": q["type"], "noul": 0.6, "confidence": 0.6}
        return {"answers": answers, "usage": {"input_tokens": 10}}


def test_speculative_choice_one_call():
    from laya.patterns import speculative_choice
    primary = {"type": "choice", "instructions": "Next operation?",
               "criteria": {"CLICK": None, "TYPE_TEXT": None, "WAIT": None, "DONE": None}}
    elements = {"e1": "button Search", "e2": "textbox Origin", "e3": "link Help"}
    deps = {"CLICK": {"type": "choice", "instructions": "Click which element?", "criteria": elements},
            "TYPE_TEXT": {"type": "choice", "instructions": "Type into which field?", "criteria": {"e2": "textbox Origin"}}}
    m = Recording({"operation": "CLICK", "operation__if_CLICK": "e1"})
    out = speculative_choice(m, {"url": "x"}, primary, deps, truncate="left")
    assert len(m.calls) == 1                                          # everything in one forward pass
    assert set(m.calls[0][1]) == {"operation", "operation__if_CLICK", "operation__if_TYPE_TEXT"}
    assert m.calls[0][2] == {"truncate": "left"}
    assert out["choice"] == "CLICK" and out["followup"]["choice"] == "e1"
    assert out["followup_id"] == "operation__if_CLICK"
    assert set(out["speculative"]) == {"CLICK", "TYPE_TEXT"}
    assert out["usage"] == {"input_tokens": 10}

    m = Recording({"operation": "WAIT"})
    out = speculative_choice(m, "s", primary, deps)
    assert out["choice"] == "WAIT" and out["followup"] is None and out["followup_id"] is None


def test_speculative_choice_validation():
    from laya.patterns import speculative_choice
    primary = {"type": "choice", "instructions": "?", "criteria": {"a": None}}
    with pytest.raises(ValueError, match="not in primary"):
        speculative_choice(Recording({}), "s", primary, {"zzz": {"type": "noul", "instructions": "?"}})
    with pytest.raises(ValueError, match="'choice'"):
        speculative_choice(Recording({}), "s", {"type": "noul", "instructions": "?"}, {})
