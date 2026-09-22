"""peewee_decide.evaluate: metrics on a stub agent, and the eval command on the tiny checkpoint."""
import json
import types

import pytest

from peewee_decide.evaluate import evaluate, load_model_for_eval, main as eval_main
from tests.conftest import toy_records

CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None, "c": None}}
NOUL = {"type": "noul", "instructions": "Angry?"}
SCORE = {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]}

RECORDS = [
    {"id": "A", "state": {"id": "A"}, "questions": {"team": CHOICE, "angry": NOUL},
     "targets": {"team": {"probabilities": {"a": 0.6, "b": 0.4}, "label": "b"}, "angry": {"label": "true"}},
     "meta": {"workflow": "w1"}},
    {"id": "B", "state": {"id": "B"}, "questions": {"urgent": SCORE},
     "targets": {"urgent": {"probabilities": [0.0, 0.0, 1.0]}}, "meta": {"workflow": "w2"}},
]
ANSWERS = {
    "A": {"team": {"probabilities": {"a": 0.7, "b": 0.2, "c": 0.1}, "confidence": 0.7},
          "angry": {"noul": 0.5, "confidence": 0.5}},
    "B": {"urgent": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.7}},
}


class StubAgent:
    def predict(self, state, questions, truncate=None):
        return {"answers": ANSWERS[state["id"]]}


def test_metrics_use_the_label_as_reference_and_the_noul_threshold():
    rep = evaluate(StubAgent(), RECORDS)
    assert rep["n_cases"] == 2 and rep["n_questions"] == 3
    assert rep["overall"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)       # team wrong (label b), angry and urgent right
    assert rep["by_type"]["choice"]["accuracy"] == 0.0
    assert rep["by_type"]["noul"]["accuracy"] == 1.0                            # p(true) = 0.5 counts as true
    assert rep["by_type"]["choice"]["soft_accuracy"] == pytest.approx(0.5)
    assert rep["by_type"]["choice"]["brier"] == pytest.approx(0.06)
    assert rep["by_type"]["score"]["brier"] == pytest.approx(0.14)
    assert rep["by_type"]["score"]["score_mae"] == pytest.approx(0.4)
    assert rep["by_type"]["score"]["score_within_one"] == 1.0
    assert rep["overall"]["ece"] == pytest.approx(0.3, abs=1e-3)
    assert set(rep["by_workflow"]) == {"w1", "w2"}
    assert rep["by_workflow"]["w1"]["accuracy"] == 0.5 and rep["by_workflow"]["w2"]["accuracy"] == 1.0
    assert set(rep["latency_ms"]) == {"p50", "p95"}


def test_latency_is_reported_per_case_and_per_question(monkeypatch):
    from peewee_decide import evaluate as ev
    ticks = iter([0.0, 0.010, 1.0, 1.030])                             # case A: 10 ms, 2 questions; B: 30 ms, 1
    monkeypatch.setattr(ev, "time", types.SimpleNamespace(perf_counter=lambda: next(ticks)))
    rep = evaluate(StubAgent(), RECORDS)
    assert rep["latency_ms"]["p50"] == pytest.approx(20.0)
    assert rep["latency_ms_per_question"]["p50"] == pytest.approx(17.5)  # median of 5 and 30
    assert "per question" in ev.format_report(rep)


def test_choice_keys_keep_their_type():
    tier = {"type": "choice", "instructions": "Which tier?", "criteria": [1, 2]}
    rec = {"id": "C", "state": {"id": "C"}, "questions": {"tier": tier}, "targets": {"tier": {"label": 1}}}

    class IntKeyAgent:
        def predict(self, state, questions, truncate=None):          # the runtime reports the raw keys
            return {"answers": {"tier": {"probabilities": {1: 0.8, 2: 0.2}, "confidence": 0.8}}}

    assert evaluate(IntKeyAgent(), [rec])["overall"]["accuracy"] == 1.0


def test_load_model_for_eval_accepts_a_checkpoint_directory(tiny_base):
    from peewee_decide.agent import Agent
    agent = load_model_for_eval(tiny_base, device="cpu")
    assert isinstance(agent, Agent)


def test_load_model_for_eval_routes_a_hub_repo_id_through_resolve_base(tiny_base, monkeypatch):
    import peewee_decide.train as lt
    calls = []

    def fake_resolve_base(base, token=None):
        calls.append(base)
        from peewee_decide.agent import resolve_checkpoint
        return resolve_checkpoint(tiny_base)

    monkeypatch.setattr(lt, "resolve_base", fake_resolve_base)
    from peewee_decide.agent import Agent
    agent = load_model_for_eval("convaiinnovations/laya-typed-decisions", device="cpu")
    assert isinstance(agent, Agent)
    assert calls == ["convaiinnovations/laya-typed-decisions"]


def test_eval_command_on_the_tiny_checkpoint(tiny_base, tmp_path, capsys):
    from peewee_decide.data import write_jsonl
    data = str(tmp_path / "cases.jsonl")
    write_jsonl(data, toy_records(6))
    out = str(tmp_path / "report.json")
    assert eval_main([tiny_base, "--data", data, "--device", "cpu", "--out", out]) == 0
    with open(out) as f:
        rep = json.load(f)
    assert rep["n_questions"] == 18 and 0.0 <= rep["overall"]["accuracy"] <= 1.0
    assert "accuracy" in capsys.readouterr().out
