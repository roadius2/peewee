"""Helpers of scripts/jev_compare.py on canned answers: no weights, no network."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from jev_compare import agreement, from_jev, question_chunks, read_cache, report  # noqa: E402

CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None}}
SCORE = {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]}
NOUL = {"type": "noul", "instructions": "Angry?"}

RECORDS = [
    {"id": "A", "state": "x", "questions": {"team": CHOICE, "angry": NOUL},
     "targets": {"team": {"label": "a"}, "angry": {"label": "true"}}, "meta": {"workflow": "w"}},
    {"id": "B", "state": "y", "questions": {"urgent": SCORE}, "targets": {"urgent": {"probabilities": [0, 0, 1]}}},
]
JEV = {
    "A": {"answers": {"team": {"type": "choice", "choice": "a", "confidence": 0.2,
                               "probabilities": {"a": 0.62, "b": 0.4}},
                      "angry": {"type": "noul", "noul": 0.3}},
          "latency_ms": 300.0, "usage": {"input_tokens": 100}},
    "B": {"answers": {"urgent": {"type": "score", "score": 1.9, "confidence": 0.05,
                                 "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}},
          "latency_ms": 250.0, "usage": {"input_tokens": 50}},
}
PEEWEE = {
    "A": {"answers": {"team": {"probabilities": {"a": 0.9, "b": 0.1}, "confidence": 0.9},
                      "angry": {"noul": 0.8, "confidence": 0.8}}, "latency_ms": 20.0},
    "B": {"answers": {"urgent": {"probabilities": {"0": 0.6, "1": 0.3, "2": 0.1}, "confidence": 0.6}},
          "latency_ms": 10.0},
}


def test_from_jev_renormalises_and_uses_the_answer_probability_as_confidence():
    team = from_jev(CHOICE, JEV["A"]["answers"]["team"])
    assert sum(team["probabilities"].values()) == pytest.approx(1.0)
    assert team["confidence"] == pytest.approx(0.62 / 1.02)
    assert team["jev_confidence"] == 0.2
    urgent = from_jev(SCORE, JEV["B"]["answers"]["urgent"])
    assert urgent["confidence"] == pytest.approx(0.7)


def test_from_jev_gives_noul_the_probability_of_the_reported_side():
    assert from_jev(NOUL, {"type": "noul", "noul": 0.3})["confidence"] == pytest.approx(0.7)


def test_agreement_counts_questions_where_both_pick_the_same_answer():
    peewee = {"A": PEEWEE["A"]["answers"], "B": PEEWEE["B"]["answers"]}
    jev = {k: {q: from_jev(RECORDS[i]["questions"][q], a) for q, a in v["answers"].items()}
           for i, (k, v) in enumerate(JEV.items())}
    agree = agreement(RECORDS, peewee, jev)
    assert agree["overall"] == pytest.approx(1 / 3)          # team agrees (a); angry and urgent differ
    assert agree["by_type"] == {"choice": 1.0, "noul": 0.0, "score": 0.0}


def test_report_scores_both_systems_with_their_own_latency_and_tokens():
    rep = report(RECORDS, PEEWEE, JEV)
    assert rep["jev"]["overall"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)       # angry wrong
    assert rep["peewee"]["overall"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)    # urgent wrong
    assert rep["jev"]["latency_ms"]["p50"] == pytest.approx(275.0)
    assert rep["peewee"]["latency_ms"]["p50"] == pytest.approx(15.0)
    assert rep["jev"]["input_tokens"] == 150
    assert rep["agreement"]["overall"] == pytest.approx(1 / 3)


def test_report_refuses_a_cache_that_misses_cases():
    with pytest.raises(ValueError, match="B"):
        report(RECORDS, PEEWEE, {"A": JEV["A"]})


def test_question_chunks_split_a_big_case_and_keep_every_question_once():
    qs = {"q%d" % i: NOUL for i in range(7)}
    chunks = question_chunks(qs, 3)
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert [k for c in chunks for k in c] == list(qs)
    assert question_chunks(qs, 100) == [qs]


def test_read_cache_can_leave_out_errors_so_they_are_retried(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "A", "error": "HTTP 400"}\n{"id": "B", "answers": {}}\n')
    assert set(read_cache(str(path))) == {"A", "B"}
    assert set(read_cache(str(path), skip_errors=True)) == {"B"}
