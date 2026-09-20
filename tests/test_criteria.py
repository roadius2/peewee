"""Criteria rendering: structured values must not crash or leak Python reprs (upstream PR #2)."""
import json

import pytest

from laya.common import render_criterion, render_options


@pytest.mark.parametrize("value, want", [
    ("phishing or scam", "phishing or scam"),
    ({"desc": "phishing"}, '{"desc": "phishing"}'),
    (["a", "b"], '["a", "b"]'),
    (3, "3"),
    (False, "false"),
    ({"d": "münchen"}, '{"d": "münchen"}'),
])
def test_render_criterion(value, want):
    assert render_criterion(value) == want


def test_unserialisable_falls_back_to_str():
    assert isinstance(render_criterion({"o": object()}), str)


def test_noul_dict_criteria():
    out = render_options({"t": "noul", "ins": "Is this phishing?",
                          "crit": {"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}})
    assert out == ['false: {"desc": "legitimate"}', 'true: {"desc": "phishing, scam or fraud"}']
    assert json.loads(out[1].split("true: ", 1)[1]) == {"desc": "phishing, scam or fraud"}


def test_choice_rendering():
    out = render_options({"t": "choice", "ins": "x",
                          "crit": {"billing": {"desc": "payments"}, "tech": None, "sales": ""}})
    assert out == ['billing: {"desc": "payments"}', "tech", "sales"]
    # 0 and False are real criterion values, not "missing"
    assert render_options({"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False}}) == ["zero: 0", "no: false"]


def test_score_rendering():
    assert render_options({"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2]}) == [
        'level 0: {"d": "low"}', "level 1: high", "level 2: 2"]


def test_defaults_and_plain_strings():
    assert render_options({"t": "noul", "ins": "x", "crit": None}) == [
        "false: no, the statement does not hold", "true: yes, the statement holds"]
    assert render_options({"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}}) == [
        "false: no", "true: yes it is"]
    assert render_options({"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}}) == ["a: first", "b"]


@pytest.mark.parametrize("q", [
    {"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}},
    {"t": "score", "ins": "x", "crit": [{"a": 1}, [2], None]},
    {"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}},
])
def test_every_option_is_str(q):
    assert all(isinstance(o, str) for o in render_options(q))
