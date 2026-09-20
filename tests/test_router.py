"""Routing decisions and resident-model bookkeeping. `Router._build` is stubbed: no weights."""
import logging
import threading
import time

import pytest

from laya.router import (
    BUNDLE_REPO,
    DEFAULT_MODELS,
    STANDALONE_MODELS,
    RouteDecision,
    Router,
    _repo_str,
    match_typed_decisions_workflow,
    normalise_name,
)

TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}
Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}}}
Q_TD = {i: {"type": "noul", "instructions": "x"} for i in TD["customer_service"]}


class Stub:
    def __init__(self, name):
        self.name = name

    def system_one(self, state, questions):
        return {"model": self.name, "answers": {}, "usage": {}}


@pytest.fixture
def stub_build(monkeypatch):
    """Replace model construction with an instant stub; returns the list of keys built."""
    built = []

    def _build(self, key):
        built.append(key)
        return Stub(key)

    monkeypatch.setattr(Router, "_build", _build)
    return built


# ------------------------------------------------------------------ workflow signatures
@pytest.mark.parametrize("wf", list(TD))
def test_workflow_signature_matches(wf):
    assert match_typed_decisions_workflow({i: {} for i in TD[wf]}) == wf


def test_workflow_requires_exact_id_set():
    assert match_typed_decisions_workflow({"urgency": {}, "category": {}}) is None
    assert match_typed_decisions_workflow({i: {} for i in TD["customer_service"] + ["extra"]}) is None
    assert match_typed_decisions_workflow({}) is None


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("alias, want", [
    ("en", "english"), ("laya", "english"), ("multi", "multilingual"), ("ML", "multilingual"),
    ("typed", "typed-decisions"), ("typed_decisions", "typed-decisions"), ("English", "english"),
])
def test_normalise_name(alias, want):
    assert normalise_name(alias) == want


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        normalise_name("nope")


# ------------------------------------------------------------------ routing decisions
@pytest.mark.parametrize("state, qs, kw, want", [
    ({"body": "I was charged twice, please refund."}, Q_GENERIC, {}, "english"),
    ({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "二重に請求されました"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "두 번 청구되었습니다"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "تم خصم المبلغ مرتين"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "Հայերեն տեքստ"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung "
              "fuer die Rechnung die nicht korrekt ist"}, Q_GENERIC, {}, "multilingual"),
    ({"body": "anything"}, Q_GENERIC, {"model": "multilingual"}, "multilingual"),
    ({"body": "मुझसे दो बार"}, Q_GENERIC, {"model": "english"}, "english"),
    ({"body": "x"}, Q_GENERIC, {"task": "typed_decisions"}, "typed-decisions"),
    ({"body": "मुझसे दो बार"}, Q_GENERIC, {"lang": "en"}, "english"),
    ({"body": "hello there"}, Q_GENERIC, {"lang": "de"}, "multilingual"),
    ({"body": "I was charged twice"}, Q_TD, {}, "english"),          # auto detection is opt-in
    ({}, Q_GENERIC, {}, "english"),
    (None, Q_GENERIC, {}, "english"),
])
def test_route(state, qs, kw, want):
    assert Router().route(state, qs, **kw)["model"] == want


def test_auto_task_detection_is_opt_in():
    r = Router(auto_task_detection=True)
    assert r.route({"body": "I was charged twice"}, Q_TD)["model"] == "typed-decisions"
    assert r.route({"body": "I was charged twice"}, Q_GENERIC)["model"] == "english"
    assert r.route({"body": "x"}, Q_TD, model="multilingual")["model"] == "multilingual"


def test_workflow_decision_repo_is_a_string():
    # Regression: this branch used to return the raw (repo, subfolder) tuple.
    d = Router(auto_task_detection=True).route({"body": "x"}, Q_TD)
    assert d["repo"] == "convaiinnovations/laya/typed-decisions"
    assert d["workflow"] == "customer_service"


def test_decision_payload():
    d = Router().route({"body": "मुझसे दो बार शुल्क लिया गया"}, Q_GENERIC)
    assert isinstance(d, RouteDecision) and isinstance(d, dict)
    assert d["repo"] == "convaiinnovations/laya/multilingual"
    assert d.model == "multilingual"
    assert d["detection"]["script"] == "devanagari"
    assert d.reason


def test_custom_default():
    assert Router(default="multilingual").route("12345", Q_GENERIC)["model"] == "multilingual"


# ------------------------------------------------------------------ bundle vs standalone
def test_model_specs():
    assert DEFAULT_MODELS["english"] == (BUNDLE_REPO, None)
    assert DEFAULT_MODELS["multilingual"] == (BUNDLE_REPO, "multilingual")
    assert sorted(STANDALONE_MODELS) == sorted(DEFAULT_MODELS)
    assert _repo_str((BUNDLE_REPO, None)) == "convaiinnovations/laya"
    assert _repo_str((BUNDLE_REPO, "multilingual")) == "convaiinnovations/laya/multilingual"
    assert _repo_str("some/repo") == "some/repo"


def test_standalone_and_local_overrides():
    hi = {"m": "मुझसे दो बार"}
    assert Router().route(hi, Q_GENERIC)["repo"] == "convaiinnovations/laya/multilingual"
    assert Router(standalone_repos=True).route(hi, Q_GENERIC)["repo"] == "convaiinnovations/laya-multilingual"
    assert Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"}).route(hi, Q_GENERIC)["repo"] == "/tmp/ml"


# ------------------------------------------------------------------ residency / LRU
def test_default_keeps_two_resident(stub_build):
    r = Router()
    r.load("english")
    r.load("multilingual")
    assert sorted(r.loaded) == ["english", "multilingual"]
    assert stub_build == ["english", "multilingual"]


def test_lru_eviction(stub_build):
    r = Router(max_loaded=1)
    r.load("english")
    r.load("multilingual")
    assert r.loaded == ["multilingual"] and sorted(r._agents) == ["multilingual"]

    r = Router(max_loaded=2)
    r.load("english")
    r.load("multilingual")
    r.load("typed-decisions")
    assert r.loaded == ["multilingual", "typed-decisions"]

    r = Router(max_loaded=2)
    r.load("english")
    r.load("multilingual")
    r.load("english")
    r.load("typed-decisions")
    assert sorted(r.loaded) == ["english", "typed-decisions"]

    r.unload("english")
    assert "english" not in r.loaded
    r.unload()
    assert r.loaded == []


def test_eviction_is_logged(stub_build, caplog):
    r = Router(max_loaded=1)
    with caplog.at_level(logging.WARNING, logger="laya.router"):
        r.load("english")
        assert not caplog.records
        r.load("multilingual")
    assert any("evicted ['english']" in rec.getMessage() for rec in caplog.records)


def test_incremental_preload_keeps_residents(stub_build):
    # upstream PR #27: a second preload must not evict what the first one built
    r = Router(max_loaded=1)
    r.preload(["english"])
    r.preload(["multilingual"])
    assert sorted(r.loaded) == ["english", "multilingual"]
    assert r.max_loaded >= 2


def test_preload_all_and_touch(stub_build):
    r = Router(max_loaded=1, preload=True)
    assert sorted(r.loaded) == ["english", "multilingual", "typed-decisions"]
    r.load("english")                       # touching a resident model evicts nothing
    assert sorted(r.loaded) == ["english", "multilingual", "typed-decisions"]
    assert stub_build.count("english") == 1


def test_attach(stub_build):
    r = Router(max_loaded=1)
    sentinel = Stub("already-built")
    r.attach("en", sentinel)
    assert r._agents["english"] is sentinel and "english" in r.loaded
    r.max_loaded = 2
    r.load("multilingual")
    assert r._agents["english"] is sentinel
    assert "english" not in stub_build


def test_predict_attaches_routing(stub_build):
    out = Router().predict({"body": "I was charged twice"}, Q_GENERIC)
    assert out["model"] == "english"
    assert out["routing"]["model"] == "english"


# ------------------------------------------------------------------ thread safety
def test_concurrent_load_builds_once(monkeypatch):
    built = []

    def slow_build(self, key):
        time.sleep(0.05)
        built.append(key)
        return Stub(key)

    monkeypatch.setattr(Router, "_build", slow_build)
    r = Router()
    threads = [threading.Thread(target=r.load, args=("english",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built == ["english"]
    assert r.loaded == ["english"]


def test_concurrent_mixed_loads_respect_cap(monkeypatch):
    def slow_build(self, key):
        time.sleep(0.02)
        return Stub(key)

    monkeypatch.setattr(Router, "_build", slow_build)
    r = Router(max_loaded=2)
    names = ["english", "multilingual", "typed-decisions"] * 4
    threads = [threading.Thread(target=r.load, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(r.loaded) == 2
    assert sorted(r._agents) == sorted(r.loaded)
