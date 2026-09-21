"""Dynamic batching and the HTTP service, with a stub router: no weights, no network."""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from laya.router import Router
from laya.serving import DecisionService, DynamicBatcher, Metrics, Settings, create_app

Q = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}},
     "angry": {"type": "noul", "instructions": "Angry?"}}


class StubAgent:
    """Records every predict_many call; answers deterministically from the state text."""

    def __init__(self, name, delay=0.0, fail_on=None):
        self.name = name
        self.delay = delay
        self.fail_on = fail_on
        self.calls = []
        self.lock = threading.Lock()

    def predict_many(self, requests, truncate=None, max_batch=64):
        with self.lock:
            self.calls.append([len(qs) for _, qs in requests])
        if self.delay:
            time.sleep(self.delay)
        out = []
        for state, qs in requests:
            text = state if isinstance(state, str) else str(state)
            if self.fail_on and self.fail_on in text:
                raise RuntimeError("boom")
            answers = {}
            for qid, q in qs.items():
                if q["type"] == "choice":
                    keys = list(q["criteria"])
                    answers[qid] = {"type": "choice", "choice": keys[len(text) % len(keys)],
                                    "probabilities": {k: 0.5 for k in keys}, "confidence": 0.9, "entropy": 0.1}
                elif q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 0.2, "confidence": 0.8, "entropy": 0.7}
                else:
                    answers[qid] = {"type": "score", "score": 1.0, "level": 1, "confidence": 0.6, "entropy": 0.5}
            out.append({"model": self.name, "answers": answers,
                        "usage": {"input_tokens": len(text), "truncated": "LONG" in text, "truncation": truncate or "right"}})
        return out

    def system_one(self, state, questions):
        return self.predict_many([(state, questions)])[0]


@pytest.fixture
def stub_router(monkeypatch):
    agents = {}

    def _build(self, key):
        agents.setdefault(key, StubAgent(key))
        return agents[key]

    monkeypatch.setattr(Router, "_build", _build)
    r = Router(preload=True)
    r._stub_agents = agents
    return r


# ------------------------------------------------------------------ batcher
def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_batcher_groups_concurrent_requests():
    agent = StubAgent("english", delay=0.01)
    b = DynamicBatcher("english", agent, ThreadPoolExecutor(1), max_batch=32, max_wait=0.02)

    async def go():
        return await asyncio.gather(*[b.submit("s%d" % i, Q) for i in range(6)])

    results = run(go())
    assert len(results) == 6 and all(set(r["answers"]) == {"dept", "angry"} for r in results)
    assert len(agent.calls) == 1 and agent.calls[0] == [2] * 6          # one forward pass for all six


def test_batcher_drains_queue_while_a_pass_is_running():
    """Under load the timer must not slice the queue into many small passes: whatever arrives
    while one forward pass runs goes out together as soon as it finishes."""
    agent = StubAgent("english", delay=0.05)
    b = DynamicBatcher("english", agent, ThreadPoolExecutor(1), max_batch=64, max_wait=0.001)

    async def go():
        tasks = []
        for i in range(20):
            tasks.append(asyncio.ensure_future(b.submit("s%d" % i, Q)))
            await asyncio.sleep(0.002)          # arrivals spread over 40 ms, one pass takes 50 ms
        return await asyncio.gather(*tasks)

    results = run(go())
    assert len(results) == 20
    assert len(agent.calls) <= 3, agent.calls          # first arrival, then the drained rest
    assert max(len(c) for c in agent.calls) >= 15


def test_batcher_flushes_at_max_batch():
    agent = StubAgent("english")
    b = DynamicBatcher("english", agent, ThreadPoolExecutor(1), max_batch=4, max_wait=5.0)

    async def go():
        return await asyncio.wait_for(asyncio.gather(*[b.submit("s%d" % i, Q) for i in range(4)]), timeout=1.0)

    run(go())                                   # 4 requests x 2 questions >= max_batch: no 5 s wait
    assert len(agent.calls) == 1


def test_batcher_separates_truncation_sides():
    agent = StubAgent("english")
    b = DynamicBatcher("english", agent, ThreadPoolExecutor(1), max_batch=32, max_wait=0.01)

    async def go():
        return await asyncio.gather(b.submit("a", Q, "left"), b.submit("b", Q, "right"), b.submit("c", Q, "left"))

    res = run(go())
    assert [r["usage"]["truncation"] for r in res] == ["left", "right", "left"]
    assert sorted(len(c) for c in agent.calls) == [1, 2]


def test_batcher_propagates_errors_per_batch():
    agent = StubAgent("english", fail_on="bad")
    b = DynamicBatcher("english", agent, ThreadPoolExecutor(1), max_batch=32, max_wait=0.01)

    async def go():
        return await asyncio.gather(b.submit("good", Q), b.submit("bad", Q), return_exceptions=True)

    good, bad = run(go())
    assert isinstance(good, RuntimeError) and isinstance(bad, RuntimeError)   # same forward pass


def test_metrics_counts_without_prometheus():
    m = Metrics(prometheus=False)
    assert m.enabled is False
    m.request("english", "ok", 0.01)
    m.result("english", {"answers": {"a": {"type": "noul", "confidence": 0.9}}, "usage": {"truncated": True}})
    m.batch("english", 5, 0.02)
    assert m.counts["requests:english:ok"] == 1
    assert m.counts["questions:english:noul"] == 1
    assert m.counts["truncated:english"] == 1
    assert m.counts["batched_questions:english"] == 5


# ------------------------------------------------------------------ service
def test_service_routes_and_batches_per_model(stub_router):
    svc = DecisionService(stub_router, Settings(max_wait_ms=10))

    async def go():
        return await asyncio.gather(
            svc.decide({"body": "I was charged twice"}, Q),
            svc.decide({"body": "मुझसे दो बार शुल्क लिया गया"}, Q),
            svc.decide({"body": "another english one"}, Q),
        )

    res = run(go())
    assert [r["routing"]["model"] for r in res] == ["english", "multilingual", "english"]
    assert stub_router._stub_agents["english"].calls == [[2, 2]]
    assert stub_router._stub_agents["multilingual"].calls == [[2]]
    h = svc.health()
    assert h["ready"] is True and h["batches"] == {"english": 1, "multilingual": 1}


def test_service_validation(stub_router):
    svc = DecisionService(stub_router, Settings(max_state_chars=20))
    with pytest.raises(ValueError, match="non-empty"):
        run(svc.decide("s", {}))
    with pytest.raises(ValueError, match="type choice"):
        run(svc.decide("s", {"x": {"type": "bogus", "instructions": "?"}}))
    with pytest.raises(ValueError, match="criteria"):
        run(svc.decide("s", {"x": {"type": "choice", "instructions": "?"}}))
    with pytest.raises(ValueError, match="LAYA_MAX_STATE_CHARS"):
        run(svc.decide("x" * 21, Q))


def test_settings_from_env():
    s = Settings.from_env({"LAYA_MODELS": "english, typed", "LAYA_MAX_BATCH": "8", "LAYA_MAX_WAIT_MS": "2.5",
                           "LAYA_CALIBRATION": "english=/c/en.json,multilingual=/c/ml.json", "LAYA_API_KEY": "k",
                           "LAYA_BACKEND": "ONNX"})
    assert s.models == ["english", "typed"] and s.max_batch == 8 and s.max_wait_ms == 2.5
    assert s.calibration_for("english") == "/c/en.json" and s.calibration_for("typed-decisions") is None
    assert s.api_key == "k" and s.backend == "onnx"
    assert Settings(calibration="/one.json").calibration_for("anything") == "/one.json"


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def client(stub_router):
    from fastapi.testclient import TestClient
    app = create_app(router=stub_router, settings=Settings(max_wait_ms=5))
    with TestClient(app) as c:
        yield c


def test_http_decide(client):
    r = client.post("/v1/decide", json={"state": {"body": "charged twice"}, "questions": Q})
    assert r.status_code == 200
    body = r.json()
    assert body["routing"]["model"] == "english" and set(body["answers"]) == {"dept", "angry"}
    r = client.post("/v1/decide", json={"state": {"body": "मुझसे दो बार"}, "questions": Q, "model": "english"})
    assert r.json()["routing"]["model"] == "english"


def test_http_validation_and_health(client):
    assert client.post("/v1/decide", json={"state": "s", "questions": {}}).status_code == 422
    h = client.get("/healthz").json()
    assert h["status"] == "ok" and h["ready"] is True
    assert client.get("/readyz").status_code == 200
    m = client.get("/metrics")
    assert m.status_code in (200, 501)


def test_http_batch(client):
    r = client.post("/v1/decide/batch", json={"requests": [
        {"state": {"body": "charged twice"}, "questions": Q},
        {"state": {"body": "bad"}, "questions": {}},
        {"state": {"body": "मुझसे दो बार"}, "questions": Q},
    ]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["routing"]["model"] == "english"
    assert res[1]["status"] == 422
    assert res[2]["routing"]["model"] == "multilingual"
    assert client.post("/v1/decide/batch", json={"requests": []}).status_code == 422


def test_http_api_key(stub_router):
    from fastapi.testclient import TestClient
    app = create_app(router=stub_router, settings=Settings(api_key="secret"))
    with TestClient(app) as c:
        assert c.post("/v1/decide", json={"state": "s", "questions": Q}).status_code == 401
        ok = c.post("/v1/decide", json={"state": "s", "questions": Q}, headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200
        assert c.get("/healthz").status_code == 200          # health never needs the key


def test_client_against_app(client):
    """The stdlib client, with urlopen redirected into the test app."""
    import io
    from urllib.error import HTTPError

    from laya.client import LayaClient, LayaServiceError

    def opener(req, timeout=None):
        method = req.get_method()
        path = req.full_url.replace("http://test", "")
        data = req.data
        resp = client.request(method, path, content=data, headers=dict(req.header_items()))
        if resp.status_code >= 400:
            raise HTTPError(req.full_url, resp.status_code, "err", {}, io.BytesIO(resp.content))

        class R(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return R(resp.content)

    c = LayaClient("http://test", opener=opener)
    assert c.health()["status"] == "ok"
    out = c.decide({"body": "charged twice"}, Q, truncate="left")
    assert out["routing"]["model"] == "english" and out["usage"]["truncation"] == "left"
    assert len(c.decide_many([{"state": "a", "questions": Q}, {"state": "b", "questions": Q}])) == 2
    with pytest.raises(LayaServiceError) as ei:
        c.decide("s", {})
    assert ei.value.status == 422
