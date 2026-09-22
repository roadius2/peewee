"""Peewee as a service: dynamic batching, HTTP API, health and metrics.

    peewee serve                       # or: python -m peewee_decide.serving
    uvicorn peewee_decide.serving:create_app --factory

One process holds a preloaded `Router`. Every incoming request is routed (pure Python,
microseconds) to a checkpoint name and queued on that checkpoint's `DynamicBatcher`. A batcher
flushes when it holds `max_batch` questions or `max_wait_ms` has elapsed since the first one
arrived, runs `Agent.predict_many` on the whole group in one forward pass, and resolves each
request's future. Hundreds of agents polling the service therefore share forward passes instead
of queueing behind each other.

Environment
-----------
PEEWEE_MODELS        comma-separated checkpoints to preload (default: english,multilingual)
PEEWEE_DEVICE        cuda | cpu | mps (default: auto)
PEEWEE_HF_TOKEN      Hugging Face token for private repos
PEEWEE_CALIBRATION   path to a calibration JSON applied to every loaded checkpoint, or
                   name=path,name=path for per-checkpoint files
PEEWEE_TRUNCATE      default truncation side: left | right (default: agent default)
PEEWEE_MAX_BATCH     questions per forward pass (default 32)
PEEWEE_MAX_WAIT_MS   how long a batch waits for company before it runs (default 5)
PEEWEE_WORKERS       inference threads; keep 1 per GPU (default 1)
PEEWEE_API_KEY       optional bearer token required on /v1/* endpoints
PEEWEE_MAX_STATE_CHARS   reject states longer than this many characters (default 200000)
PEEWEE_BACKEND       torch (default) | onnx (see peewee_decide.onnx_backend; PEEWEE_MODELS then names
                   exported directories: name=path,name=path). With onnx, PEEWEE_DEVICE=cuda
                   requires the CUDA provider and fails at startup without it; unset, the
                   backend picks CUDA when available and warns if it has to run on CPU.

Endpoints
---------
GET  /healthz              liveness: loaded checkpoints, device, queue depth
GET  /readyz               readiness: 200 only once every configured checkpoint is resident
GET  /metrics              Prometheus exposition (requires prometheus-client)
POST /v1/decide            {"state", "questions", "model"?, "lang"?, "truncate"?}
POST /v1/decide/batch      {"requests": [{"state", "questions", ...}, ...]}
"""
import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("peewee_decide.serving")


# ------------------------------------------------------------------------------ settings
@dataclass
class Settings:
    models: List[str] = field(default_factory=lambda: ["english", "multilingual"])
    device: Optional[str] = None
    hf_token: Optional[str] = None
    calibration: Optional[str] = None
    truncate: Optional[str] = None
    max_batch: int = 32
    max_wait_ms: float = 5.0
    workers: int = 1
    api_key: Optional[str] = None
    max_state_chars: int = 200_000
    backend: str = "torch"
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Settings":
        e = os.environ if env is None else env
        s = cls()
        if e.get("PEEWEE_MODELS"):
            s.models = [m.strip() for m in e["PEEWEE_MODELS"].split(",") if m.strip()]
        s.device = e.get("PEEWEE_DEVICE") or None
        s.hf_token = e.get("PEEWEE_HF_TOKEN") or e.get("HF_TOKEN") or None
        s.calibration = e.get("PEEWEE_CALIBRATION") or None
        s.truncate = e.get("PEEWEE_TRUNCATE") or None
        s.max_batch = int(e.get("PEEWEE_MAX_BATCH", s.max_batch))
        s.max_wait_ms = float(e.get("PEEWEE_MAX_WAIT_MS", s.max_wait_ms))
        s.workers = int(e.get("PEEWEE_WORKERS", s.workers))
        s.api_key = e.get("PEEWEE_API_KEY") or None
        s.max_state_chars = int(e.get("PEEWEE_MAX_STATE_CHARS", s.max_state_chars))
        s.backend = (e.get("PEEWEE_BACKEND") or s.backend).lower()
        s.host = e.get("PEEWEE_HOST", s.host)
        s.port = int(e.get("PEEWEE_PORT", s.port))
        return s

    def calibration_for(self, name: str) -> Optional[str]:
        """Resolve PEEWEE_CALIBRATION for one checkpoint: a single path, or name=path pairs."""
        if not self.calibration:
            return None
        if "=" not in self.calibration:
            return self.calibration
        for pair in self.calibration.split(","):
            k, _, v = pair.partition("=")
            if k.strip() == name and v.strip():
                return v.strip()
        return None


# ------------------------------------------------------------------------------ metrics
_PROM: Dict[str, Any] = {}


def _prometheus_collectors() -> Optional[Dict[str, Any]]:
    """Create the Prometheus collectors once per process (the default registry rejects duplicates)."""
    if _PROM:
        return _PROM
    try:
        from prometheus_client import Counter, Gauge, Histogram
    except ImportError:      # metrics are optional; the service works without them
        return None
    _PROM.update(
        requests=Counter("peewee_requests_total", "Decision requests", ["model", "status"]),
        questions=Counter("peewee_questions_total", "Questions answered", ["model", "type"]),
        truncated=Counter("peewee_truncated_total", "Requests whose state was truncated", ["model"]),
        latency=Histogram("peewee_request_seconds", "Wall time per request", ["model"],
                          buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5)),
        batch_size=Histogram("peewee_batch_questions", "Questions per forward pass", ["model"],
                             buckets=(1, 2, 4, 8, 16, 32, 64, 128)),
        forward=Histogram("peewee_forward_seconds", "Wall time per forward pass", ["model"],
                          buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5)),
        queue_depth=Gauge("peewee_queue_depth", "Requests waiting for a batch", ["model"]),
        confidence=Histogram("peewee_confidence", "Reported confidence", ["model", "type"],
                             buckets=(.5, .6, .7, .8, .9, .95, .99, 1.0)),
    )
    return _PROM


class Metrics:
    """Prometheus metrics when prometheus-client is installed; plain counters in a dict always."""

    def __init__(self, prometheus: bool = True):
        self.counts: Dict[str, float] = {}
        coll = _prometheus_collectors() if prometheus else None
        self.enabled = coll is not None
        if coll:
            self.requests, self.questions, self.truncated = coll["requests"], coll["questions"], coll["truncated"]
            self.latency, self.batch_size, self.forward = coll["latency"], coll["batch_size"], coll["forward"]
            self.queue_depth, self.confidence = coll["queue_depth"], coll["confidence"]

    def _bump(self, key: str, n: float = 1):
        self.counts[key] = self.counts.get(key, 0) + n

    def request(self, model: str, status: str, seconds: float):
        if self.enabled:
            self.requests.labels(model, status).inc()
            self.latency.labels(model).observe(seconds)
        self._bump("requests:%s:%s" % (model, status))

    def result(self, model: str, result: Dict[str, Any]):
        for ans in result.get("answers", {}).values():
            if self.enabled:
                self.questions.labels(model, ans["type"]).inc()
                self.confidence.labels(model, ans["type"]).observe(float(ans.get("confidence", 0)))
            self._bump("questions:%s:%s" % (model, ans["type"]))
        if result.get("usage", {}).get("truncated"):
            if self.enabled:
                self.truncated.labels(model).inc()
            self._bump("truncated:%s" % model)

    def batch(self, model: str, n_questions: int, seconds: float):
        if self.enabled:
            self.batch_size.labels(model).observe(n_questions)
            self.forward.labels(model).observe(seconds)
        self._bump("batches:%s" % model)
        self._bump("batched_questions:%s" % model, n_questions)

    def queue(self, model: str, depth: int):
        if self.enabled:
            self.queue_depth.labels(model).set(depth)


# ------------------------------------------------------------------------------ batching
class DynamicBatcher:
    """Collect requests for one agent and answer them in shared forward passes.

    Requests are `(state, questions, truncate)`. While the agent is idle, a flush happens when
    the queued question count reaches `max_batch` or `max_wait` seconds after the first request,
    whichever comes first. While a forward pass is running (up to `max_inflight` of them, one per
    inference worker), arrivals accumulate and go out together the moment a pass finishes, so
    under load the batch grows to whatever queued instead of the timer slicing it into many
    small passes. Inference runs in `executor` so the event loop keeps accepting requests.
    """

    def __init__(self, name: str, agent: Any, executor: ThreadPoolExecutor, max_batch: int = 32,
                 max_wait: float = 0.005, metrics: Optional[Metrics] = None, max_inflight: int = 1):
        self.name = name
        self.agent = agent
        self.executor = executor
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, float(max_wait))
        self.max_inflight = max(1, int(max_inflight))
        self.metrics = metrics or Metrics()
        self._queue: List[Tuple[Any, Dict[str, Any], Optional[str], "asyncio.Future"]] = []
        self._queued_questions = 0
        self._flush_task: Optional[asyncio.Task] = None
        self._inflight = 0
        self._lock = asyncio.Lock()
        self.batches = 0

    @property
    def depth(self) -> int:
        return len(self._queue)

    async def submit(self, state: Any, questions: Dict[str, Any], truncate: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._queue.append((state, questions, truncate, fut))
            self._queued_questions += max(1, len(questions))
            self.metrics.queue(self.name, len(self._queue))
            if self._inflight >= self.max_inflight:
                pass                                # a running pass drains us when it finishes
            elif self._queued_questions >= self.max_batch:
                self._schedule_flush(0.0)
            elif self._flush_task is None:
                self._schedule_flush(self.max_wait)
        return await fut

    def _schedule_flush(self, delay: float):
        if self._flush_task is not None and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.get_running_loop().create_task(self._flush_after(delay))

    async def _flush_after(self, delay: float):
        try:
            if delay > 0:
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        async with self._lock:
            batch, self._queue = self._queue, []
            self._queued_questions = 0
            self._flush_task = None
            self._inflight += 1
            self.metrics.queue(self.name, 0)
        try:
            if batch:
                await self._run(batch)
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._queue and self._flush_task is None:
                    self._schedule_flush(0.0)       # drain what arrived while we were busy

    async def _run(self, batch):
        loop = asyncio.get_running_loop()
        # group by truncation side: predict_many takes one side per call
        by_side: Dict[Optional[str], list] = {}
        for entry in batch:
            by_side.setdefault(entry[2], []).append(entry)
        for side, entries in by_side.items():
            reqs = [(st, qs) for st, qs, _, _ in entries]
            n_q = sum(len(qs) for _, qs in reqs)
            t0 = time.perf_counter()
            try:
                results = await loop.run_in_executor(
                    self.executor, lambda: self.agent.predict_many(reqs, truncate=side, max_batch=self.max_batch))
            except Exception as e:      # one bad request must not strand the others silently
                logger.exception("peewee_decide.serving: forward pass failed for %d request(s) on %r", len(entries), self.name)
                for _, _, _, fut in entries:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self.batches += 1
            self.metrics.batch(self.name, n_q, time.perf_counter() - t0)
            for (_, _, _, fut), res in zip(entries, results):
                if not fut.done():
                    fut.set_result(res)

    async def drain(self):
        """Flush whatever is queued now (used on shutdown and in tests)."""
        async with self._lock:
            batch, self._queue = self._queue, []
            self._queued_questions = 0
            if self._flush_task is not None and not self._flush_task.done():
                self._flush_task.cancel()
            self._flush_task = None
        if batch:
            await self._run(batch)


# ------------------------------------------------------------------------------ service core
class DecisionService:
    """Router + one DynamicBatcher per checkpoint. Framework-agnostic; the FastAPI app wraps it."""

    def __init__(self, router: Any, settings: Optional[Settings] = None, metrics: Optional[Metrics] = None):
        self.router = router
        self.settings = settings or Settings()
        self.metrics = metrics or Metrics()
        self.executor = ThreadPoolExecutor(max_workers=max(1, self.settings.workers), thread_name_prefix="peewee-infer")
        self._batchers: Dict[str, DynamicBatcher] = {}
        self._block = threading.Lock()
        self.started_at = time.time()

    def batcher(self, name: str) -> DynamicBatcher:
        with self._block:
            b = self._batchers.get(name)
            if b is None:
                agent = self.router.load(name)
                b = DynamicBatcher(name, agent, self.executor, self.settings.max_batch,
                                   self.settings.max_wait_ms / 1000.0, self.metrics,
                                   max_inflight=max(1, self.settings.workers))
                self._batchers[name] = b
            return b

    def validate(self, state: Any, questions: Any) -> None:
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object of question_id -> definition")
        for qid, q in questions.items():
            if not isinstance(q, dict) or q.get("type") not in ("choice", "score", "noul"):
                raise ValueError("question %r must have type choice, score or noul" % qid)
            if "instructions" not in q:
                raise ValueError("question %r is missing 'instructions'" % qid)
            if q["type"] in ("choice", "score") and not q.get("criteria"):
                raise ValueError("question %r needs 'criteria'" % qid)
        from .lang import state_text
        if len(state_text(state, max_chars=self.settings.max_state_chars + 1)) > self.settings.max_state_chars:
            raise ValueError("state exceeds PEEWEE_MAX_STATE_CHARS=%d" % self.settings.max_state_chars)

    async def decide(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None,
                     lang: Optional[str] = None, truncate: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.validate(state, questions)
        decision = self.router.route(state, questions, model=model, lang=lang)
        name = decision["model"]
        try:
            result = await self.batcher(name).submit(state, questions, truncate or self.settings.truncate)
        except Exception:
            self.metrics.request(name, "error", time.perf_counter() - t0)
            raise
        result["routing"] = dict(decision)
        self.metrics.request(name, "ok", time.perf_counter() - t0)
        self.metrics.result(name, result)
        return result

    async def decide_many(self, requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        coros = [self.decide(r.get("state"), r.get("questions"), r.get("model"), r.get("lang"), r.get("truncate"))
                 for r in requests]
        return await asyncio.gather(*coros, return_exceptions=True)

    def health(self) -> Dict[str, Any]:
        loaded = list(getattr(self.router, "loaded", []))
        return {
            "status": "ok",
            "loaded": loaded,
            "configured": list(self.settings.models),
            "ready": all(m in loaded for m in self.settings.models),
            "device": self.settings.device or "auto",
            "backend": self.settings.backend,
            "queue_depth": {n: b.depth for n, b in self._batchers.items()},
            "batches": {n: b.batches for n, b in self._batchers.items()},
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    async def shutdown(self):
        for b in list(self._batchers.values()):
            await b.drain()
        self.executor.shutdown(wait=True)


def build_router(settings: Settings):
    """Construct and preload the Router the service will run. Split out so tests can stub it."""
    from .router import Router, normalise_name
    if settings.backend == "onnx":
        from .onnx_backend import OnnxAgent, providers_for_device
        providers = providers_for_device(settings.device)
        router = Router(max_loaded=max(2, len(settings.models)), device=settings.device, token=settings.hf_token)
        names = []
        for spec in settings.models:
            name, _, path = spec.partition("=")
            if not path:
                raise ValueError("PEEWEE_BACKEND=onnx needs PEEWEE_MODELS as name=exported_dir[,name=dir]")
            key = normalise_name(name)
            router.attach(key, OnnxAgent(path, calibration=settings.calibration_for(key), truncate=settings.truncate,
                                         providers=providers))
            names.append(key)
        settings.models = names
        return router
    router = Router(max_loaded=max(2, len(settings.models)), device=settings.device, token=settings.hf_token)
    router.preload(settings.models)
    for name in settings.models:
        cal = settings.calibration_for(normalise_name(name))
        if cal:
            router.load(name).load_calibration(cal)
    return router


# ------------------------------------------------------------------------------ FastAPI app
def create_app(router: Any = None, settings: Optional[Settings] = None):
    """FastAPI application factory. `router` is built from settings when not given."""
    from contextlib import asynccontextmanager

    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse, Response
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    settings = settings or Settings.from_env()
    metrics = Metrics()
    holder: Dict[str, Any] = {"service": None}

    @asynccontextmanager
    async def lifespan(app):
        r = router if router is not None else build_router(settings)
        holder["service"] = DecisionService(r, settings, metrics)
        logger.info("peewee_decide.serving: ready with %s", holder["service"].health()["loaded"])
        try:
            yield
        finally:
            await holder["service"].shutdown()

    app = FastAPI(title="Peewee decision service", version=_version(), lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    async def auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        if settings.api_key and (creds is None or creds.credentials != settings.api_key):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def service() -> DecisionService:
        s = holder["service"]
        if s is None:
            raise HTTPException(status_code=503, detail="service is starting")
        return s

    @app.get("/healthz")
    async def healthz():
        s = holder["service"]
        return s.health() if s else {"status": "starting", "ready": False}

    @app.get("/readyz")
    async def readyz():
        s = holder["service"]
        h = s.health() if s else {"status": "starting", "ready": False}
        return JSONResponse(h, status_code=200 if h.get("ready") else 503)

    @app.get("/metrics")
    async def metrics_endpoint():
        if not metrics.enabled:
            return PlainTextResponse("prometheus-client is not installed\n", status_code=501)
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/decide", dependencies=[Depends(auth)])
    async def decide(body: Dict[str, Any], request: Request):
        s = service()
        try:
            return await s.decide(body.get("state"), body.get("questions"), body.get("model"),
                                  body.get("lang"), body.get("truncate"))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/v1/decide/batch", dependencies=[Depends(auth)])
    async def decide_batch(body: Dict[str, Any]):
        s = service()
        reqs = body.get("requests")
        if not isinstance(reqs, list) or not reqs:
            raise HTTPException(status_code=422, detail="'requests' must be a non-empty list")
        results = await s.decide_many(reqs)
        out = []
        for r in results:
            if isinstance(r, ValueError):
                out.append({"error": str(r), "status": 422})
            elif isinstance(r, Exception):
                logger.error("peewee_decide.serving: batch item failed: %r", r)
                out.append({"error": "inference failed", "status": 500})
            else:
                out.append(r)
        return {"results": out}

    return app


def _version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0"


def main(argv: Optional[List[str]] = None):
    """`peewee serve` / `python -m peewee_decide.serving`."""
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="peewee serve", description="Run the Peewee decision service")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--models", help="comma-separated checkpoints (overrides PEEWEE_MODELS)")
    p.add_argument("--device")
    p.add_argument("--log-level", default=os.environ.get("PEEWEE_LOG_LEVEL", "info"))
    args = p.parse_args(argv)
    env = dict(os.environ)
    if args.models:
        env["PEEWEE_MODELS"] = args.models
    if args.device:
        env["PEEWEE_DEVICE"] = args.device
    if args.host:
        env["PEEWEE_HOST"] = args.host
    if args.port:
        env["PEEWEE_PORT"] = str(args.port)
    os.environ.update({k: v for k, v in env.items() if k.startswith("PEEWEE_")})
    settings = Settings.from_env(env)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run("peewee_decide.serving:create_app", factory=True, host=settings.host, port=settings.port,
                log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
