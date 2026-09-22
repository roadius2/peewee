"""Minimal client for the Peewee decision service. Standard library only.

    from peewee_decide.client import PeeweeClient
    c = PeeweeClient("http://localhost:8000", api_key=None)
    c.decide({"body": "billed twice"}, questions)["answers"]["department"]["choice"]
    c.decide_many([{"state": s1, "questions": q}, {"state": s2, "questions": q}])
    c.health()
"""
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence


class PeeweeServiceError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__("Peewee service returned %d: %s" % (status, detail))
        self.status = status
        self.detail = detail


class PeeweeClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None,
                 timeout: float = 30.0, opener=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        try:
            with self._open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8"))
            except Exception:
                detail = e.reason
            raise PeeweeServiceError(e.code, detail.get("detail", detail) if isinstance(detail, dict) else detail)

    def health(self) -> Dict[str, Any]:
        return self._call("GET", "/healthz")

    def decide(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None,
               lang: Optional[str] = None, truncate: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"state": state, "questions": questions}
        if model:
            body["model"] = model
        if lang:
            body["lang"] = lang
        if truncate:
            body["truncate"] = truncate
        return self._call("POST", "/v1/decide", body)

    def decide_many(self, requests: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._call("POST", "/v1/decide/batch", {"requests": list(requests)})["results"]
