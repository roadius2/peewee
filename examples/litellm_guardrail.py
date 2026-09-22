"""Use the Peewee decision service as a LiteLLM guardrail and model router.

Two hooks, both calling the service over HTTP so every proxy worker shares one model:

  * `PeeweeGuardrail`  -- pre-call guardrail: scores the last user message with the guard preset
                        and blocks jailbreaks / injections above a threshold.
  * `peewee_pick_model` -- a plain function you can call from a custom router or a pre-call hook
                        to pick a deployment from Peewee's difficulty / domain / sensitivity
                        answers, so the frontier model is only used when it is needed.

Install: pip install litellm peewee-decide      (the service itself runs separately: `peewee serve`)

litellm proxy config.yaml:
    guardrails:
      - guardrail_name: peewee
        litellm_params:
          guardrail: examples.litellm_guardrail.PeeweeGuardrail
          mode: pre_call
          service_url: http://peewee:8000
          block_threshold: 0.8
"""
from typing import Any, Dict, Optional

from peewee_decide.client import PeeweeClient
from peewee_decide.presets import guard_questions, router_questions

try:  # litellm is optional here so the module imports without it
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pragma: no cover
    CustomGuardrail = object  # type: ignore


def _last_user_text(data: Dict[str, Any]) -> str:
    for m in reversed(data.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


class PeeweeGuardrail(CustomGuardrail):  # type: ignore[misc]
    """Block prompts Peewee scores as jailbreak or injection above `block_threshold`."""

    def __init__(self, service_url: str = "http://localhost:8000", api_key: Optional[str] = None,
                 block_threshold: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.client = PeeweeClient(service_url, api_key=api_key)
        self.block_threshold = float(block_threshold)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type: str):
        prompt = _last_user_text(data)
        if not prompt:
            return data
        res = self.client.decide({"prompt": prompt}, guard_questions(), truncate="left")
        a = res["answers"]
        worst = max(a["jailbreak"]["noul"], a["prompt_injection"]["noul"])
        if worst >= self.block_threshold:
            raise ValueError("blocked by peewee guardrail: jailbreak=%.2f injection=%.2f harm=%s" % (
                a["jailbreak"]["noul"], a["prompt_injection"]["noul"], a["harm_severity"]["level"]))
        # annotate for downstream routing / logging
        data.setdefault("metadata", {})["peewee_guard"] = {k: v.get("noul", v.get("score")) for k, v in a.items()}
        return data


def peewee_pick_model(client: PeeweeClient, request_text: str,
                    tiers: Dict[str, str] = None) -> Dict[str, Any]:
    """Map Peewee's routing answers to a deployment name.

    `tiers` maps "small" / "reasoning" / "frontier" to your LiteLLM model names. The rule is
    deliberately simple and worth tuning on your own traffic: frontier for hard or sensitive
    requests, reasoning for moderate difficulty or math/code, small otherwise.
    """
    tiers = tiers or {"small": "qwen-small", "reasoning": "qwen-reasoning", "frontier": "claude"}
    res = client.decide({"request": request_text}, router_questions())
    a = res["answers"]
    level = a["difficulty"]["level"]
    domain = a["domain"]["choice"]
    sensitive = a["is_sensitive"]["noul"] >= 0.7
    if level >= 3 or sensitive:
        tier = "frontier"
    elif level == 2 or domain in ("math_or_logic", "code"):
        tier = "reasoning"
    else:
        tier = "small"
    return {"model": tiers[tier], "tier": tier, "answers": a, "usage": res["usage"], "routing": res["routing"]}
