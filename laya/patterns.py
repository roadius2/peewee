"""Composition patterns over `predict`: things a single question cannot do well.

Everything here works with anything that has `.predict(state, questions, **kw)`, so an
`Agent` and a `Router` are interchangeable.
"""
from typing import Any, Dict, Mapping, Optional, Sequence, Union

Options = Union[Mapping[str, Optional[str]], Sequence[str]]


def _as_dict(options: Options) -> Dict[str, Optional[str]]:
    if isinstance(options, Mapping):
        return dict(options)
    return {o: None for o in options}


def hierarchical_choice(
    model: Any,
    state: Any,
    groups: Mapping[str, Options],
    instructions: str,
    group_descriptions: Optional[Mapping[str, str]] = None,
    group_instructions: Optional[str] = None,
    question_id: str = "choice",
    **predict_kw,
) -> Dict[str, Any]:
    """Two-stage choice for large option sets: pick a group, then pick within it.

    At the default head budget a single `choice` question gives 77 options three or four
    tokens each, and accuracy collapses (Banking77: 0.425). Splitting the options into groups
    keeps every option readable. Two forward passes instead of one.

    `groups` maps a group name to its options (a dict of option -> description, or a list).
    `group_descriptions` optionally describes each group for the first stage. The result is
    the leaf choice plus both stage results; `confidence` is the product of the two stages'
    top probabilities, which is the probability of the full path under the model.
    """
    if not groups:
        raise ValueError("groups must not be empty")
    group_descriptions = dict(group_descriptions or {})
    group_q = {question_id: {
        "type": "choice",
        "instructions": group_instructions or instructions,
        "criteria": {g: group_descriptions.get(g, ", ".join(list(_as_dict(opts))[:8])) for g, opts in groups.items()},
    }}
    stage1 = model.predict(state, group_q, **predict_kw)
    a1 = stage1["answers"][question_id]
    group = a1["choice"]
    members = _as_dict(groups[group])
    if len(members) == 1:
        leaf = next(iter(members))
        return {"choice": leaf, "group": group, "confidence": a1["confidence"],
                "probabilities": {leaf: 1.0}, "stages": [stage1, None]}
    leaf_q = {question_id: {"type": "choice", "instructions": instructions, "criteria": members}}
    stage2 = model.predict(state, leaf_q, **predict_kw)
    a2 = stage2["answers"][question_id]
    return {
        "choice": a2["choice"],
        "group": group,
        "confidence": round(float(a1["confidence"]) * float(a2["confidence"]), 4),
        "probabilities": a2["probabilities"],
        "group_probabilities": a1["probabilities"],
        "stages": [stage1, stage2],
    }


def select_tool(model: Any, state: Any, tools: Mapping[str, Mapping[str, Any]],
                instructions: str = "Which tool should be called next to make progress on `task`?",
                max_flat: int = 16, **predict_kw) -> Dict[str, Any]:
    """Pick one tool out of `tools`, going hierarchical automatically above `max_flat` tools.

    `tools` maps a tool name to a dict with at least `description`, and optionally `group`.
    With `max_flat` or fewer tools this is a single `choice`; above that, tools are bucketed
    by their `group` (tools without one go to "other") and `hierarchical_choice` is used.
    """
    if len(tools) <= max_flat:
        q = {"tool": {"type": "choice", "instructions": instructions,
                      "criteria": {n: t.get("description") for n, t in tools.items()}}}
        res = model.predict(state, q, **predict_kw)
        a = res["answers"]["tool"]
        return {"choice": a["choice"], "group": None, "confidence": a["confidence"],
                "probabilities": a["probabilities"], "stages": [res]}
    groups: Dict[str, Dict[str, Optional[str]]] = {}
    for name, t in tools.items():
        groups.setdefault(t.get("group") or "other", {})[name] = t.get("description")
    return hierarchical_choice(model, state, groups, instructions, question_id="tool", **predict_kw)
