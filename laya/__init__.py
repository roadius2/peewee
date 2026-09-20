"""Laya: Fast, non-autoregressive System 1 decision engine with calibrated probabilities."""

from .agent import Agent, RLAgent, load
from .calibrate import (
    calibration_report,
    collect_records,
    fit_temperature_map,
    target_from_label,
)
from .common import (
    QTYPES,
    QTYPE_NAMES,
    confidence_from_probs,
    ece_score,
    normalized_entropy,
    proper_reward,
    render_options,
    td_lambda_targets,
    top_probability,
)
from .email import clean_email_body, email_state
from .lang import analyse as detect_language
from .lang import detect_script, is_english, register_language_detector
from .patterns import hierarchical_choice, select_tool
from .presets import (
    email_questions,
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router

__version__ = "0.4.0.dev0"
__all__ = [
    "Agent",
    "RLAgent",
    "load",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "detect_language",
    "detect_script",
    "is_english",
    "register_language_detector",
    "collect_records",
    "fit_temperature_map",
    "calibration_report",
    "target_from_label",
    "hierarchical_choice",
    "select_tool",
    "normalized_entropy",
    "top_probability",
    "clean_email_body",
    "email_questions",
    "email_state",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "proper_reward",
    "td_lambda_targets",
    "ece_score",
    "confidence_from_probs",
    "render_options",
    "QTYPES",
    "QTYPE_NAMES",
    "__version__",
]
