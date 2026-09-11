"""
Lift formula implementations. Single source of truth for all retrieval-lift math.

Given positive/negative feedback counts (pos, neg) for a candidate under a
routing scope, return a bounded lift to be added to the FAISS similarity.
"""
from __future__ import annotations

import math

from ..config import LiftConfig


def laplace(pos: float, neg: float, cfg: LiftConfig) -> float:
    p = (pos + cfg.alpha) / (pos + neg + cfg.alpha + cfg.beta)
    n = pos + neg
    scale = min(1.0, n / 2.0)
    return (p - 0.5) * scale * cfg.multiplier


def tanh(pos: float, neg: float, cfg: LiftConfig) -> float:
    evidence = (pos - neg) * 0.5
    return math.tanh(evidence / cfg.sensitivity) * cfg.cap


def bayesian_lcb(pos: float, neg: float, cfg: LiftConfig) -> float:
    p = (pos + 1.0) / (pos + neg + 2.0)
    n = pos + neg
    if n <= 0:
        return 0.0
    sigma = math.sqrt(p * (1.0 - p) / (n + 1.0))
    lcb = (p - 0.5) - cfg.lcb_k * sigma
    return lcb * min(1.0, n / 2.0) * cfg.multiplier


_REGISTRY = {
    "laplace": laplace,
    "tanh": tanh,
    "bayesian_lcb": bayesian_lcb,
}


def compute_lift(pos: float, neg: float, cfg: LiftConfig) -> float:
    if cfg.name == "none" or (pos == 0 and neg == 0):
        return 0.0
    fn = _REGISTRY.get(cfg.name)
    if fn is None:
        raise ValueError(f"Unknown lift formula: {cfg.name}")
    lift = fn(pos, neg, cfg)
    if cfg.positive_only and lift < 0:
        lift = 0.0
    return max(-cfg.cap, min(cfg.cap, lift))


def raw_popularity(pos: float, neg: float) -> float:
    """(pos - neg) / (pos + neg + 1) — used for orthogonality analysis, not reranking."""
    return (pos - neg) / (pos + neg + 1.0)