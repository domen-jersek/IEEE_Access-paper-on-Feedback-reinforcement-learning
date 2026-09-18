"""
Lift formula implementations. Single source of truth for all retrieval-lift math.

Given positive/negative feedback counts (pos, neg) for a candidate under a
routing scope, return a bounded lift to be added to the retrieval similarity.

Formulas
--------
laplace      (SIKDD)  p = (pos+a)/(n+a+b); lift = (p - 0.5) * min(1, n/2) * m
tanh                  lift = tanh((pos-neg)/2 / sensitivity) * cap
bayesian_lcb          lower confidence bound variant of laplace
laplace_eb   (P1.5)   empirical-Bayes centred Laplace:
                        a = kappa * p_bar,  b = kappa * (1 - p_bar)
                        p_post = (pos + a) / (n + kappa)
                        lift   = (p_post - p_bar) * m
                      where p_bar is the empirical mean judge score of the scope
                      (passed in as `prior_mean`; defaults to 0.5, in which case the
                      formula reduces to laplace without the min(1, n/2) shrink).
                      With p_bar != 0.5 the neutral point moves to the population
                      mean, so "no evidence" (lift 0) and "average evidence" coincide
                      instead of "no evidence" outranking every judged candidate.
"""
from __future__ import annotations

import math
from typing import Optional

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


def laplace_eb(pos: float, neg: float, cfg: LiftConfig, prior_mean: float = 0.5) -> float:
    n = pos + neg
    kappa = max(cfg.prior_strength, 1e-9)
    p_bar = min(max(prior_mean, 1e-6), 1.0 - 1e-6)
    a = kappa * p_bar
    p_post = (pos + a) / (n + kappa)
    return (p_post - p_bar) * cfg.multiplier


_REGISTRY = {
    "laplace": laplace,
    "tanh": tanh,
    "bayesian_lcb": bayesian_lcb,
}
_REGISTRY_WITH_PRIOR = {
    "laplace_eb": laplace_eb,
}


def compute_lift(pos: float, neg: float, cfg: LiftConfig, prior_mean: Optional[float] = None) -> float:
    """
    Bounded lift in [-cap, cap]. `prior_mean` is only used by prior-aware formulas
    (laplace_eb); legacy formulas ignore it, so existing results are unaffected.
    """
    if cfg.name == "none" or (pos == 0 and neg == 0):
        return 0.0
    fn = _REGISTRY.get(cfg.name)
    if fn is not None:
        lift = fn(pos, neg, cfg)
    else:
        fn_p = _REGISTRY_WITH_PRIOR.get(cfg.name)
        if fn_p is None:
            raise ValueError(f"Unknown lift formula: {cfg.name}")
        lift = fn_p(pos, neg, cfg, 0.5 if prior_mean is None else prior_mean)
    if cfg.positive_only and lift < 0:
        lift = 0.0
    return max(-cfg.cap, min(cfg.cap, lift))


def raw_popularity(pos: float, neg: float) -> float:
    """(pos - neg) / (pos + neg + 1) — used for orthogonality analysis, not reranking."""
    return (pos - neg) / (pos + neg + 1.0)
