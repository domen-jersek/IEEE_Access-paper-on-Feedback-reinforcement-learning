"""Tests for the P5 semantic relevance filter (scalar and vectorised paths)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LiftConfig, RoutingConfig
from src.feedback.lift import compute_lift
from src.retrieval.search import candidate_lift, per_scope_lifts
from src.evaluation.ladder import PoolTensor


def _cand(g=(3, 1), c=(2, 0), t=(0, 2), i=(1, 0)):
    return {
        "global": {"pos": g[0], "neg": g[1]},
        "class:X": {"pos": c[0], "neg": c[1]},
        "team:Y": {"pos": t[0], "neg": t[1]},
        "intersection:X:Y": {"pos": i[0], "neg": i[1]},
    }


def _tensor(cand, semantic_value, prior=(0.2, 0.3, 0.4, 0.5)):
    from src.retrieval.search import scope_evidence
    ev = scope_evidence(cand, "X", "Y")
    order = ("global", "class", "team", "intersection")
    pos = np.array([[[ev[s]["pos"] for s in order]]])
    neg = np.array([[[ev[s]["neg"] for s in order]]])
    T = PoolTensor(["q"], np.array([["c"]], dtype=object), np.array([[0.5]]), np.array([[True]]),
                   pos, neg, np.array([list(prior)]), {"minilm": np.zeros((1, 1))},
                   np.zeros((1, 1)), np.zeros((1, 1)), ["X"], ["Y"],
                   semantic=np.array([[semantic_value]]))
    return T


def test_semantic_factories():
    assert RoutingConfig.semantic_intersection(0.7).semantic_tau == 0.7
    assert RoutingConfig.semantic_backoff(0.6, min_evidence=2).name == "semantic_backoff"


def test_semantic_intersection_filters_below_tau():
    cfg = LiftConfig.laplace()
    cand = _cand()
    routing = RoutingConfig.semantic_intersection(tau=0.7)
    kept, scope = candidate_lift(cand, "X", "Y", routing, cfg, semantic_sim=0.8)
    assert kept == pytest.approx(per_scope_lifts(cand, "X", "Y", cfg)["intersection"])
    assert scope == "intersection"
    filtered, scope = candidate_lift(cand, "X", "Y", routing, cfg, semantic_sim=0.5)
    assert filtered == 0.0 and scope == "semantic_filtered"


def test_semantic_filter_inactive_without_similarity():
    cfg = LiftConfig.laplace()
    routing = RoutingConfig.semantic_intersection(tau=0.99)
    lift, scope = candidate_lift(_cand(), "X", "Y", routing, cfg, semantic_sim=None)
    assert scope == "intersection" and lift != 0.0


def test_semantic_backoff_filters_then_backs_off():
    cfg = LiftConfig.laplace()
    cand = _cand(i=(1, 0), t=(4, 0), c=(0, 0), g=(10, 10))
    routing = RoutingConfig.semantic_backoff(tau=0.7, min_evidence=3)
    lift, scope = candidate_lift(cand, "X", "Y", routing, cfg, semantic_sim=0.9)
    assert scope == "team" and lift == pytest.approx(compute_lift(4, 0, cfg))
    lift, scope = candidate_lift(cand, "X", "Y", routing, cfg, semantic_sim=0.2)
    assert lift == 0.0 and scope == "semantic_filtered"


@pytest.mark.parametrize("semantic_value", [0.5, 0.9])
def test_pool_tensor_matches_scalar_semantic(semantic_value):
    cfg = LiftConfig.laplace()
    cand = _cand(i=(1, 0), t=(4, 0), c=(0, 0), g=(10, 10))
    T = _tensor(cand, semantic_value)
    for routing in (RoutingConfig.semantic_intersection(tau=0.7),
                    RoutingConfig.semantic_backoff(tau=0.7, min_evidence=3)):
        vec = T.routed_lift(cfg, routing)[0, 0]
        scalar, _ = candidate_lift(cand, "X", "Y", routing, cfg, semantic_sim=semantic_value)
        assert vec == pytest.approx(scalar)


def test_pool_tensor_semantic_default_none_does_not_filter():
    cfg = LiftConfig.laplace()
    cand = _cand()
    ev_order = ("global", "class", "team", "intersection")
    from src.retrieval.search import scope_evidence
    ev = scope_evidence(cand, "X", "Y")
    pos = np.array([[[ev[s]["pos"] for s in ev_order]]])
    neg = np.array([[[ev[s]["neg"] for s in ev_order]]])
    T = PoolTensor(["q"], np.array([["c"]], dtype=object), np.array([[0.5]]), np.array([[True]]),
                   pos, neg, np.array([[0.2, 0.3, 0.4, 0.5]]), {"minilm": np.zeros((1, 1))},
                   np.zeros((1, 1)), np.zeros((1, 1)), ["X"], ["Y"])
    assert T.semantic is None
    vec = T.routed_lift(cfg, RoutingConfig.semantic_intersection(tau=0.99))[0, 0]
    assert vec == pytest.approx(per_scope_lifts(cand, "X", "Y", cfg)["intersection"])