"""Tests for configuration-conditioned gate features (P5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EvalConfig, GatingConfig, LiftConfig, RoutingConfig
from src.evaluation.ladder import PoolTensor
from src.gate.pool_features import config_pool_features, gate_features_from_pool, single_query_tensor
from src.retrieval.search import apply_feedback_to_pool


def _tensor():
    Q, K = 2, 4
    cand = np.array([["c0", "c1", "c2", "c3"], ["c0", "c1", "c2", "c3"]], dtype=object)
    score = np.array([[0.90, 0.80, 0.70, 0.60], [0.55, 0.50, 0.45, 0.40]])
    mask = np.ones((Q, K), dtype=bool)
    pos = np.zeros((Q, K, 4)); neg = np.zeros((Q, K, 4))
    # candidate 0 strongly positive in global scope, others zero
    pos[0, 0, 0] = 8.0; neg[0, 0, 0] = 1.0
    pos[1, 2, 0] = 5.0; neg[1, 2, 0] = 0.0
    prior = np.full((Q, 4), 0.3)
    semantic = np.array([[0.95, 0.90, 0.85, 0.80], [0.60, 0.58, 0.56, 0.54]])
    return PoolTensor(["q0", "q1"], cand, score, mask, pos, neg, prior,
                      {"minilm": np.zeros((Q, K))}, np.zeros((Q, K)), np.zeros((Q, K)),
                      ["X", "X"], ["Y", "Y"], semantic)


def test_feature_shape_and_columns():
    T = _tensor()
    f = config_pool_features(T, LiftConfig.laplace(), RoutingConfig.global_(), T.baseline_top(5), k=3)
    assert len(f) == 2
    for col in ("top1_score", "lift_max", "n_promotable", "score_lift_corr",
                "semantic_top1", "max_lift_global", "coverage_global"):
        assert col in f.columns
    assert len(f.columns) >= 25


def test_zero_lift_has_no_promotion():
    T = _tensor()
    f = config_pool_features(T, LiftConfig(name="none", cap=0.0), RoutingConfig(name="none"), T.baseline_top(5), k=3)
    assert (f["n_promotable"] == 0).all()
    assert (f["n_top_changed"] == 0).all()
    assert np.allclose(f["lift_max"], 0.0)


def test_promotion_margin_reflects_lift():
    T = _tensor()
    f = config_pool_features(T, LiftConfig.laplace(), RoutingConfig.global_(), T.baseline_top(5), k=3)
    assert f.loc[0, "lift_max"] > 0
    assert f.loc[0, "n_promotable"] >= 1


def test_semantic_features_present_only_with_semantic():
    T = _tensor()
    f = config_pool_features(T, LiftConfig.laplace(), RoutingConfig.global_(), T.baseline_top(5), k=3)
    assert "semantic_mean" in f.columns
    T2 = PoolTensor(T.query_ids, T.cand_ids, T.score, T.mask, T.pos, T.neg, T.prior,
                    T.useful, T.same_reply, T.same_group, T.query_class, T.query_team, None)
    f2 = config_pool_features(T2, LiftConfig.laplace(), RoutingConfig.global_(), T2.baseline_top(5), k=3)
    assert "semantic_mean" not in f2.columns


def _live_pool():
    pool = pd.DataFrame({"seq_id": ["c0", "c1", "c2", "c3"],
                         "faiss_score": [0.9, 0.8, 0.7, 0.6],
                         "faiss_rank": [1, 2, 3, 4]})
    fb = {"c0": {"global": {"pos": 8.0, "neg": 1.0}}}
    sem = np.array([0.95, 0.9, 0.85, 0.8])
    return pool, fb, sem


def test_live_features_match_tensor_features():
    pool, fb, sem = _live_pool()
    lcfg, routing = LiftConfig.laplace(), RoutingConfig.global_()
    T = single_query_tensor(pool, "q0", "X", "Y", fb, None, sem)
    tensor_feat = config_pool_features(T, lcfg, routing, T.baseline_top(5), k=5).iloc[0]
    live = gate_features_from_pool(pool, lcfg, routing, "q0", "X", "Y", fb, None,
                                   list(tensor_feat.index), semantic_sims=sem)
    assert np.allclose(live, tensor_feat.to_numpy(float), equal_nan=True)


class _StubGate:
    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, X):
        return np.array([[1 - self.proba, self.proba]])


def test_learned_gate_closes_pool_when_probability_low():
    pool, fb, sem = _live_pool()
    cfg = EvalConfig(experiment_id="t", lift=LiftConfig.laplace(), routing=RoutingConfig.global_(),
                     gating=GatingConfig.learned(), generator_model="x", judge_model="x")
    gate = {"model": _StubGate(0.1), "features": ["lift_max", "n_promotable"], "threshold": 0.5}
    out = apply_feedback_to_pool(pool, fb, cfg, "X", "Y", None, semantic_sims=sem, gate=gate, query_id="q0")
    assert out["gate_active"].all()
    assert np.allclose(out["feedback_lift"], 0.0)
    assert np.allclose(out["enhanced_score"], out["faiss_score"])


def test_learned_gate_keeps_pool_when_probability_high():
    pool, fb, sem = _live_pool()
    cfg = EvalConfig(experiment_id="t", lift=LiftConfig.laplace(), routing=RoutingConfig.global_(),
                     gating=GatingConfig.learned(), generator_model="x", judge_model="x")
    gate = {"model": _StubGate(0.9), "features": ["lift_max", "n_promotable"], "threshold": 0.5}
    out = apply_feedback_to_pool(pool, fb, cfg, "X", "Y", None, semantic_sims=sem, gate=gate, query_id="q0")
    assert not out["gate_active"].any()
    assert out["feedback_lift"].abs().sum() > 0