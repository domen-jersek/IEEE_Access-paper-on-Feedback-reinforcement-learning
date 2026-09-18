"""
Consistency tests for the P0-P3 additions:
  * legacy routings are numerically unchanged (candidate_lift == old _routing_score + compute_lift)
  * backoff / blend behave as specified
  * laplace_eb centring
  * vectorised PoolTensor lifts == scalar lifts
  * BM25 / RRF retrievers return aligned, self-excluded pools
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LiftConfig, RoutingConfig
from src.feedback.lift import compute_lift
from src.feedback.loader import aggregate_feedback_scores, FeedbackBundle
from src.retrieval.search import candidate_lift, _routing_score, per_scope_lifts, scope_evidence
from src.evaluation.ladder import PoolTensor, SCOPE_IDX


def _cand(g=(3, 1), c=(2, 0), t=(0, 2), i=(1, 0)):
    return {
        "global": {"pos": g[0], "neg": g[1]},
        "class:X": {"pos": c[0], "neg": c[1]},
        "team:Y": {"pos": t[0], "neg": t[1]},
        "intersection:X:Y": {"pos": i[0], "neg": i[1]},
    }


@pytest.mark.parametrize("routing", [RoutingConfig.global_(), RoutingConfig.team_only(),
                                     RoutingConfig.class_only(), RoutingConfig.intersection()])
def test_legacy_routing_unchanged(routing):
    cfg = LiftConfig.laplace()
    cand = _cand()
    pos, neg = _routing_score("", cand, "X", "Y", routing)
    old = compute_lift(pos, neg, cfg)
    new, _ = candidate_lift(cand, "X", "Y", routing, cfg)
    assert new == pytest.approx(old)


def test_backoff_picks_narrowest_with_evidence():
    cfg = LiftConfig.laplace()
    cand = _cand(i=(1, 0), t=(4, 0), c=(0, 0), g=(10, 10))
    lift, scope = candidate_lift(cand, "X", "Y", RoutingConfig.backoff(min_evidence=3), cfg)
    assert scope == "team" and lift == pytest.approx(compute_lift(4, 0, cfg))
    lift, scope = candidate_lift(cand, "X", "Y", RoutingConfig.backoff(min_evidence=1), cfg)
    assert scope == "intersection"
    lift, scope = candidate_lift(_cand(g=(0, 0), c=(0, 0), t=(0, 0), i=(0, 0)), "X", "Y", RoutingConfig.backoff(), cfg)
    assert lift == 0.0 and scope == "none"


def test_blend_is_weighted_sum_of_scope_lifts():
    cfg = LiftConfig.laplace()
    cand = _cand()
    ls = per_scope_lifts(cand, "X", "Y", cfg)
    r = RoutingConfig.blend(w_global=0.1, w_class=0.2, w_team=0.3, w_intersection=0.4)
    lift, _ = candidate_lift(cand, "X", "Y", r, cfg)
    assert lift == pytest.approx(0.1 * ls["global"] + 0.2 * ls["class"] + 0.3 * ls["team"] + 0.4 * ls["intersection"])


def test_laplace_eb_centering():
    cfg = LiftConfig.laplace_eb(prior_strength=2.0)
    # average evidence at the prior mean -> zero lift
    assert compute_lift(2.3, 7.7, cfg, prior_mean=0.23) == pytest.approx(0.0, abs=1e-12)
    # with prior 0.5 it equals laplace without the min(1, n/2) shrink (n >= 2 -> identical)
    assert compute_lift(3, 1, cfg, prior_mean=0.5) == pytest.approx(compute_lift(3, 1, LiftConfig.laplace()))
    # evidence above the empirical mean is positive even if below 0.5
    assert compute_lift(4, 6, cfg, prior_mean=0.23) > 0
    assert compute_lift(4, 6, LiftConfig.laplace()) < 0


def test_to_dict_hash_stable_for_defaults():
    from src.config import DEFAULT_METHODS
    d = DEFAULT_METHODS["M2_team"].to_dict()
    assert "scale_mode" not in d["lift"] and "w_intersection" not in d["routing"] and "retriever" not in d
    d2 = LiftConfig.laplace_eb().to_dict()
    assert d2["name"] == "laplace_eb" and "prior_strength" not in d2  # default kappa not emitted
    assert "scale_mode" in LiftConfig(name="laplace", scale_mode="pool_std").to_dict()


def test_pool_tensor_matches_scalar_lifts():
    cand = _cand()
    cfg_l, cfg_eb = LiftConfig.laplace(), LiftConfig.laplace_eb(5.0)
    ev = scope_evidence(cand, "X", "Y")
    pos = np.array([[[ev[s]["pos"] for s in ("global", "class", "team", "intersection")]]])
    neg = np.array([[[ev[s]["neg"] for s in ("global", "class", "team", "intersection")]]])
    prior = np.array([[0.2, 0.3, 0.4, 0.5]])
    T = PoolTensor(["q"], np.array([["c"]], dtype=object), np.array([[0.5]]), np.array([[True]]), pos, neg, prior,
                   {"minilm": np.zeros((1, 1))}, np.zeros((1, 1)), np.zeros((1, 1)), ["X"], ["Y"])
    L = T.lifts(cfg_l)[0, 0]
    for j, s in enumerate(("global", "class", "team", "intersection")):
        assert L[j] == pytest.approx(compute_lift(ev[s]["pos"], ev[s]["neg"], cfg_l))
    Leb = T.lifts(cfg_eb)[0, 0]
    for j, s in enumerate(("global", "class", "team", "intersection")):
        assert Leb[j] == pytest.approx(compute_lift(ev[s]["pos"], ev[s]["neg"], cfg_eb, prior_mean=prior[0, j]))
    for routing in (RoutingConfig.global_(), RoutingConfig.team_only(), RoutingConfig.intersection(),
                    RoutingConfig.backoff(min_evidence=2), RoutingConfig.blend(0.1, 0.2, 0.3, 0.4)):
        assert T.routed_lift(cfg_l, routing)[0, 0] == pytest.approx(candidate_lift(cand, "X", "Y", routing, cfg_l)[0])


def test_bundle_minus_query_subtracts_only_that_query():
    rows = pd.DataFrame({
        "query_id": ["q1", "q1", "q2"], "candidate_id": ["c", "d", "c"],
        "query_class": ["X", "X", "X"], "query_team": ["Y", "Y", "Z"], "score": [1.0, 0.0, 0.5],
    })
    b = FeedbackBundle(rows)
    assert b.scores["c"]["global"]["pos"] == pytest.approx(1.5)
    m = b.minus_query("q1")
    assert m["c"]["global"]["pos"] == pytest.approx(0.5) and m["c"]["global"]["neg"] == pytest.approx(0.5)
    assert m["c"]["team:Y"]["pos"] == pytest.approx(0.0)
    assert m["d"]["global"]["neg"] == pytest.approx(0.0)
    assert b.scores["c"]["global"]["pos"] == pytest.approx(1.5)  # original untouched
    assert b.prior_mean("global") == pytest.approx(0.5)
    assert b.prior_mean("team:Y") == pytest.approx(0.5)
    assert b.prior_mean("team:missing") == pytest.approx(0.5)  # falls back to global


def test_bm25_and_rrf_pools_are_aligned_and_self_excluded():
    pytest.importorskip("rank_bm25")
    from src.retrieval.retrievers import BM25Retriever, HybridRRFRetriever
    meta = pd.DataFrame({
        "seq_id": ["R-1", "R-2", "R-3", "R-4"],
        "Title_anon": ["vpn access request", "new laptop hardware", "vpn token reset", "printer not working"],
        "Description_anon": ["need vpn", "laptop broken", "token expired vpn", ""],
        "first_reply": ["a", "b", "c", "d"],
    })
    bm = BM25Retriever(meta)
    pool = bm.search_pool("token reset", None, 3, exclude_idxs={0})
    assert 0 not in set(pool["_faiss_idx"]) and list(pool["faiss_rank"]) == [1, 2, 3]
    assert pool.iloc[0]["seq_id"] == "R-3"

    class Fake(BM25Retriever):
        pass
    rrf = HybridRRFRetriever([bm, Fake(meta)], k_rrf=60, pool_each=4)
    p2 = rrf.search_pool("token reset", None, 2, exclude_idxs={0})
    assert 0 not in set(p2["_faiss_idx"]) and p2.iloc[0]["faiss_score"] == pytest.approx(2 / 61)
