"""
Retrieval orchestrator: baseline vs feedback-enhanced retrieval.
Handles candidate scoring, lift application, and top-k selection.

Backward compatibility
----------------------
`retrieve_baseline` / `retrieve_feedback` accept either a `FAISSIndex` (legacy,
exactly the SIKDD code path) or any `BaseRetriever` from `retrievers.py`.
With a FAISSIndex + legacy routing (global / categorical / categorical_intersection)
+ legacy lift (laplace / tanh / bayesian_lcb) + scale_mode="absolute" the numerics
are identical to the already-completed replication runs; the only differences are
additional output columns (faiss_rank on the feedback side, per-scope evidence).

New in P1.5 / P3
----------------
* `priors`               : per-scope empirical means used by `laplace_eb`
* scale_mode="pool_std"  : lift expressed in units of the pool's retrieval-score std
* routing "backoff"      : narrowest scope with n >= min_evidence
* routing "blend"        : weighted sum of per-scope lifts
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import EvalConfig, LiftConfig, RoutingConfig
from ..feedback.lift import compute_lift as _compute_lift_raw
from .index import FAISSIndex

log = logging.getLogger(__name__)

SCOPES = ("global", "class", "team", "intersection")


# ---------------------------------------------------------------------------
# Scope evidence helpers
# ---------------------------------------------------------------------------

def scope_key(scope: str, query_class: str, query_team: str) -> str:
    if scope == "global":
        return "global"
    if scope == "class":
        return f"class:{query_class}"
    if scope == "team":
        return f"team:{query_team}"
    if scope == "intersection":
        return f"intersection:{query_class}:{query_team}"
    raise ValueError(scope)


def scope_evidence(candidate_data: dict, query_class: str, query_team: str) -> dict[str, dict[str, float]]:
    """{scope: {"pos", "neg", "n"}} for the four scopes of this query."""
    out = {}
    for s in SCOPES:
        e = candidate_data.get(scope_key(s, query_class, query_team), {"pos": 0.0, "neg": 0.0})
        pos, neg = float(e.get("pos", 0.0)), float(e.get("neg", 0.0))
        out[s] = {"pos": pos, "neg": neg, "n": pos + neg}
    return out


def _prior_for(priors, key: str) -> Optional[float]:
    if priors is None:
        return None
    if hasattr(priors, "prior_mean"):
        return priors.prior_mean(key)
    p = priors.get(key) or priors.get("global")
    if p is None:
        return None
    return float(p["mean"]) if isinstance(p, dict) else float(p)


def _legacy_prior_scope(routing: RoutingConfig, query_class: str, query_team: str) -> str:
    if routing.name == "global":
        return "global"
    if routing.name == "categorical_intersection":
        return scope_key("intersection", query_class, query_team)
    if routing.name == "categorical":
        if routing.w_team > 0 and routing.w_class == 0 and routing.w_global == 0:
            return scope_key("team", query_class, query_team)
        if routing.w_class > 0 and routing.w_team == 0 and routing.w_global == 0:
            return scope_key("class", query_class, query_team)
    return "global"


def _routing_score(
    candidate_id: str,
    candidate_data: dict,
    query_class: str,
    query_team: str,
    routing,
) -> tuple[float, float]:
    """Legacy (SIKDD) aggregation into a single (pos, neg) pair."""
    pos, neg = 0.0, 0.0
    if routing.name == "global" or routing.w_global > 0:
        g = candidate_data.get("global", {"pos": 0, "neg": 0})
        pos += routing.w_global * g["pos"]
        neg += routing.w_global * g["neg"]
    if routing.w_class > 0 and routing.name == "categorical":
        c = candidate_data.get(f"class:{query_class}", {"pos": 0, "neg": 0})
        pos += routing.w_class * c["pos"]
        neg += routing.w_class * c["neg"]
    if routing.w_team > 0 and routing.name == "categorical":
        t = candidate_data.get(f"team:{query_team}", {"pos": 0, "neg": 0})
        pos += routing.w_team * t["pos"]
        neg += routing.w_team * t["neg"]
    if routing.name == "categorical_intersection":
        key = f"intersection:{query_class}:{query_team}"
        isec = candidate_data.get(key, {"pos": 0, "neg": 0})
        pos, neg = isec["pos"], isec["neg"]
    return pos, neg


def per_scope_lifts(candidate_data: dict, query_class: str, query_team: str,
                    lift_cfg: LiftConfig, priors=None) -> dict[str, float]:
    """Lift of every scope computed independently (used by backoff / blend / records)."""
    ev = scope_evidence(candidate_data, query_class, query_team)
    out = {}
    for s in SCOPES:
        out[s] = _compute_lift_raw(ev[s]["pos"], ev[s]["neg"], lift_cfg,
                                   _prior_for(priors, scope_key(s, query_class, query_team)))
    return out


def candidate_lift(candidate_data: dict, query_class: str, query_team: str,
                   routing: RoutingConfig, lift_cfg: LiftConfig, priors=None,
                   semantic_sim: Optional[float] = None) -> tuple[float, str]:
    """
    Returns (lift, scope_used).
    Legacy routings reproduce the SIKDD numerics exactly (single pos/neg -> lift).
    `semantic_sim` (query<->candidate ticket-text cosine) is only consulted by the
    `semantic_intersection` / `semantic_backoff` routings; when it is None the lift
    is applied unfiltered so callers without an embedding store still work.
    """
    if routing.name in ("global", "categorical", "categorical_intersection"):
        pos, neg = _routing_score("", candidate_data, query_class, query_team, routing)
        prior = _prior_for(priors, _legacy_prior_scope(routing, query_class, query_team))
        return _compute_lift_raw(pos, neg, lift_cfg, prior), routing.name
    if routing.name == "backoff":
        ev = scope_evidence(candidate_data, query_class, query_team)
        for s in routing.backoff_order:
            if ev[s]["n"] >= routing.min_evidence:
                prior = _prior_for(priors, scope_key(s, query_class, query_team))
                return _compute_lift_raw(ev[s]["pos"], ev[s]["neg"], lift_cfg, prior), s
        return 0.0, "none"
    if routing.name == "semantic_intersection":
        if semantic_sim is not None and semantic_sim < routing.semantic_tau:
            return 0.0, "semantic_filtered"
        ev = scope_evidence(candidate_data, query_class, query_team)["intersection"]
        prior = _prior_for(priors, scope_key("intersection", query_class, query_team))
        return _compute_lift_raw(ev["pos"], ev["neg"], lift_cfg, prior), "intersection"
    if routing.name == "semantic_backoff":
        if semantic_sim is not None and semantic_sim < routing.semantic_tau:
            return 0.0, "semantic_filtered"
        ev = scope_evidence(candidate_data, query_class, query_team)
        for s in routing.backoff_order:
            if ev[s]["n"] >= routing.min_evidence:
                prior = _prior_for(priors, scope_key(s, query_class, query_team))
                return _compute_lift_raw(ev[s]["pos"], ev[s]["neg"], lift_cfg, prior), s
        return 0.0, "none"
    if routing.name == "blend":
        lifts = per_scope_lifts(candidate_data, query_class, query_team, lift_cfg, priors)
        w = {"global": routing.w_global, "class": routing.w_class,
             "team": routing.w_team, "intersection": routing.w_intersection}
        return float(sum(w[s] * lifts[s] for s in SCOPES)), "blend"
    if routing.name == "none":
        return 0.0, "none"
    raise ValueError(f"Unknown routing: {routing.name}")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _search(source, query_text: Optional[str], query_embedding: np.ndarray, k: int, exclude_idxs: set) -> pd.DataFrame:
    """Dispatch to FAISSIndex (legacy) or a BaseRetriever."""
    if isinstance(source, FAISSIndex):
        scores, indices = source.search(query_embedding, k, exclude_idxs)
        results = source.get_metadata(indices[0])
        results["faiss_score"] = scores[0]
        results["faiss_rank"] = range(1, len(results) + 1)
        return results
    return source.search_pool(query_text or "", query_embedding, k, exclude_idxs)


def retrieve_baseline(
    faiss_index,
    query_embedding: np.ndarray,
    top_k: int,
    exclude_idxs: set,
    query_text: Optional[str] = None,
) -> pd.DataFrame:
    return _search(faiss_index, query_text, query_embedding, top_k, exclude_idxs)


def apply_feedback_to_pool(
    pool: pd.DataFrame,
    feedback_scores: dict,
    config: EvalConfig,
    query_class: str,
    query_team: str,
    priors=None,
    semantic_sims: Optional[np.ndarray] = None,
    gate: Optional[dict] = None,
    query_id: Optional[str] = None,
) -> pd.DataFrame:
    """Add feedback_lift_raw / lift_scope / feedback_lift / enhanced_score / gate columns to a pool.

    `semantic_sims` (optional, aligned with `pool` rows) feeds the semantic
    relevance filter and the gate's cosine features; when omitted those are inactive.
    `gate` (optional dict with keys model/features/threshold) enables the learned
    control policy: when P(help) < threshold the pool keeps its baseline ranking.
    """
    results = pool.reset_index(drop=True)

    lifts, scopes = [], []
    for i, (_, row) in enumerate(results.iterrows()):
        sim = float(semantic_sims[i]) if semantic_sims is not None else None
        lift, scope_used = candidate_lift(
            feedback_scores.get(str(row["seq_id"]), {}),
            query_class, query_team, config.routing, config.lift, priors,
            semantic_sim=sim,
        )
        lifts.append(lift)
        scopes.append(scope_used)
    results["feedback_lift_raw"] = lifts
    results["lift_scope"] = scopes

    # Scale the bounded lift into the retriever's score units if requested.
    if config.lift.scale_mode == "pool_std" and config.lift.cap > 0:
        pool_std = float(results["faiss_score"].std()) if len(results) > 1 else 0.0
        factor = (config.lift.pool_lambda * pool_std) / config.lift.cap
        results["feedback_lift"] = results["feedback_lift_raw"] * factor
        results.attrs["lift_scale_factor"] = factor
    else:
        results["feedback_lift"] = results["feedback_lift_raw"]
        results.attrs["lift_scale_factor"] = 1.0

    results["enhanced_score"] = results["faiss_score"] + results["feedback_lift"]

    gating_cfg = config.gating
    if gating_cfg.name == "static":
        top1_faiss = results["faiss_score"].iloc[0]
        if top1_faiss >= gating_cfg.faiss_ceiling:
            results["enhanced_score"] = results["faiss_score"]
            results["feedback_lift"] = 0.0
            results["gate_active"] = True
            results["gate_reason"] = f"static_ceiling_{gating_cfg.faiss_ceiling}"
        else:
            results["gate_active"] = False
    elif gating_cfg.name == "learned" and gate is not None and query_id is not None:
        from ..gate.pool_features import gate_features_from_pool
        x = gate_features_from_pool(results, config.lift, config.routing, query_id, query_class, query_team,
                                    feedback_scores, priors, gate["features"], semantic_sims=semantic_sims)
        proba = float(gate["model"].predict_proba(x.reshape(1, -1))[0, 1])
        results["gate_proba"] = proba
        if proba < float(gate["threshold"]):
            results["feedback_lift"] = 0.0
            results["enhanced_score"] = results["faiss_score"]
            results["gate_active"] = True
            results["gate_reason"] = f"learned_p{proba:.3f}_below_{gate['threshold']:.2f}"
        else:
            results["gate_active"] = False
    elif gating_cfg.name == "learned":
        results["gate_active"] = None
    else:
        results["gate_active"] = False
    return results


def retrieve_feedback(
    faiss_index,
    query_embedding: np.ndarray,
    top_k: int,
    exclude_idxs: set,
    feedback_scores: dict,
    config: EvalConfig,
    query_class: str,
    query_team: str,
    search_k: int = 100,
    query_text: Optional[str] = None,
    priors=None,
    return_pool: bool = False,
    semantic_sims: Optional[np.ndarray] = None,
    text_sim=None,
    query_id: Optional[str] = None,
    gate: Optional[dict] = None,
):
    pool = _search(faiss_index, query_text, query_embedding, search_k, exclude_idxs)
    if semantic_sims is None and text_sim is not None and query_id is not None:
        semantic_sims = np.array([text_sim.s(query_id, str(cid)) for cid in pool["seq_id"]], dtype=float)
    results = apply_feedback_to_pool(pool, feedback_scores, config, query_class, query_team, priors,
                                     semantic_sims=semantic_sims, gate=gate, query_id=query_id)

    pool = results
    # NOTE: default (non-stable) sort kept on purpose — identical tie-ordering to the SIKDD runs.
    results = results.sort_values("enhanced_score", ascending=False).head(top_k)
    results = results.reset_index(drop=True)
    results["feedback_rank"] = range(1, len(results) + 1)
    if return_pool:
        return results, pool
    return results


def compute_overlap(baseline_df: pd.DataFrame, feedback_df: pd.DataFrame) -> int:
    bl_ids = set(baseline_df["seq_id"])
    fb_ids = set(feedback_df["seq_id"])
    return len(bl_ids & fb_ids)


def compute_retrieval_pool_features(
    baseline_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> dict:
    top1_faiss = float(baseline_df["faiss_score"].iloc[0])
    top5_faiss_mean = float(baseline_df["faiss_score"].mean())
    retrieval_margin = float(baseline_df["faiss_score"].iloc[0] - baseline_df["faiss_score"].iloc[1]) if len(baseline_df) > 1 else 0.0
    top5_spread = float(baseline_df["faiss_score"].max() - baseline_df["faiss_score"].min())

    return {
        "top1_faiss": top1_faiss,
        "top5_faiss_mean": top5_faiss_mean,
        "retrieval_margin": retrieval_margin,
        "top5_spread": top5_spread,
    }


def pool_feedback_coverage(pool: pd.DataFrame, feedback_scores: dict, query_class: str, query_team: str) -> dict:
    """Fraction of pool candidates with any evidence, per scope (P2 coverage confound)."""
    out = {}
    n = max(len(pool), 1)
    for s in SCOPES:
        key = scope_key(s, query_class, query_team)
        cnt = 0
        for cid in pool["seq_id"]:
            e = feedback_scores.get(str(cid), {}).get(key)
            if e and (e.get("pos", 0) + e.get("neg", 0)) > 0:
                cnt += 1
        out[s] = cnt / n
    return out
