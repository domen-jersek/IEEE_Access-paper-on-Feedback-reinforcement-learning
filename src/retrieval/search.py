"""
Retrieval orchestrator: baseline (FAISS) vs feedback-enhanced retrieval.
Handles candidate scoring, lift application, and top-k selection.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import EvalConfig
from ..feedback.lift import compute_lift as _compute_lift_raw

log = logging.getLogger(__name__)


def _routing_score(
    candidate_id: str,
    candidate_data: dict,
    query_class: str,
    query_team: str,
    routing,
) -> tuple[float, float]:
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


def retrieve_baseline(
    faiss_index,
    query_embedding: np.ndarray,
    top_k: int,
    exclude_idxs: set,
) -> pd.DataFrame:
    scores, indices = faiss_index.search(query_embedding, top_k, exclude_idxs)
    results = faiss_index.get_metadata(indices[0])
    results["faiss_score"] = scores[0]
    results["faiss_rank"] = range(1, len(results) + 1)
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
) -> pd.DataFrame:
    scores_all, indices_all = faiss_index.search(query_embedding, search_k, exclude_idxs)

    results = faiss_index.get_metadata(indices_all[0])
    results["faiss_score"] = scores_all[0]

    lifts = []
    for _, row in results.iterrows():
        pos, neg = _routing_score(
            str(row["seq_id"]),
            feedback_scores.get(str(row["seq_id"]), {}),
            query_class,
            query_team,
            config.routing,
        )
        lift = _compute_lift_raw(pos, neg, config.lift)
        lifts.append(lift)
    results["feedback_lift"] = lifts
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
    elif gating_cfg.name == "learned":
        results["gate_active"] = None
    else:
        results["gate_active"] = False

    results = results.sort_values("enhanced_score", ascending=False).head(top_k)
    results = results.reset_index(drop=True)
    results["feedback_rank"] = range(1, len(results) + 1)
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