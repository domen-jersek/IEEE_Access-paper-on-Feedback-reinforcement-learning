"""
Pre-generation feature extraction for the learned gate.

Features must be computable BEFORE generation (no oracle access to reference reply).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def extract_gate_features(
    baseline_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    feedback_scores: dict,
    query_info: dict,
) -> dict[str, float]:
    f = {}

    f["top1_faiss"] = float(baseline_df["faiss_score"].iloc[0]) if len(baseline_df) > 0 else 0.0
    f["top5_mean_faiss"] = float(baseline_df["faiss_score"].mean()) if len(baseline_df) > 0 else 0.0
    f["top5_min_faiss"] = float(baseline_df["faiss_score"].min()) if len(baseline_df) > 0 else 0.0
    f["top5_std_faiss"] = float(baseline_df["faiss_score"].std()) if len(baseline_df) > 1 else 0.0
    f["retrieval_margin"] = (
        float(baseline_df["faiss_score"].iloc[0] - baseline_df["faiss_score"].iloc[1])
        if len(baseline_df) > 1
        else 0.0
    )
    f["top5_spread"] = f["top5_mean_faiss"] - f["top1_faiss"]

    bl_ids = set(baseline_df["seq_id"])
    fb_ids = set(feedback_df["seq_id"])
    f["retrieval_overlap"] = float(len(bl_ids & fb_ids))

    rng = feedback_df if len(feedback_df) > 0 else baseline_df.head(5)
    lifts = rng["feedback_lift"].tolist() if "feedback_lift" in rng.columns else [0.0] * 5
    lifts = [float(l) for l in lifts]
    f["max_lift"] = max(lifts) if lifts else 0.0
    f["mean_lift"] = float(np.mean(lifts)) if lifts else 0.0
    f["n_positive_lifts"] = float(sum(1 for l in lifts if l > 0))
    f["n_negative_lifts"] = float(sum(1 for l in lifts if l < 0))
    f["lift_conflict"] = 1.0 if (f["n_positive_lifts"] > 0 and f["n_negative_lifts"] > 0) else 0.0

    evidence_count = 0.0
    for cid in rng["seq_id"]:
        cdata = feedback_scores.get(str(cid), {})
        g = cdata.get("global", {"pos": 0, "neg": 0})
        evidence_count += g["pos"] + g["neg"]
    f["evidence_density"] = evidence_count

    f["query_desc_len"] = float(len(str(query_info.get("description", ""))))
    f["query_title_len"] = float(len(str(query_info.get("title", ""))))

    return f


def extract_feature_matrix(
    all_results: list[dict],
    feedback_scores: dict,
    teams: list[str],
    classes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_list = []
    targets = []
    feature_names = None

    for r in all_results:
        bl_df = pd.DataFrame(r.get("baseline", {}).get("retrieval", []))
        fb_df = pd.DataFrame(r.get("feedback", {}).get("retrieval", []))
        if len(bl_df) == 0:
            continue

        feat = extract_gate_features(
            bl_df, fb_df, feedback_scores,
            {
                "title": r.get("query_title", ""),
                "description": r.get("query_description", ""),
            },
        )

        query_team = r.get("expected_team", "Unknown")
        query_class = r.get("expected_class", "other")
        for t in teams:
            feat[f"team_{t}"] = 1.0 if t == query_team else 0.0
        for c in classes:
            feat[f"class_{c}"] = 1.0 if c == query_class else 0.0

        if feature_names is None:
            feature_names = sorted(feat.keys())

        feature_vec = [feat.get(k, 0.0) for k in feature_names]
        feature_list.append(feature_vec)
        targets.append(float(r["deltas"]["delta_cosine"] > 0))

    X = np.array(feature_list)
    y = np.array(targets)
    return X, y, feature_names


def features_from_config(
    baseline_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    config,
    query_info: dict,
) -> dict[str, float]:
    return extract_gate_features(baseline_df, feedback_df, {}, query_info)