"""
Configuration-conditioned, pre-generation gate features (P5 gate study).

Given a retrieval pool and a feedback *configuration* (routing + lift + scaling),
compute per-query features that describe how the intervention would reshape the
ranking. These features are available before generation and are independent of
the generated answer, so a gate trained on them can be applied to new tickets.

Three families:
  * pool distribution   - lift/score distribution over the whole candidate pool,
                          promotion potential (how much the pool can move).
  * cosine correlation  - query<->candidate semantic cosine statistics and the
                          agreement between retrieval score and feedback lift.
  * routing/evidence    - per-scope evidence, coverage and maximum lift.

No reference-reply information is used (the offline proxy is deliberately
excluded here); `proxy_*` columns are added separately in the study script and
are flagged as oracle-diagnostic only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LiftConfig, RoutingConfig
from ..evaluation.ladder import PoolTensor, SCOPE_IDX
from ..retrieval.search import SCOPES, scope_key


def _row_pearson(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = np.where(mask, x, np.nan)
    y = np.where(mask, y, np.nan)
    xm = np.nanmean(x, axis=1, keepdims=True)
    ym = np.nanmean(y, axis=1, keepdims=True)
    xc, yc = x - xm, y - ym
    num = np.nansum(xc * yc, axis=1)
    den = np.sqrt(np.nansum(xc ** 2, axis=1) * np.nansum(yc ** 2, axis=1))
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def config_pool_features(T: PoolTensor, lcfg: LiftConfig, routing: RoutingConfig,
                         base_top: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Per-query features for one (routing, lift, scaling) configuration."""
    Q = len(T.query_ids)
    rows = np.arange(Q)[:, None]
    mask = T.mask
    lift = T.routed_lift(lcfg, routing)
    factor = T.scale_factor(lcfg)
    top = T.rerank(lift, factor, k)

    l = np.where(mask, lift, 0.0)
    sc = np.where(mask, T.score, np.nan)
    b5 = T.score[rows, base_top]
    b1 = b5[:, 0]
    enhanced = T.score + lift * factor[:, None]
    enh = np.where(mask, enhanced, -np.inf)

    n_promotable = (enh > b1[:, None]).sum(axis=1)
    best_margin = (np.where(mask, enhanced - b1[:, None], -np.inf)).max(axis=1)
    base_aligned = base_top[:, :top.shape[1]]
    n_changed = (top != base_aligned).sum(axis=1)

    valid = mask.sum(axis=1)
    lmean = l.sum(axis=1) / np.maximum(valid, 1)
    lstd = np.sqrt(np.maximum((np.where(mask, (l - lmean[:, None]) ** 2, 0.0)).sum(axis=1) / np.maximum(valid, 1), 0.0))

    f = {
        "top1_score": b1,
        "top5_mean_score": b5.mean(axis=1),
        "top5_min_score": b5.min(axis=1),
        "top5_std_score": b5.std(axis=1),
        "retrieval_margin": b1 - b5[:, 1],
        "pool_std": np.nanstd(sc, axis=1),
        "lift_max": lift.max(axis=1),
        "lift_mean": lmean,
        "lift_std": lstd,
        "lift_min": l.min(axis=1),
        "lift_frac_positive": (l > 0).sum(axis=1) / np.maximum(valid, 1),
        "lift_frac_negative": (l < 0).sum(axis=1) / np.maximum(valid, 1),
        "n_promotable": n_promotable,
        "best_promotion_margin": best_margin,
        "n_top_changed": n_changed,
        "score_lift_corr": _row_pearson(np.where(mask, T.score, 0.0), np.where(mask, lift, 0.0), mask),
        "intervention_scale": factor,
        "n_candidates": valid,
    }
    for s in SCOPES:
        j = SCOPE_IDX[s]
        f[f"evidence_{s}_top5"] = T.n[rows, base_top][..., j].sum(axis=1)
        f[f"coverage_{s}"] = ((T.n[..., j] > 0) & mask).sum(axis=1) / np.maximum(valid, 1)
        f[f"max_lift_{s}"] = T.lifts(lcfg)[..., j].max(axis=1)
    if T.semantic is not None:
        sem = np.where(mask, T.semantic, np.nan)
        f["semantic_top1"] = T.semantic[rows, base_top][:, 0]
        f["semantic_mean"] = np.nanmean(sem, axis=1)
        f["semantic_std"] = np.nanstd(sem, axis=1)
        f["semantic_lift_corr"] = _row_pearson(np.nan_to_num(sem, nan=0.0), np.where(mask, lift, 0.0), mask)
    return pd.DataFrame(f)


# ---------------------------------------------------------------------------
# Live (single-query) path: build a 1-row PoolTensor from a retrieval pool and
# reuse `config_pool_features` so the inference-time features are identical to
# the training-time features. This is what the evaluation pipeline calls when a
# learned gate is enabled.
# ---------------------------------------------------------------------------

def _prior_mean(priors, key: str) -> float:
    if priors is None:
        return 0.5
    if hasattr(priors, "prior_mean"):
        return float(priors.prior_mean(key))
    p = priors.get(key) or priors.get("global")
    if p is None:
        return 0.5
    return float(p["mean"]) if isinstance(p, dict) else float(p)


def single_query_tensor(pool_df: pd.DataFrame, query_id: str, query_class: str, query_team: str,
                        feedback_scores: dict, priors=None, semantic_sims=None) -> PoolTensor:
    """1-row PoolTensor from a retrieval pool (columns: seq_id, faiss_score, faiss_rank)."""
    pool = pool_df.sort_values("faiss_rank") if "faiss_rank" in pool_df.columns else pool_df
    K = len(pool)
    cand = pool["seq_id"].astype(str).to_numpy().reshape(1, K)
    score = pool["faiss_score"].to_numpy(float).reshape(1, K)
    mask = np.ones((1, K), dtype=bool)
    pos = np.zeros((1, K, 4))
    neg = np.zeros((1, K, 4))
    prior = np.full((1, 4), 0.5)
    for j, s in enumerate(SCOPES):
        prior[0, j] = _prior_mean(priors, scope_key(s, query_class, query_team))
    for ci, cid in enumerate(cand[0]):
        cdata = feedback_scores.get(str(cid), {})
        for j, s in enumerate(SCOPES):
            e = cdata.get(scope_key(s, query_class, query_team))
            if e:
                pos[0, ci, j] = e.get("pos", 0.0)
                neg[0, ci, j] = e.get("neg", 0.0)
    semantic = None
    if semantic_sims is not None:
        semantic = np.asarray(semantic_sims, dtype=float).reshape(1, K)
    zeros = np.zeros((1, K))
    return PoolTensor([query_id], cand, score, mask, pos, neg, prior,
                      {"minilm": zeros}, zeros.copy(), zeros.copy(),
                      [query_class], [query_team], semantic)


def gate_features_from_pool(pool_df: pd.DataFrame, lift_cfg: LiftConfig, routing: RoutingConfig,
                            query_id: str, query_class: str, query_team: str, feedback_scores: dict,
                            priors, feature_names: list[str], semantic_sims=None, k: int = 5) -> np.ndarray:
    """Feature vector (in `feature_names` order) for the live gating decision."""
    T = single_query_tensor(pool_df, query_id, query_class, query_team, feedback_scores, priors, semantic_sims)
    base = T.baseline_top(k)
    f = config_pool_features(T, lift_cfg, routing, base, k=k)
    missing = [c for c in feature_names if c not in f.columns]
    if missing:
        raise ValueError(f"Gate features missing from the live pool: {missing}")
    return f.iloc[0][feature_names].to_numpy(float)