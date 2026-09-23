"""
Gate policy evaluation helpers (P5 gate study).

Two methodological rules are enforced here:

1. **Thresholds are frozen on dev.**  ``best_dev_threshold`` picks the policy
   cutoff from dev out-of-fold probabilities only; ``eval_policy_stats`` scores
   that frozen cutoff on eval. Eval labels are never used to choose a threshold.

2. **Confidence intervals resample tickets, not rows.**  A ticket contributes
   one row per feedback configuration, so rows of the same ticket are not
   independent. All bootstrap intervals use ``clustered_bootstrap_*``, which
   resample tickets with replacement and keep all rows of a resampled ticket.

``counterfactual_decomposition`` reports *why* a gate helps or hurts: the mean
generated delta of the tickets it closes versus the ones it keeps. The gain of a
gate over always-on is exactly ``-(n_closed / n) * mean_delta_closed``.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)


def grouped_oof_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold P(help) with folds assigned by ticket (no ticket leakage)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {n_groups}")
    proba = np.full(len(y), np.nan)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[train_idx], y[train_idx])
        proba[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return proba


def _cluster_indices(groups: np.ndarray) -> dict:
    return {g: np.where(groups == g)[0] for g in np.unique(groups)}


def clustered_bootstrap_auc(
    y: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """AUC with a 95% CI from resampling tickets with replacement."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    groups = np.asarray(groups)
    point = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    index = _cluster_indices(groups)
    uniq = np.array(list(index.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in sample])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], proba[idx]))
    if not aucs:
        return {"auc": point, "auc_ci_lower": float("nan"), "auc_ci_upper": float("nan"),
                "auc_n_resamples": 0}
    return {"auc": point,
            "auc_ci_lower": float(np.percentile(aucs, 2.5)),
            "auc_ci_upper": float(np.percentile(aucs, 97.5)),
            "auc_n_resamples": len(aucs)}


def clustered_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI of a mean, resampling tickets with replacement."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    index = _cluster_indices(groups)
    uniq = np.array(list(index.keys()), dtype=object)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in sample])
        means.append(float(np.mean(values[idx])))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def policy_value(delta: np.ndarray, proba: np.ndarray, threshold: float) -> float:
    delta = np.asarray(delta, dtype=float)
    proba = np.asarray(proba, dtype=float)
    return float(np.mean(np.where(proba >= threshold, delta, 0.0)))


def best_dev_threshold(
    dev_delta: np.ndarray,
    dev_proba: np.ndarray,
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[float, float]:
    """Threshold maximising the dev policy value (ties resolved to the smaller threshold)."""
    best_t, best_v = float("nan"), -np.inf
    for t in thresholds:
        v = policy_value(dev_delta, dev_proba, t)
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t, best_v


def eval_policy_stats(
    eval_delta: np.ndarray,
    eval_proba: np.ndarray,
    eval_groups: np.ndarray,
    threshold: float,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict:
    """Score a dev-frozen threshold on eval (no eval tuning)."""
    delta = np.asarray(eval_delta, dtype=float)
    proba = np.asarray(eval_proba, dtype=float)
    always = float(np.mean(delta))
    oracle = float(np.mean(np.where(delta > 0, delta, 0.0)))
    values = np.where(proba >= threshold, delta, 0.0)
    policy = float(np.mean(values))
    lo, hi = clustered_bootstrap_mean(values, eval_groups, n_resamples=n_resamples, seed=seed)
    recovery = (policy - always) / (oracle - always) if oracle > always else float("nan")
    return {
        "always_on": always,
        "oracle": oracle,
        "threshold": float(threshold),
        "eval_policy": policy,
        "eval_policy_ci_lower": lo,
        "eval_policy_ci_upper": hi,
        "pct_open": float(np.mean(proba >= threshold)),
        "gain_vs_always_on": policy - always,
        "ceiling_recovery": recovery,
    }


def counterfactual_decomposition(
    delta: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict:
    """Open/closed counterfactual means and harm rates for a frozen threshold."""
    delta = np.asarray(delta, dtype=float)
    proba = np.asarray(proba, dtype=float)
    open_ = proba >= threshold
    closed = ~open_

    def _mean(mask) -> float:
        return float(np.mean(delta[mask])) if mask.any() else float("nan")

    def _harm(mask) -> float:
        return float(np.mean(delta[mask] < 0)) if mask.any() else float("nan")

    policy = float(np.mean(np.where(open_, delta, 0.0)))
    return {
        "n": int(len(delta)),
        "n_open": int(open_.sum()),
        "n_closed": int(closed.sum()),
        "pct_open": float(open_.mean()),
        "mean_delta_all": float(delta.mean()),
        "mean_delta_open": _mean(open_),
        "mean_delta_closed": _mean(closed),
        "harm_rate_all": float(np.mean(delta < 0)),
        "harm_rate_open": _harm(open_),
        "harm_rate_closed": _harm(closed),
        "policy_value": policy,
        "gain_vs_always_on": policy - float(delta.mean()),
    }


def fit_final_gate(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Fit the deployable gate on all dev rows (used for eval predictions)."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    clf.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return clf


# ---------------------------------------------------------------------------
# Magnitude-aware policy (A5): expected-delta regression instead of P(sign)
# ---------------------------------------------------------------------------

def _default_regressor(seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0,
        random_state=seed,
    )


def grouped_oof_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    model_factory=None,
) -> np.ndarray:
    """Out-of-fold expected-delta predictions with folds assigned by ticket."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"need at least {n_splits} groups, got {len(np.unique(groups))}")
    factory = model_factory or (lambda: _default_regressor(seed))
    pred = np.full(len(y), np.nan)
    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    return pred


def fit_final_regressor(X: np.ndarray, y: np.ndarray, seed: int = 42, model_factory=None):
    """Fit the expected-delta regressor on all dev rows (used for eval predictions)."""
    model = (model_factory or (lambda: _default_regressor(seed)))()
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model


def best_dev_tau(
    dev_delta: np.ndarray,
    dev_pred: np.ndarray,
    taus: Sequence[float],
) -> tuple[float, float]:
    """Expected-value cutoff maximising the dev policy value (ties -> smaller tau)."""
    return best_dev_threshold(dev_delta, dev_pred, thresholds=taus)


def regression_quality(y: np.ndarray, pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
    }
