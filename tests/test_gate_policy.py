"""Gate policy helpers: grouped CV, dev-frozen thresholds, clustered bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gate.policy import (
    best_dev_tau,
    best_dev_threshold,
    clustered_bootstrap_auc,
    clustered_bootstrap_mean,
    counterfactual_decomposition,
    eval_policy_stats,
    grouped_oof_proba,
    grouped_oof_regression,
    policy_value,
    regression_quality,
)


def _group_dependent_dataset(n_groups: int = 10, rows_per_group: int = 20):
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    y = (groups < n_groups // 2).astype(int)
    X = np.zeros((len(groups), n_groups))
    X[np.arange(len(groups)), groups] = 1.0
    return X, y, groups


def test_grouped_oof_prevents_ticket_leakage():
    X, y, groups = _group_dependent_dataset()
    proba = grouped_oof_proba(X, y, groups, n_splits=5)
    assert not np.isnan(proba).any()
    auc = clustered_bootstrap_auc(y, proba, groups, n_resamples=200)["auc"]
    # With ticket-level folds the group->label mapping is unlearnable: AUC ~ chance.
    assert abs(auc - 0.5) < 0.05


def test_grouped_oof_rejects_too_few_groups():
    X, y, groups = _group_dependent_dataset(n_groups=3, rows_per_group=5)
    try:
        grouped_oof_proba(X, y, groups, n_splits=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError for fewer groups than folds")


def test_best_dev_threshold_is_dev_frozen():
    dev_delta = np.array([0.10, 0.10, 0.10, -0.10, -0.10, -0.10])
    dev_proba = np.array([0.9, 0.8, 0.65, 0.35, 0.30, 0.20])
    t, v = best_dev_threshold(dev_delta, dev_proba, thresholds=(0.4, 0.7))
    assert t == 0.4 and abs(v - 0.05) < 1e-12  # opens the three positives only

    # On eval, a higher threshold would look better; the frozen 0.4 must be used.
    ev_delta = np.array([0.10, 0.10, 0.10, -0.10, -0.10, -0.10])
    ev_proba = np.array([0.9, 0.8, 0.65, 0.60, 0.50, 0.45])
    stats = eval_policy_stats(ev_delta, ev_proba, np.arange(6), threshold=t, n_resamples=100)
    assert stats["threshold"] == 0.4
    assert abs(stats["eval_policy"] - policy_value(ev_delta, ev_proba, 0.4)) < 1e-12
    assert abs(stats["eval_policy"] - 0.0) < 1e-12
    assert stats["pct_open"] == 1.0


def test_counterfactual_decomposition_math():
    delta = np.array([0.20, 0.10, -0.30, 0.0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    d = counterfactual_decomposition(delta, proba, threshold=0.5)
    assert d["n_open"] == 2 and d["n_closed"] == 2
    assert abs(d["mean_delta_closed"] - (-0.15)) < 1e-12
    assert abs(d["harm_rate_closed"] - 0.5) < 1e-12
    assert abs(d["gain_vs_always_on"] - (-(2 / 4) * (-0.15))) < 1e-12


def test_clustered_bootstrap_mean_ci_contains_point():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 3)
    values = rng.normal(0.5, 1.0, size=len(groups))
    lo, hi = clustered_bootstrap_mean(values, groups, n_resamples=300)
    assert lo <= values.mean() <= hi


def test_grouped_oof_regression_shape_and_finiteness():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(20), 10)
    X = rng.normal(size=(len(groups), 3))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.1, size=len(groups))
    pred = grouped_oof_regression(X, y, groups, n_splits=5)
    assert len(pred) == len(y) and np.isfinite(pred).all()
    assert regression_quality(y, pred)["r2"] > 0.5


def test_best_dev_tau_prefers_positive_cutoff_when_needed():
    dev_delta = np.array([0.10, 0.10, 0.10, -0.10, -0.10, -0.10])
    dev_pred = np.array([0.10, 0.10, 0.02, 0.01, -0.02, -0.03])
    tau, val = best_dev_tau(dev_delta, dev_pred, taus=[-0.05, 0.0, 0.015, 0.05])
    assert tau == 0.015
    assert abs(val - 0.05) < 1e-12
