"""
Learned gate model training with nested cross-validation.

Uses XGBoost classifier (primary) and logistic regression (baseline comparison)
with nested CV: inner 3-fold for hyperparameter tuning, outer 5-fold for evaluation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def train_evaluate_gate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    n_outer: int = 5,
    n_inner: int = 3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scaling is fitted INSIDE each outer fold (no leakage from the test fold).
    X_scaled = np.asarray(X, dtype=float)

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)

    xgb_param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
    }

    lr_param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    }

    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except ImportError:
        log.warning("XGBoost not available, using logistic regression only")
        xgb_available = False

    results = {"outer_folds": [], "feature_importance": {}, "lr_coefficients": {}}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_scaled, y)):
        scaler = StandardScaler().fit(X_scaled[train_idx])
        X_tr, X_te = scaler.transform(X_scaled[train_idx]), scaler.transform(X_scaled[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        fold_models = {}

        lr_gs = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_param_grid, cv=inner_cv, scoring="roc_auc")
        lr_gs.fit(X_tr, y_tr)
        lr_best = lr_gs.best_estimator_
        lr_pred = lr_best.predict(X_te)
        lr_proba = lr_best.predict_proba(X_te)[:, 1]
        fold_models["logistic"] = {
            "accuracy": float(accuracy_score(y_te, lr_pred)),
            "auc": float(roc_auc_score(y_te, lr_proba)),
            "best_params": lr_gs.best_params_,
        }
        for k, v in zip(feature_names, lr_best.coef_[0].tolist()):
            results["lr_coefficients"][k] = results["lr_coefficients"].get(k, 0.0) + v / n_outer

        if xgb_available:
            xgb_gs = GridSearchCV(XGBClassifier(random_state=42, eval_metric="logloss"), xgb_param_grid, cv=inner_cv, scoring="roc_auc")
            xgb_gs.fit(X_tr, y_tr)
            xgb_best = xgb_gs.best_estimator_
            xgb_pred = xgb_best.predict(X_te)
            xgb_proba = xgb_best.predict_proba(X_te)[:, 1]
            fold_models["xgboost"] = {
                "accuracy": float(accuracy_score(y_te, xgb_pred)),
                "auc": float(roc_auc_score(y_te, xgb_proba)),
                "best_params": xgb_gs.best_params_,
            }
            importance = dict(zip(feature_names, xgb_best.feature_importances_.tolist()))
            for k, v in importance.items():
                results["feature_importance"][k] = results["feature_importance"].get(k, 0.0) + v / n_outer

        results["outer_folds"].append({
            "fold": fold_idx,
            "train_n": int(len(train_idx)),
            "test_n": int(len(test_idx)),
            "models": fold_models,
        })

    agg = {}
    for model_name in ["logistic"] + (["xgboost"] if xgb_available else []):
        accs = [f["models"][model_name]["accuracy"] for f in results["outer_folds"]]
        aucs = [f["models"][model_name]["auc"] for f in results["outer_folds"]]
        agg[model_name] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
        }

    results["aggregate"] = agg
    results["n_samples"] = int(len(y))
    results["n_features"] = int(X.shape[1])
    results["feature_names"] = feature_names
    results["class_balance"] = float(np.mean(y))

    out_path = output_dir / "gate_cv_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    importance_path = output_dir / "feature_importance.json"
    sorted_imp = sorted(results["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    importance_path.write_text(json.dumps(sorted_imp, indent=2), encoding="utf-8")

    log.info("Gate CV results: LR AUC=%.3f±%.3f", agg["logistic"]["auc_mean"], agg["logistic"]["auc_std"])
    if xgb_available:
        log.info("Gate CV results: XGB AUC=%.3f±%.3f", agg["xgboost"]["auc_mean"], agg["xgboost"]["auc_std"])

    return results


def fit_gate_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str = "xgboost",
) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8,
            random_state=42, eval_metric="logloss",
        )
    elif model_type == "logistic":
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_scaled, y)
    return model, scaler


def predict_gate(
    model, scaler, X: np.ndarray, decision_threshold: float = 0.5
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    return (proba >= decision_threshold).astype(int)


def gate_policy_value(delta: np.ndarray, proba: np.ndarray, thresholds: Optional[list[float]] = None) -> pd.DataFrame:
    """
    Value of a gate policy: apply feedback only when P(improve) >= t.
    Returns mean delta under the policy for each threshold, plus the
    always-on / never-on references and the oracle (apply iff delta > 0).
    """
    thresholds = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = [{"policy": "always_on", "threshold": None, "mean_delta": float(delta.mean()), "pct_open": 1.0},
            {"policy": "never_on", "threshold": None, "mean_delta": 0.0, "pct_open": 0.0},
            {"policy": "oracle", "threshold": None, "mean_delta": float(np.where(delta > 0, delta, 0).mean()), "pct_open": float(np.mean(delta > 0))}]
    for t in thresholds:
        open_ = proba >= t
        rows.append({"policy": "learned", "threshold": t, "mean_delta": float(np.where(open_, delta, 0).mean()), "pct_open": float(open_.mean())})
    return pd.DataFrame(rows)