#!/usr/bin/env python3
"""
16_gate_pilot.py — P5 learned-gate pilot (dev -> eval)
======================================================
Trains a *generalizable* pre-generation gate on generated-answer labels from one
or more feedback runs, with honest out-of-fold probabilities, and reports:

  * out-of-fold AUC for logistic and (if installed) XGBoost
  * policy value: always-on vs never-on vs oracle vs learned thresholds
  * ceiling recovery = (learned - always_on) / (oracle - always_on)
  * a simple pool-distribution rule sweep (top1 similarity, max lift) as the
    interpretable fallback when the learned gate is weak

Features exclude team/class one-hot so the gate is dataset-independent. The
label is the generated delta_cosine > 0 from the supplied *_details.json.

Pilot has no eval labels yet: train/CV on dev. Pass --eval-details later (once an
eval run exists) to confirm without retraining on eval.

Outputs: results/gate_pilot_<tag>/{gate_pilot.json, policy.csv, rules.csv}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.utils import (setup_logging, load_dataset, get_arg_parser, write_run_manifest,
                               append_registry, make_run_id)
from src.config import ProjectPaths
from src.gate.features import extract_feature_matrix
from src.gate.model import gate_policy_value
from src.feedback.loader import load_feedback_as_scores

DATASET_SPECIFIC_PREFIXES = ("team_", "class_")


def _newest_details(folder: Path) -> Path:
    files = sorted(folder.glob("*_details.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no *_details.json in {folder}")
    return files[-1]


def load_run(path: str) -> tuple[list[dict], Path]:
    p = Path(path)
    if p.is_dir():
        p = _newest_details(p)
    return json.loads(p.read_text(encoding="utf-8")), p


def generalizable_matrix(results: list[dict], feedback_scores: dict, teams, classes):
    X, y, names = extract_feature_matrix(results, feedback_scores, teams, classes)
    keep = [i for i, n in enumerate(names) if not n.startswith(DATASET_SPECIFIC_PREFIXES)]
    return X[:, keep], y, [names[i] for i in keep]


def oof_proba(X: np.ndarray, y: np.ndarray, model: str) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if model == "xgboost":
        from xgboost import XGBClassifier
        clf = make_pipeline(StandardScaler(), XGBClassifier(learning_rate=0.05, max_depth=3,
                                                            n_estimators=100, subsample=0.8,
                                                            random_state=42, eval_metric="logloss"))
    else:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]


def ceiling_recovery(policy: pd.DataFrame) -> float:
    always = float(policy.loc[policy["policy"] == "always_on", "mean_delta"].iloc[0])
    oracle = float(policy.loc[policy["policy"] == "oracle", "mean_delta"].iloc[0])
    learned = policy[policy["policy"] == "learned"]["mean_delta"].max()
    if not np.isfinite(learned) or (oracle - always) <= 0:
        return float("nan")
    return float((learned - always) / (oracle - always))


def rule_sweep(features: pd.DataFrame, delta: np.ndarray) -> pd.DataFrame:
    rows = []
    for col, direction in (("top1_faiss", "below"), ("max_lift", "above")):
        if col not in features:
            continue
        vals = features[col].to_numpy(float)
        for q in (0.2, 0.4, 0.6, 0.8):
            t = float(np.quantile(vals, q))
            open_ = vals <= t if direction == "below" else vals >= t
            rows.append({"rule": f"{col}_{direction}_{t:.3f}", "feature": col, "threshold": t,
                         "pct_open": float(open_.mean()), "mean_delta": float(np.where(open_, delta, 0).mean())})
    return pd.DataFrame(rows)


def evaluate(name: str, results: list[dict], feedback_scores: dict, teams, classes,
             model: str, log) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    X, y, names = generalizable_matrix(results, feedback_scores, teams, classes)
    delta = np.array([r["deltas"]["delta_cosine"] for r in results if r.get("deltas")], dtype=float)
    n = min(len(X), len(delta))
    X, y, delta = X[:n], y[:n], delta[:n]
    proba = oof_proba(X, y, model)
    auc = float(roc_auc_score(y, proba)) if 0 < y.mean() < 1 else float("nan")
    policy = gate_policy_value(delta, proba)
    features = pd.DataFrame(X, columns=names)
    rules = rule_sweep(features, delta)
    summary = {
        "run": name, "n": int(n), "n_features": int(X.shape[1]), "model": model,
        "base_rate_improved": float(y.mean()), "oof_auc": auc,
        "mean_delta_always_on": float(delta.mean()),
        "mean_delta_never_on": 0.0,
        "mean_delta_oracle": float(np.where(delta > 0, delta, 0).mean()),
        "ceiling_recovery": ceiling_recovery(policy),
        "best_rule": (rules.sort_values("mean_delta", ascending=False).iloc[0].to_dict() if len(rules) else None),
    }
    log.info("%-38s n=%d AUC=%.3f always=%+.4f oracle=%+.4f recovery=%.2f",
             name, n, auc, summary["mean_delta_always_on"], summary["mean_delta_oracle"], summary["ceiling_recovery"])
    return summary, policy.assign(run=name), rules.assign(run=name)


def main() -> None:
    parser = get_arg_parser("Learned-gate pilot on generated labels (dev -> eval)")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Result folders or *_details.json paths (feedback method runs)")
    parser.add_argument("--feedback-protocol", default="conditioned")
    parser.add_argument("--model", default="logistic", choices=["logistic", "xgboost"])
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("gate_pilot")

    paths = ProjectPaths()
    out_dir = paths.results / f"gate_pilot_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("gate_pilot")

    df = load_dataset()
    teams = sorted(df["Team->Name"].astype(str).unique())
    classes = sorted(df["intent_class"].astype(str).unique())
    fb_db = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    feedback_scores = load_feedback_as_scores(fb_db) if fb_db.exists() else {}

    summaries, policies, rules = [], [], []
    inputs = []
    for run in args.runs:
        results, used = load_run(run)
        inputs.append(used)
        # only records that actually have both retrieval sides and a delta
        results = [r for r in results if r.get("baseline", {}).get("retrieval") and r.get("deltas")]
        s, p, ru = evaluate(used.parent.name if used.is_file() else run, results,
                            feedback_scores, teams, classes, args.model, log)
        summaries.append(s)
        policies.append(p)
        rules.append(ru)

    summary_df = pd.DataFrame(summaries)
    policy_df = pd.concat(policies, ignore_index=True)
    rules_df = pd.concat(rules, ignore_index=True)
    summary_df.to_csv(out_dir / "gate_pilot.csv", index=False)
    policy_df.to_csv(out_dir / "policy.csv", index=False)
    rules_df.to_csv(out_dir / "rules.csv", index=False)
    (out_dir / "gate_pilot.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    manifest = write_run_manifest(out_dir, script="experiments/16_gate_pilot.py", args=args, inputs=inputs,
                                  extra={"n_runs": len(summaries), "model": args.model,
                                         "n_features": summaries[0]["n_features"] if summaries else 0}, run_id=run_id)
    append_registry(run_id=run_id, phase="P5-gate-pilot", script="16_gate_pilot.py", out_dir=out_dir,
                    manifest=manifest, headline_metric="max_oof_auc",
                    headline_value=float(summary_df["oof_auc"].max()) if len(summary_df) else "")

    print("\n=== GATE PILOT (generated labels, out-of-fold) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== POLICY (per run) ===")
    print(policy_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== BEST SIMPLE RULE (fallback) ===")
    best_rules = rules_df.sort_values("mean_delta", ascending=False).groupby("run").head(1)
    print(best_rules.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()