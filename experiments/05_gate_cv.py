#!/usr/bin/env python3
"""
05_gate_cv.py — Phase 4 / P3
=============================
Learned gate: predict, from pre-generation retrieval features only, whether
applying feedback will improve the answer.

Two modes
---------
A) Legacy (SIKDD): features + labels (delta_cosine > 0) from a *_details.json
       python experiments/05_gate_cv.py --details-json results/<run>/<...>_details.json

B) Proxy-labelled (P3, no API cost): feature tables written by 12_learn_blend.py
       python experiments/05_gate_cv.py --features-parquet results/blend/gate_features_train.parquet \
              --eval-features-parquet results/blend/gate_features_dev.parquet \
              --eval-details results/M2_team_dev_conditioned_continuous/<...>_details.json
   Trains with nested CV on TRAIN (label = proxy improvement), then fits on all of
   train and evaluates on DEV against (i) the proxy label and (ii) the GENERATED
   delta_cosine label from --eval-details (joined on ticket_id), including the
   value of the gate policy (mean generated delta when opening only on P >= t).

Outputs: results/gate[_<tag>]/{gate_cv_results.json, feature_importance.json,
         dev_eval.json, dev_policy_value.csv, manifest.json}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from experiments.utils import (setup_logging, load_dataset, get_arg_parser, write_run_manifest, append_registry, make_run_id)
from src.config import ProjectPaths
from src.gate.features import extract_feature_matrix
from src.gate.model import train_evaluate_gate, fit_gate_model, gate_policy_value
from src.feedback.loader import load_feedback_as_scores

NON_FEATURES = {"ticket_id", "expected_class", "expected_team", "routing", "proxy_delta_top1", "proxy_label_improved",
                "proxy_label_harmed", "base_proxy_top1", "top1_changed"}


def frame_to_xy(df: pd.DataFrame, label: str, one_hot: bool, teams: list[str], classes: list[str]):
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feats].to_numpy(float)
    names = list(feats)
    if one_hot:
        X = np.hstack([X, np.stack([(df["expected_team"] == t).astype(float).to_numpy() for t in teams], 1),
                       np.stack([(df["expected_class"] == c).astype(float).to_numpy() for c in classes], 1)])
        names += [f"team_{t}" for t in teams] + [f"class_{c}" for c in classes]
    y = df[label].to_numpy(float)
    return np.nan_to_num(X), y, names


def main() -> None:
    parser = get_arg_parser("Gate CV training")
    parser.add_argument("--details-json", type=str, default=None, help="legacy mode: *_details.json (train split)")
    parser.add_argument("--features-parquet", type=str, default=None, help="proxy mode: train feature table from 12_learn_blend.py")
    parser.add_argument("--eval-features-parquet", type=str, default=None, help="proxy mode: dev feature table")
    parser.add_argument("--eval-details", type=str, default=None, help="dev *_details.json with generated deltas (optional)")
    parser.add_argument("--label", default="proxy_label_improved", choices=["proxy_label_improved", "top1_changed"])
    parser.add_argument("--one-hot", action="store_true", help="add team/class one-hot features (dataset-specific!)")
    parser.add_argument("--model", default="logistic", choices=["logistic", "xgboost"])
    parser.add_argument("--feedback-protocol", type=str, default="conditioned")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("gate_cv")
    paths = ProjectPaths()
    output_dir = paths.results / ("gate" + (f"_{args.tag}" if args.tag else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("gate_cv")
    df = load_dataset()
    teams = sorted(df["Team->Name"].astype(str).unique())
    classes = sorted(df["intent_class"].astype(str).unique())
    inputs = []

    if args.details_json:
        details_path = Path(args.details_json)
        inputs.append(details_path)
        results = json.loads(details_path.read_text("utf-8"))
        fb_db_path = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
        feedback_scores = load_feedback_as_scores(fb_db_path) if fb_db_path.exists() else {}
        X, y, feature_names = extract_feature_matrix(results, feedback_scores, teams, classes)
        eval_df = None
    elif args.features_parquet:
        tr = pd.read_parquet(args.features_parquet)
        inputs.append(Path(args.features_parquet))
        X, y, feature_names = frame_to_xy(tr, args.label, args.one_hot, teams, classes)
        eval_df = pd.read_parquet(args.eval_features_parquet) if args.eval_features_parquet else None
        if eval_df is not None:
            inputs.append(Path(args.eval_features_parquet))
    else:
        raise SystemExit("give --details-json or --features-parquet")

    log.info("Feature matrix: %s, class balance=%.2f", X.shape, float(np.mean(y)))
    if len(X) < 50:
        log.error("Too few samples (%d) for CV.", len(X))
        return
    results_cv = train_evaluate_gate(X, y, feature_names, output_dir)
    summary_val = results_cv["aggregate"]["logistic"]["auc_mean"]

    dev_eval = {}
    if eval_df is not None:
        Xd, yd, _ = frame_to_xy(eval_df, args.label, args.one_hot, teams, classes)
        model, scaler = fit_gate_model(X, y, feature_names, model_type=args.model)
        proba = model.predict_proba(scaler.transform(Xd))[:, 1]
        dev_eval["proxy_label"] = {"n": int(len(yd)), "auc": float(roc_auc_score(yd, proba)) if 0 < yd.mean() < 1 else None,
                                   "accuracy@0.5": float(accuracy_score(yd, proba >= 0.5)), "base_rate": float(yd.mean())}
        pol = gate_policy_value(eval_df["proxy_delta_top1"].to_numpy(float), proba)
        pol["label_source"] = "proxy_delta_top1"
        pols = [pol]
        if args.eval_details:
            recs = json.loads(Path(args.eval_details).read_text("utf-8"))
            inputs.append(Path(args.eval_details))
            gen = pd.DataFrame({"ticket_id": [r["ticket_id"] for r in recs], "delta_cosine": [r["deltas"]["delta_cosine"] for r in recs]})
            j = eval_df[["ticket_id"]].copy()
            j["proba"] = proba
            j = j.merge(gen, on="ticket_id", how="inner")
            yg = (j["delta_cosine"] > 0).astype(float).to_numpy()
            dev_eval["generated_label"] = {"n": int(len(j)), "auc": float(roc_auc_score(yg, j["proba"])) if 0 < yg.mean() < 1 else None,
                                           "base_rate": float(yg.mean()), "details": args.eval_details,
                                           "note": "generated deltas come from the routing of --eval-details; features/proba from the routing in the feature table"}
            pg = gate_policy_value(j["delta_cosine"].to_numpy(float), j["proba"].to_numpy())
            pg["label_source"] = "generated_delta_cosine"
            pols.append(pg)
        pd.concat(pols, ignore_index=True).to_csv(output_dir / "dev_policy_value.csv", index=False)
        (output_dir / "dev_eval.json").write_text(json.dumps(dev_eval, indent=2), encoding="utf-8")
        summary_val = dev_eval.get("generated_label", dev_eval["proxy_label"]).get("auc")

    manifest = write_run_manifest(output_dir, script="experiments/05_gate_cv.py", args=args, inputs=inputs,
                                  extra={"n_samples": int(len(y)), "n_features": int(X.shape[1])}, run_id=run_id)
    append_registry(run_id=run_id, phase="P3-gate", script="05_gate_cv.py", out_dir=output_dir, manifest=manifest,
                    headline_metric="dev AUC (generated label if available)", headline_value=summary_val)

    print("\n=== GATE CV (train, nested) ===")
    for k, v in results_cv.get("aggregate", {}).items():
        print(f"  {k}: AUC={v['auc_mean']:.3f}±{v['auc_std']:.3f}  acc={v['accuracy_mean']:.3f}")
    top_coef = sorted(results_cv.get("lr_coefficients", {}).items(), key=lambda kv: -abs(kv[1]))[:10]
    print("  top |LR coef|:", [(k, round(v, 3)) for k, v in top_coef])
    if dev_eval:
        print("=== DEV ===", json.dumps(dev_eval, indent=1))
        print(pd.read_csv(output_dir / "dev_policy_value.csv").to_string(index=False))
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
