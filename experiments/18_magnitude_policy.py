#!/usr/bin/env python3
"""
18_magnitude_policy.py — P6 magnitude-aware control policy (A5)
===============================================================
The sign gate (17_gate_study.py) predicts P(delta > 0). The decision that
maximises the mean generated-answer delta is to apply feedback when the
*expected delta* (sign x magnitude) exceeds a cutoff. This script:

  1. trains an expected-delta regressor (HistGradientBoosting, plus a Ridge
     comparison) on the same (ticket, configuration) dev features with
     GroupKFold by ticket,
  2. freezes an expected-value cutoff tau on dev and applies it to eval,
  3. compares the expected-value policy against the sign gate, always-on,
     never-on and the oracle, with ticket-clustered bootstrap CIs and the
     open/closed counterfactual decomposition,
  4. evaluates magnitude-aware action selection (choose the configuration with
     the highest predicted delta, or abstain) against best-fixed and oracle.

No API calls. Inputs: results/gate_study_general/features_{dev,eval}.parquet.
Outputs: results/magnitude_policy/{dev_predictions.parquet, eval_predictions.parquet,
         policy_table.csv, decomposition.csv, tau_curves.csv, action_selection.csv,
         summary.json, manifest_*.json}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.utils import (setup_logging, get_arg_parser, write_run_manifest,
                               append_registry, make_run_id, set_seeds)
from src.config import ProjectPaths
from src.gate.policy import (best_dev_threshold, best_dev_tau, clustered_bootstrap_auc,
                             clustered_bootstrap_mean, counterfactual_decomposition,
                             eval_policy_stats, fit_final_gate, fit_final_regressor,
                             grouped_oof_proba, grouped_oof_regression, policy_value,
                             regression_quality)

META = {"config_key", "protocol", "agg", "split", "ticket_id", "delta", "proxy_delta_top1"}


def tau_grid(pred: np.ndarray, n: int = 19) -> list[float]:
    qs = np.quantile(np.asarray(pred, dtype=float), np.linspace(0.05, 0.95, n))
    return sorted({0.0, *[float(q) for q in qs]})


def policy_row(key: str, protocol: str, part: pd.DataFrame, score_col: str,
               threshold: float, n_resamples: int, seed: int) -> dict:
    delta = part["delta"].to_numpy(float)
    score = part[score_col].to_numpy(float)
    groups = part["ticket_id"].to_numpy()
    stats = eval_policy_stats(delta, score, groups, threshold, n_resamples=n_resamples, seed=seed)
    return {"config_key": key, "protocol": protocol, "score": score_col,
            "threshold": float(threshold), "n": len(part), **stats}


def action_selection(df: pd.DataFrame, score_col: str, label: str, n_resamples: int, seed: int) -> dict:
    """Per (protocol, ticket): pick the configuration with the highest score, or abstain."""
    score = df.pivot_table(index=["protocol", "ticket_id"], columns="config_key", values=score_col)
    delta = df.pivot_table(index=["protocol", "ticket_id"], columns="config_key", values="delta")
    valid = delta.notna().all(axis=1)
    score, delta = score[valid], delta[valid]
    chosen = score.idxmax(axis=1)
    best = score.max(axis=1)
    achieved = np.where(best.to_numpy(float) > 0.0,
                        np.array([delta.loc[i, c] for i, c in chosen.items()], dtype=float),
                        0.0)
    fixed = delta.mean(axis=0)
    oracle = delta.max(axis=1).to_numpy(float)
    groups = np.array([i[1] for i in delta.index], dtype=object)
    lo, hi = clustered_bootstrap_mean(achieved, groups, n_resamples=n_resamples, seed=seed)
    return {
        "policy": label, "n_tickets": int(len(delta)),
        "mean_delta": float(np.mean(achieved)), "ci_lower": lo, "ci_upper": hi,
        "pct_abstain": float(np.mean(best.to_numpy(float) <= 0.0)),
        "best_fixed_mean": float(fixed.max()), "best_fixed_config": str(fixed.idxmax()),
        "oracle_mean": float(np.mean(oracle)),
    }


def main() -> None:
    parser = get_arg_parser("Magnitude-aware control policy (expected-delta regression)")
    parser.add_argument("--features-dir", default="results/gate_study_general")
    parser.add_argument("--tag", default="")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("magnitude_policy")
    set_seeds(args.seed)

    paths = ProjectPaths()
    src_dir = paths.root / args.features_dir
    out_dir = paths.results / ("magnitude_policy" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("magnitude_policy")

    dev = pd.read_parquet(src_dir / "features_dev.parquet")
    ev = pd.read_parquet(src_dir / "features_eval.parquet")
    dev_l = dev.dropna(subset=["delta"]).copy()
    ev_l = ev.dropna(subset=["delta"]).copy()
    feature_cols = [c for c in dev_l.columns if c not in META]
    deployable = [c for c in feature_cols if not c.startswith("proxy_")]

    Xd = np.nan_to_num(dev_l[deployable].to_numpy(float))
    yd = dev_l["delta"].to_numpy(float)
    Xe = np.nan_to_num(ev_l[deployable].to_numpy(float))
    groups_dev = dev_l["ticket_id"].to_numpy()

    # --- sign gate (reference policy) ---
    y_sign = (yd > 0).astype(int)
    sign_oof = grouped_oof_proba(Xd, y_sign, groups_dev)
    sign_auc = clustered_bootstrap_auc(y_sign, sign_oof, groups_dev)
    sign_clf = fit_final_gate(Xd, y_sign)
    dev_l["sign_proba"] = sign_oof
    ev_l["sign_proba"] = sign_clf.predict_proba(Xe)[:, 1]

    # --- expected-delta regressors ---
    mag_oof = grouped_oof_regression(Xd, yd, groups_dev)
    ridge_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))  # noqa: E731
    ridge_oof = grouped_oof_regression(Xd, yd, groups_dev, model_factory=ridge_factory)
    mag_reg = fit_final_regressor(Xd, yd)
    dev_l["mag_pred"] = mag_oof
    dev_l["ridge_pred"] = ridge_oof
    ev_l["mag_pred"] = mag_reg.predict(Xe)
    ev_l["ridge_pred"] = fit_final_regressor(Xd, yd, model_factory=ridge_factory).predict(Xe)

    quality = {
        "gbr": regression_quality(yd, mag_oof),
        "ridge": regression_quality(yd, ridge_oof),
        "sign_auc": sign_auc,
    }
    log.info("dev OOF quality: GBR R2=%.3f rho=%.3f | Ridge R2=%.3f rho=%.3f | sign AUC=%.3f",
             quality["gbr"]["r2"], quality["gbr"]["spearman"], quality["ridge"]["r2"],
             quality["ridge"]["spearman"], sign_auc["auc"])

    # --- dev-frozen thresholds per (config, protocol) ---
    thresholds = {}
    tau_rows = []
    for (key, protocol), part in dev_l.groupby(["config_key", "protocol"]):
        sign_tau, sign_dev = best_dev_threshold(part["delta"].to_numpy(float), part["sign_proba"].to_numpy(float))
        taus = tau_grid(part["mag_pred"].to_numpy(float))
        mag_tau, mag_dev = best_dev_tau(part["delta"].to_numpy(float), part["mag_pred"].to_numpy(float), taus)
        ridge_tau, ridge_dev = best_dev_tau(part["delta"].to_numpy(float), part["ridge_pred"].to_numpy(float),
                                            tau_grid(part["ridge_pred"].to_numpy(float)))
        thresholds[(key, protocol)] = {"sign_tau": sign_tau, "mag_tau": mag_tau, "ridge_tau": ridge_tau}
        for tau in taus:
            tau_rows.append({"split": "dev", "config_key": key, "protocol": protocol, "tau": tau,
                             "policy_value": policy_value(part["delta"].to_numpy(float), part["mag_pred"].to_numpy(float), tau)})
        log.info("dev %-22s %-11s sign_tau=%.2f (%.4f)  mag_tau=%+.4f (%.4f)  ridge_tau=%+.4f (%.4f)",
                 key, protocol, sign_tau, sign_dev, mag_tau, mag_dev, ridge_tau, ridge_dev)

    # --- eval policies ---
    rows, decomp = [], []
    for (key, protocol), part in ev_l.groupby(["config_key", "protocol"]):
        frozen = thresholds.get((key, protocol))
        if frozen is None:
            continue
        for score_col, tau_key, label in (("sign_proba", "sign_tau", "sign_gate"),
                                          ("mag_pred", "mag_tau", "magnitude_gbr"),
                                          ("ridge_pred", "ridge_tau", "magnitude_ridge")):
            row = policy_row(key, protocol, part, score_col, frozen[tau_key], args.n_resamples, args.seed)
            row["policy"] = label
            rows.append(row)
        delta = part["delta"].to_numpy(float)
        score = part["mag_pred"].to_numpy(float)
        decomp.append({"config_key": key, "protocol": protocol, "tau": frozen["mag_tau"],
                       **counterfactual_decomposition(delta, score, frozen["mag_tau"])})
        for tau in tau_grid(score):
            tau_rows.append({"split": "eval", "config_key": key, "protocol": protocol, "tau": tau,
                             "policy_value": policy_value(delta, score, tau)})

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "policy_table.csv", index=False)
    pd.DataFrame(decomp).to_csv(out_dir / "decomposition.csv", index=False)
    pd.DataFrame(tau_rows).to_csv(out_dir / "tau_curves.csv", index=False)

    # --- magnitude-aware action selection (eval; all six configurations complete) ---
    actions = [
        action_selection(ev_l, "sign_proba", "sign_action", args.n_resamples, args.seed),
        action_selection(ev_l, "mag_pred", "magnitude_action", args.n_resamples, args.seed),
    ]
    actions_df = pd.DataFrame(actions)
    actions_df.to_csv(out_dir / "action_selection.csv", index=False)

    dev_l.to_parquet(out_dir / "dev_predictions.parquet", index=False)
    ev_l.to_parquet(out_dir / "eval_predictions.parquet", index=False)

    summary = {
        "run_id": run_id,
        "source_features": str(src_dir),
        "cv": "GroupKFold(n_splits=5) by ticket",
        "n_dev_samples": int(len(dev_l)), "n_eval_samples": int(len(ev_l)),
        "quality": quality,
        "dev_thresholds": {f"{k}|{p}": v for (k, p), v in thresholds.items()},
        "policy_table": table.to_dict("records"),
        "decomposition": decomp,
        "action_selection": actions,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = write_run_manifest(out_dir, script="experiments/18_magnitude_policy.py", args=args,
                                  inputs=[src_dir / "features_dev.parquet", src_dir / "features_eval.parquet"],
                                  extra={"n_features": len(deployable), "quality": quality},
                                  run_id=run_id)
    append_registry(run_id=run_id, phase="P6-magnitude-policy", script="18_magnitude_policy.py",
                    out_dir=out_dir, manifest=manifest,
                    headline_metric="gbr_oof_spearman", headline_value=round(quality["gbr"]["spearman"], 4))

    pd.set_option("display.width", 260)
    print("\n=== DEV OOF QUALITY ===")
    print(json.dumps(quality, indent=2))
    print("\n=== POLICY TABLE (eval, dev-frozen thresholds) ===")
    cols = ["config_key", "protocol", "policy", "threshold", "always_on", "oracle", "eval_policy",
            "eval_policy_ci_lower", "eval_policy_ci_upper", "pct_open", "gain_vs_always_on", "ceiling_recovery"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== MAGNITUDE DECOMPOSITION (eval) ===")
    print(pd.DataFrame(decomp).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== ACTION SELECTION (eval) ===")
    print(actions_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
