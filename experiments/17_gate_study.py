#!/usr/bin/env python3
"""
17_gate_study.py — P5 general gate study (dev -> eval, both protocols)
======================================================================
Builds ONE benefit gate that is general across routing signals: every
(ticket, configuration) pair is a sample, features describe the pool
distribution, cosine correlation and routing evidence for that configuration,
and the label is the generated-answer delta of the matching run. The gate is
trained on dev and applied unchanged to eval.

Methodological rules (A3):
  * out-of-fold probabilities use GroupKFold by ticket (a ticket contributes
    one row per configuration, so row-level folds would leak);
  * thresholds are selected on dev only and frozen before eval; the eval
    threshold sweep is written as a *diagnostic* file, never as the policy;
  * bootstrap intervals resample tickets (clusters), not rows;
  * every label source records its generation regime (model + system prompt +
    cache) so runs from different regimes cannot be mixed silently.

Outputs: results/gate_study_<tag>/{features_dev.parquet, features_eval.parquet,
         static_gate.csv, static_gate_eval.csv, learned_gate.csv,
         learned_gate_decomposition.csv, learned_gate_eval_sweep.csv,
         learned_gate_proxy.csv, multi_action.csv, summary.json, gate_model.joblib}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.utils import (setup_logging, load_dataset, load_split, split_path, get_arg_parser,
                               write_run_manifest, append_registry, make_run_id, set_seeds)
from src.config import ProjectPaths, LiftConfig, RoutingConfig
from src.evaluation.ladder import collect_pools, build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.feedback.loader import load_feedback_bundle
from src.gate.policy import (THRESHOLDS, best_dev_threshold, clustered_bootstrap_auc,
                             counterfactual_decomposition, eval_policy_stats, fit_final_gate,
                             grouped_oof_proba, policy_value)
from src.gate.pool_features import config_pool_features
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.retrieval.retrievers import build_retriever
from src.retrieval.semantic import TicketTextSimilarity

META = {"config_key", "protocol", "agg", "split", "ticket_id", "delta", "proxy_delta_top1"}

# (config_key, protocol, agg, routing, lift, run folder)
CONFIGS = [
    ("global_laplace", "conditioned", "continuous", RoutingConfig.global_(), LiftConfig.laplace(), "M1_global_dev_conditioned_continuous"),
    ("global_laplace", "blind", "continuous", RoutingConfig.global_(), LiftConfig.laplace(), "M1_global_dev_blind_continuous"),
    ("global_laplace_binary", "conditioned", "binary", RoutingConfig.global_(), LiftConfig.laplace(), "M1_global_dev_conditioned_binary"),
    ("team_laplace", "conditioned", "continuous", RoutingConfig.team_only(), LiftConfig.laplace(), "M2_team_dev_conditioned_continuous"),
    ("team_laplace", "blind", "continuous", RoutingConfig.team_only(), LiftConfig.laplace(), "M2_team_dev_blind_continuous"),
    ("class_laplace", "conditioned", "continuous", RoutingConfig.class_only(), LiftConfig.laplace(), "M3_class_dev_conditioned_continuous"),
    ("class_laplace", "blind", "continuous", RoutingConfig.class_only(), LiftConfig.laplace(), "M3_class_dev_blind_continuous"),
    ("intersection_laplace", "conditioned", "continuous", RoutingConfig.intersection(), LiftConfig.laplace(), "M4_intersection_dev_conditioned_continuous"),
    ("intersection_laplace", "blind", "continuous", RoutingConfig.intersection(), LiftConfig.laplace(), "M4_intersection_dev_blind_continuous"),
    ("intersection_eb", "conditioned", "continuous", RoutingConfig.intersection(), LiftConfig.laplace_eb(10.0, "pool_std", 1.0), "M4_intersection_dev_conditioned_continuous_liftlaplace_eb_k10_pool_std1"),
    ("intersection_eb", "blind", "continuous", RoutingConfig.intersection(), LiftConfig.laplace_eb(10.0, "pool_std", 1.0), "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std1"),
    ("backoff_eb", "conditioned", "continuous", RoutingConfig.backoff(min_evidence=2.0), LiftConfig.laplace_eb(2.0, "pool_std", 0.5), "M5_backoff_dev_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    ("backoff_eb", "blind", "continuous", RoutingConfig.backoff(min_evidence=2.0), LiftConfig.laplace_eb(2.0, "pool_std", 0.5), "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
]

EVAL_CONFIGS = [
    ("team_laplace", "conditioned", "continuous", RoutingConfig.team_only(), LiftConfig.laplace(), "M2_team_eval_conditioned_continuous"),
    ("team_laplace", "blind", "continuous", RoutingConfig.team_only(), LiftConfig.laplace(), "M2_team_eval_blind_continuous"),
    ("intersection_laplace", "conditioned", "continuous", RoutingConfig.intersection(), LiftConfig.laplace(), "M4_intersection_eval_conditioned_continuous"),
    ("intersection_laplace", "blind", "continuous", RoutingConfig.intersection(), LiftConfig.laplace(), "M4_intersection_eval_blind_continuous"),
    ("backoff_eb", "conditioned", "continuous", RoutingConfig.backoff(min_evidence=2.0), LiftConfig.laplace_eb(2.0, "pool_std", 0.5), "M5_backoff_eval_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    ("backoff_eb", "blind", "continuous", RoutingConfig.backoff(min_evidence=2.0), LiftConfig.laplace_eb(2.0, "pool_std", 0.5), "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
]


def newest_anchored_details(folder_name: str) -> tuple[Path, dict]:
    """Newest details file, preferring a summary whose generation regime is warm.

    Returns (details_path, provenance). Runs recorded before the regime block
    existed (eval runs of Sep 20) return a provenance dict with ``anchored=None``.
    """
    folder = ProjectPaths().results / folder_name
    summaries = sorted(folder.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"no summary in {folder}")
    warm = []
    for path in summaries:
        try:
            block = json.loads(path.read_text(encoding="utf-8")).get("generation_regime") or {}
        except json.JSONDecodeError:
            continue
        if block.get("warm") is True:
            warm.append(path)
    chosen = warm[-1] if warm else summaries[-1]
    summary = json.loads(chosen.read_text(encoding="utf-8"))
    regime = summary.get("generation_regime") or {}
    details = chosen.with_name(chosen.name.replace("_summary.json", "_details.json"))
    if not details.exists():
        raise FileNotFoundError(details)
    return details, {
        "summary": chosen.name,
        "regime_id": regime.get("regime_id"),
        "warm": regime.get("warm"),
        "cache_misses": regime.get("cache_misses"),
        "anchored": bool(warm) if regime else None,
    }


def load_labels(folder_name: str) -> tuple[dict[str, float], dict]:
    details, provenance = newest_anchored_details(folder_name)
    records = json.loads(details.read_text(encoding="utf-8"))
    labels = {r["ticket_id"]: float(r["deltas"]["delta_cosine"]) for r in records if r.get("deltas")}
    provenance["n_labeled"] = len(labels)
    return labels, provenance


def build_tensors(paths, df, split, retriever_name, encoder, sim, text_sim, log):
    """One PoolTensor per (split, protocol, agg-mode); pools are shared."""
    idx = FAISSIndex(dim=384)
    idx.load(paths.data_processed / "faiss_index" / "faiss.index",
             paths.data_processed / "faiss_index" / "faiss_metadata.parquet")
    retr = build_retriever(retriever_name, paths, idx, encoder, ce_pool=100)
    pools_dir = paths.results / "retriever_ladder" / "pools"
    tensors, base_tops = {}, {}
    for sp in ("dev", "eval"):
        queries = df.set_index("seq_id").loc[split[sp]].reset_index()
        pool_path = pools_dir / f"{retriever_name}_{sp}_seed42.parquet"
        pools = collect_pools(retr, queries, encoder, 100, cache_path=pool_path, id_to_index=idx.id_to_index)
        for protocol, mode in (("conditioned", "continuous"), ("blind", "continuous"), ("conditioned", "binary")):
            fb_db = paths.data_processed / f"feedback_{protocol}.db"
            bundle = load_feedback_bundle(fb_db, exclude_query_ids=set(split[sp]), mode=mode)
            key = (sp, protocol, mode)
            tensors[key] = build_pool_tensor(pools, queries, bundle, {"minilm": sim}, text_sim=text_sim)
            base_tops[key] = tensors[key].baseline_top(5)
            log.info("tensor %s: %d x %d", key, *tensors[key].score.shape)
    return tensors, base_tops


def main() -> None:
    parser = get_arg_parser("General gate study (dev -> eval, both protocols)")
    parser.add_argument("--retriever", default="dense_minilm")
    parser.add_argument("--tag", default="general")
    parser.add_argument("--kappas", nargs="*", type=float, default=[2.0, 10.0])
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("gate_study")
    set_seeds(42)

    paths = ProjectPaths()
    out_dir = paths.results / f"gate_study_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("gate_study")

    df = load_dataset()
    split = load_split(42)
    encoder = TicketEncoder("all-MiniLM-L6-v2")
    sim = ReplySimilarity(df, "minilm")
    text_sim = TicketTextSimilarity(df, encoder=encoder)
    run_ids = []
    label_provenance: dict[str, dict] = {}

    for split_name, configs in (("dev", CONFIGS), ("eval", EVAL_CONFIGS)):
        tensors, base_tops = build_tensors(paths, df, split, args.retriever, encoder, sim, text_sim, log)
        frames = []
        for key, protocol, agg, routing, lift, run_folder in configs:
            T = tensors[(split_name, protocol, agg if agg == "binary" else "continuous")]
            base = base_tops[(split_name, protocol, agg if agg == "binary" else "continuous")]
            feat = config_pool_features(T, lift, routing, base, k=5)
            rows = np.arange(len(T.query_ids))[:, None]
            top = T.rerank(T.routed_lift(lift, routing), T.scale_factor(lift), 5)
            qidx = np.array([sim.pos[str(t)] for t in T.query_ids])
            cand_idx = np.vectorize(lambda cid: sim.pos.get(str(cid), 0))(T.cand_ids)
            fb_sim = sim.sim[qidx[:, None], cand_idx]
            feat["proxy_delta_top1"] = fb_sim[rows, top][:, 0] - fb_sim[rows, base][:, 0]
            feat.insert(0, "ticket_id", T.query_ids)
            feat.insert(0, "split", split_name)
            feat.insert(0, "agg", agg)
            feat.insert(0, "protocol", protocol)
            feat.insert(0, "config_key", key)
            labels, provenance = load_labels(run_folder)
            label_provenance[f"{split_name}/{key}/{protocol}"] = provenance
            feat["delta"] = [labels.get(t, np.nan) for t in feat["ticket_id"]]
            frames.append(feat)
            log.info("%s %-22s %-11s %-9s n=%d labeled=%d mean=%.4f anchored=%s",
                     split_name, key, protocol, agg, len(feat), int(feat["delta"].notna().sum()),
                     float(feat["delta"].mean()), provenance.get("anchored"))
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(out_dir / f"features_{split_name}.parquet", index=False)
        run_ids.append(run_folder)
    log.info("saved feature tables")

    regimes = {k: v.get("regime_id") for k, v in label_provenance.items() if v.get("regime_id")}
    if len(set(regimes.values())) > 1:
        log.warning("LABEL PROVENANCE: mixed generation regimes across configs: %s", regimes)
    unanchored = [k for k, v in label_provenance.items() if v.get("anchored") is not True]
    if unanchored:
        log.warning("LABEL PROVENANCE: configs without a warm anchored run: %s", unanchored)

    dev = pd.read_parquet(out_dir / "features_dev.parquet")
    ev = pd.read_parquet(out_dir / "features_eval.parquet")
    dev_l = dev.dropna(subset=["delta"]).copy()
    ev_l = ev.dropna(subset=["delta"]).copy()
    feature_cols = [c for c in dev_l.columns if c not in META]
    deployable = [c for c in feature_cols if not c.startswith("proxy_")]
    oracle_cols = deployable + ["proxy_delta_top1"]

    # --- static threshold gate: best single-feature rule on dev ---
    static_rows = []
    for col in deployable:
        vals = dev_l[col].to_numpy(float)
        for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            t = float(np.quantile(vals, q))
            open_ = vals <= t
            val = float(np.mean(np.where(open_, dev_l["delta"].to_numpy(), 0.0)))
            static_rows.append({"feature": col, "direction": "below", "threshold": t, "dev_policy": val, "pct_open": float(open_.mean())})
    static = pd.DataFrame(static_rows).sort_values("dev_policy", ascending=False)
    static.to_csv(out_dir / "static_gate.csv", index=False)
    best_rule = static.iloc[0]

    static_eval = []
    for key, part in ev_l.groupby("config_key"):
        vals = part[best_rule["feature"]].to_numpy(float)
        open_ = vals <= best_rule["threshold"]
        d = part["delta"].to_numpy()
        static_eval.append({"config_key": key, "policy": "static_rule", "mean_delta": float(np.mean(np.where(open_, d, 0.0))),
                            "pct_open": float(open_.mean()), "always_on": float(d.mean())})
    static_eval = pd.DataFrame(static_eval)
    static_eval.to_csv(out_dir / "static_gate_eval.csv", index=False)

    # --- learned gate (deployable features), trained on dev with grouped CV ---
    Xd = np.nan_to_num(dev_l[deployable].to_numpy(float))
    yd = (dev_l["delta"].to_numpy() > 0).astype(int)
    Xe = np.nan_to_num(ev_l[deployable].to_numpy(float))
    groups_dev = dev_l["ticket_id"].to_numpy()
    oof = grouped_oof_proba(Xd, yd, groups_dev)
    dev_auc = clustered_bootstrap_auc(yd, oof, groups_dev)
    clf = fit_final_gate(Xd, yd)
    dev_l["gate_proba"] = oof
    ev_l["gate_proba"] = clf.predict_proba(Xe)[:, 1]

    # persist the fitted gate so the evaluation pipeline can turn it on (04_evaluate.py --gating-model)
    import joblib
    joblib.dump({"model": clf, "features": deployable}, out_dir / "gate_model.joblib")

    # dev-frozen thresholds, one per (config, protocol)
    dev_thresholds = {}
    for (key, protocol), part in dev_l.groupby(["config_key", "protocol"]):
        t, v = best_dev_threshold(part["delta"].to_numpy(), part["gate_proba"].to_numpy())
        dev_thresholds[(key, protocol)] = {
            "threshold": t, "dev_policy": v, "always_on": float(part["delta"].mean()),
            "n_rows": int(len(part)), "n_tickets": int(part["ticket_id"].nunique()),
        }

    learned_rows, decomp_rows = [], []
    for (key, protocol), part in ev_l.groupby(["config_key", "protocol"]):
        frozen = dev_thresholds.get((key, protocol))
        if frozen is None or not np.isfinite(frozen["threshold"]):
            continue
        delta = part["delta"].to_numpy(float)
        proba = part["gate_proba"].to_numpy(float)
        groups = part["ticket_id"].to_numpy()
        stats = eval_policy_stats(delta, proba, groups, frozen["threshold"])
        auc = clustered_bootstrap_auc((delta > 0).astype(int), proba, groups)
        learned_rows.append({
            "config_key": key, "protocol": protocol,
            "n_dev_tickets": frozen["n_tickets"], "dev_threshold": frozen["threshold"],
            "dev_policy": frozen["dev_policy"], "n_eval": len(part),
            "eval_auc": auc["auc"], "eval_auc_ci_lower": auc["auc_ci_lower"],
            "eval_auc_ci_upper": auc["auc_ci_upper"], **stats,
        })
        decomp_rows.append({"config_key": key, "protocol": protocol,
                            "dev_threshold": frozen["threshold"],
                            **counterfactual_decomposition(delta, proba, frozen["threshold"])})
    learned = pd.DataFrame(learned_rows)
    learned.to_csv(out_dir / "learned_gate.csv", index=False)
    pd.DataFrame(decomp_rows).to_csv(out_dir / "learned_gate_decomposition.csv", index=False)

    # diagnostic only: eval-threshold sweep (never used to pick the policy)
    sweep_rows = []
    for (key, protocol), part in ev_l.groupby(["config_key", "protocol"]):
        delta = part["delta"].to_numpy(float)
        proba = part["gate_proba"].to_numpy(float)
        for t in THRESHOLDS:
            sweep_rows.append({"config_key": key, "protocol": protocol, "threshold": t,
                               "eval_policy": policy_value(delta, proba, t),
                               "pct_open": float(np.mean(proba >= t))})
    pd.DataFrame(sweep_rows).to_csv(out_dir / "learned_gate_eval_sweep.csv", index=False)

    # --- oracle-proxy diagnostic (grouped CV + its own dev-frozen thresholds) ---
    Xp = np.nan_to_num(dev_l[oracle_cols].to_numpy(float))
    Xpe = np.nan_to_num(ev_l[oracle_cols].to_numpy(float))
    oof_p = grouped_oof_proba(Xp, yd, groups_dev)
    proxy_auc = clustered_bootstrap_auc(yd, oof_p, groups_dev)
    clf_p = fit_final_gate(Xp, yd)
    dev_l["gate_proba_proxy"] = oof_p
    ev_l["gate_proba_proxy"] = clf_p.predict_proba(Xpe)[:, 1]
    dev_thresholds_proxy = {}
    for (key, protocol), part in dev_l.groupby(["config_key", "protocol"]):
        t, v = best_dev_threshold(part["delta"].to_numpy(), part["gate_proba_proxy"].to_numpy())
        dev_thresholds_proxy[(key, protocol)] = {"threshold": t, "dev_policy": v}
    proxy_rows = []
    for (key, protocol), part in ev_l.groupby(["config_key", "protocol"]):
        frozen = dev_thresholds_proxy.get((key, protocol))
        if frozen is None or not np.isfinite(frozen["threshold"]):
            continue
        stats = eval_policy_stats(part["delta"].to_numpy(float), part["gate_proba_proxy"].to_numpy(float),
                                  part["ticket_id"].to_numpy(), frozen["threshold"])
        proxy_rows.append({"config_key": key, "protocol": protocol,
                           "dev_threshold": frozen["threshold"], **stats})
    pd.DataFrame(proxy_rows).to_csv(out_dir / "learned_gate_proxy.csv", index=False)

    # --- multi-action selector: exploratory, sign-only (see A5 for the magnitude-aware design) ---
    act = ev_l.pivot_table(index=["protocol", "ticket_id"], columns="config_key", values="gate_proba")
    delta_pivot = ev_l.pivot_table(index=["protocol", "ticket_id"], columns="config_key", values="delta")
    chosen = act.idxmax(axis=1)
    achieved = np.array([delta_pivot.loc[i, c] for i, c in chosen.items()])
    best_fixed = delta_pivot.mean(axis=0)
    oracle = delta_pivot.max(axis=1).mean()
    multi = pd.DataFrame([
        {"policy": "always_on_best_fixed", "mean_delta": float(best_fixed.max())},
        {"policy": "learned_multi_action", "mean_delta": float(achieved.mean())},
        {"policy": "never_on", "mean_delta": 0.0},
        {"policy": "oracle_action", "mean_delta": float(oracle)},
    ])
    multi.to_csv(out_dir / "multi_action.csv", index=False)

    summary = {
        "run_id": run_id, "retriever": args.retriever,
        "n_dev_samples": int(len(dev_l)), "n_eval_samples": int(len(ev_l)),
        "cv": "GroupKFold(n_splits=5) by ticket",
        "dev_deployable_oof_auc": dev_auc["auc"],
        "dev_deployable_oof_auc_ci_lower": dev_auc["auc_ci_lower"],
        "dev_deployable_oof_auc_ci_upper": dev_auc["auc_ci_upper"],
        "dev_proxy_oof_auc": proxy_auc["auc"],
        "dev_thresholds": {f"{k}|{p}": v for (k, p), v in dev_thresholds.items()},
        "label_provenance": label_provenance,
        "best_static_rule": best_rule.to_dict(),
        "learned_gate": learned.to_dict("records"),
        "decomposition": decomp_rows,
        "multi_action": multi.to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = write_run_manifest(out_dir, script="experiments/17_gate_study.py", args=args,
                                  inputs=[paths.dataset_parquet, split_path(42),
                                          paths.data_processed / "feedback_conditioned.db",
                                          paths.data_processed / "feedback_blind.db"],
                                  extra={"n_dev": len(dev_l), "n_eval": len(ev_l), "n_features": len(deployable),
                                         "cv": summary["cv"], "regime_ids": sorted(set(regimes.values()))},
                                  run_id=run_id)
    append_registry(run_id=run_id, phase="P5-gate-study", script="17_gate_study.py", out_dir=out_dir,
                    manifest=manifest, headline_metric="dev_deployable_oof_auc",
                    headline_value=round(float(dev_auc["auc"]), 4))

    pd.set_option("display.width", 240)
    print("\n=== STATIC GATE: best dev rule ===")
    print(best_rule.to_string())
    print("\n=== LEARNED GATE (grouped CV by ticket, dev-frozen thresholds) ===")
    print("dev OOF AUC = %.3f [%.3f, %.3f]" % (dev_auc["auc"], dev_auc["auc_ci_lower"], dev_auc["auc_ci_upper"]))
    print(learned.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== COUNTERFACTUAL DECOMPOSITION (eval, frozen threshold) ===")
    print(pd.DataFrame(decomp_rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== STATIC RULE on eval ===")
    print(static_eval.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n=== MULTI-ACTION (exploratory) ===")
    print(multi.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
