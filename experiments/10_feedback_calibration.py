#!/usr/bin/env python3
"""
10_feedback_calibration.py — P1.4
==================================
Judge-score calibration and lift-saturation diagnostics (no API calls).

Part A — is the LLM judge a valid feedback surrogate?
  For every (query, candidate) row of the feedback DB the "usefulness" of the
  candidate is known offline: sim(reply[candidate], reply[query]) and the
  embedding-free same-reply / same-group hits. We report
    * score distribution per protocol (mean, quantiles, % <= 0.40, % >= 0.80),
    * reliability table: mean usefulness per judge-score bin (is it monotone?),
    * AUC of the judge score for predicting same_reply / group_hit / sim >= 0.8,
    * Spearman(score, usefulness),
    * per-scope prior means p_bar (global, per team, per class, per intersection)
      -> these are the centring points used by lift `laplace_eb`.

Part B — lift saturation on the full 100-candidate pool (dev queries)
  Under the SIKDD Laplace lift (centre 0.5, cap 0.20) and under laplace_eb, per
  routing scope: fraction of pool lifts at -cap / +cap / exactly 0, number of
  distinct lift values, lift std vs. pool FAISS-score std, and how often a
  no-evidence candidate (lift 0) outranks a judged candidate.

Outputs: results/feedback_calibration/{report.json, reliability_<protocol>.csv,
         scope_priors_<protocol>.csv, saturation.csv, manifest.json}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments.utils import (setup_logging, load_dataset, load_split, get_arg_parser,
                               write_run_manifest, append_registry, make_run_id, split_path)
from src.config import ProjectPaths, LiftConfig, RoutingConfig, EvalConfig, GatingConfig, DEFAULT_METHODS
from src.evaluation.proxy import ReplySimilarity
from src.feedback.loader import load_feedback_rows, FeedbackBundle
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.retrieval.search import retrieve_feedback

BINS = [-0.001, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def part_a(rows: pd.DataFrame, sim: ReplySimilarity, protocol: str, out_dir: Path, log) -> dict:
    rows = rows.copy()
    rows["query_id"] = rows["query_id"].astype(str)
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    ok = rows["query_id"].isin(sim.pos) & rows["candidate_id"].isin(sim.pos)
    rows = rows[ok]
    qi = rows["query_id"].map(sim.pos).to_numpy()
    ci = rows["candidate_id"].map(sim.pos).to_numpy()
    rows["usefulness_sim"] = sim.sim[qi, ci]
    rows["same_reply"] = (rows["query_id"].map(sim.reply_hash) == rows["candidate_id"].map(sim.reply_hash)).astype(float)
    rows["same_group"] = (rows["query_id"].map(sim.group_hash) == rows["candidate_id"].map(sim.group_hash)).astype(float)
    rows["useful_08"] = (rows["usefulness_sim"] >= 0.8).astype(float)

    s = rows["score"].to_numpy(float)
    dist = {
        "n_rows": int(len(rows)), "mean": float(s.mean()), "std": float(s.std()),
        "quantiles": {str(q): float(np.quantile(s, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "pct_le_0.40": float(np.mean(s <= 0.40)), "pct_lt_0.5": float(np.mean(s < 0.5)),
        "pct_ge_0.80": float(np.mean(s >= 0.80)), "pct_exact_0": float(np.mean(s == 0.0)), "pct_exact_1": float(np.mean(s == 1.0)),
        "n_distinct_scores": int(len(np.unique(s))),
    }
    rows["score_bin"] = pd.cut(rows["score"], BINS)
    rel = rows.groupby("score_bin", observed=True).agg(
        n=("score", "size"), mean_score=("score", "mean"), mean_usefulness_sim=("usefulness_sim", "mean"),
        p_same_reply=("same_reply", "mean"), p_same_group=("same_group", "mean"), p_useful_08=("useful_08", "mean"),
    ).reset_index()
    rel["score_bin"] = rel["score_bin"].astype(str)
    rel.to_csv(out_dir / f"reliability_{protocol}.csv", index=False)
    monotone = bool(np.all(np.diff(rel["mean_usefulness_sim"].to_numpy()) >= -1e-9))

    auc = {}
    for target in ("same_reply", "same_group", "useful_08"):
        y = rows[target].to_numpy()
        auc[target] = float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None
    rho = spearmanr(s, rows["usefulness_sim"])[0]

    bundle = FeedbackBundle(rows[["query_id", "candidate_id", "query_class", "query_team", "score"]])
    pri = pd.DataFrame([{"scope": k, "mean": v["mean"], "n": v["n"]} for k, v in bundle.priors.items()])
    pri.to_csv(out_dir / f"scope_priors_{protocol}.csv", index=False)
    team_means = pri[pri["scope"].str.startswith("team:")]["mean"]
    class_means = pri[pri["scope"].str.startswith("class:")]["mean"]

    log.info("[%s] mean score=%.3f  %%<=0.40=%.1f%%  %%>=0.80=%.1f%%  AUC(same_reply)=%s  rho(sim)=%.3f  monotone=%s",
             protocol, dist["mean"], 100 * dist["pct_le_0.40"], 100 * dist["pct_ge_0.80"], auc["same_reply"], rho, monotone)
    return {
        "score_distribution": dist,
        "reliability_monotone_in_usefulness": monotone,
        "reliability_table": rel.to_dict("records"),
        "auc_of_score": auc,
        "spearman_score_vs_usefulness_sim": float(rho),
        "base_rates": {"same_reply": float(rows["same_reply"].mean()), "same_group": float(rows["same_group"].mean()),
                       "useful_08": float(rows["useful_08"].mean())},
        "scope_priors": {
            "global_mean": bundle.prior_mean("global"),
            "team_mean_range": [float(team_means.min()), float(team_means.max())] if len(team_means) else None,
            "class_mean_range": [float(class_means.min()), float(class_means.max())] if len(class_means) else None,
            "n_team_scopes": int(len(team_means)), "n_class_scopes": int(len(class_means)),
            "n_intersection_scopes": int(pri["scope"].str.startswith("intersection:").sum()),
        },
    }


def part_b(paths: ProjectPaths, df: pd.DataFrame, bundle: FeedbackBundle, query_ids: list[str], log) -> pd.DataFrame:
    idx = FAISSIndex(dim=384)
    idx.load(paths.data_processed / "faiss_index" / "faiss.index", paths.data_processed / "faiss_index" / "faiss_metadata.parquet")
    enc = TicketEncoder("all-MiniLM-L6-v2")
    dfi = df.set_index("seq_id")
    lifts = {"laplace": LiftConfig.laplace(), "laplace_eb_k2": LiftConfig.laplace_eb(2.0), "laplace_eb_k10": LiftConfig.laplace_eb(10.0)}
    routings = {"M1_global": RoutingConfig.global_(), "M2_team": RoutingConfig.team_only(),
                "M3_class": RoutingConfig.class_only(), "M4_intersection": RoutingConfig.intersection(),
                "M5_backoff": RoutingConfig.backoff()}
    acc = {(l, r): {"lifts": [], "faiss_std": [], "zero_beats_judged": 0, "n_q": 0} for l in lifts for r in routings}
    embs = {}
    for qid in query_ids:
        row = dfi.loc[qid]
        embs[qid] = enc.encode_ticket(str(row["Title_anon"]), str(row.get("Description_anon", "") or ""))
    for (lname, lcfg) in lifts.items():
        for (rname, rcfg) in routings.items():
            cfg = EvalConfig(experiment_id="calib", lift=lcfg, routing=rcfg, gating=GatingConfig.none(),
                             generator_model="-", judge_model="-")
            a = acc[(lname, rname)]
            for qid in query_ids:
                row = dfi.loc[qid]
                top, pool = retrieve_feedback(idx, embs[qid], 5, {idx.id_to_index(qid)}, bundle.scores, cfg,
                                              str(row["intent_class"]), str(row["Team->Name"]), 100, priors=bundle, return_pool=True)
                lv = pool["feedback_lift_raw"].to_numpy(float)
                a["lifts"].append(lv)
                a["faiss_std"].append(float(pool["faiss_score"].std()))
                # does a zero-evidence candidate sit in the top-5 above a judged (non-zero-evidence) candidate?
                judged = pool["feedback_lift_raw"] != 0
                if (top["feedback_lift_raw"] == 0).any() and judged.any():
                    a["zero_beats_judged"] += 1
                a["n_q"] += 1
    rows = []
    for (lname, rname), a in acc.items():
        lv = np.concatenate(a["lifts"])
        cap = lifts[lname].cap
        rows.append({
            "lift": lname, "routing": rname, "n_queries": a["n_q"], "n_pool_candidates": int(len(lv)),
            "pct_at_minus_cap": float(np.mean(lv <= -cap + 1e-9)), "pct_at_plus_cap": float(np.mean(lv >= cap - 1e-9)),
            "pct_zero": float(np.mean(lv == 0)), "pct_negative": float(np.mean(lv < 0)), "pct_positive": float(np.mean(lv > 0)),
            "n_distinct_lift_values": int(len(np.unique(np.round(lv, 6)))),
            "lift_std": float(lv.std()), "mean_pool_faiss_std": float(np.mean(a["faiss_std"])),
            "lift_std_over_faiss_std": float(lv.std() / max(np.mean(a["faiss_std"]), 1e-9)),
            "pct_queries_zero_evidence_in_top5_with_judged_in_pool": float(a["zero_beats_judged"] / max(a["n_q"], 1)),
        })
        log.info("%-14s %-16s  -cap=%.1f%%  +cap=%.1f%%  zero=%.1f%%  distinct=%d  lift_std/faiss_std=%.2f",
                 lname, rname, 100 * rows[-1]["pct_at_minus_cap"], 100 * rows[-1]["pct_at_plus_cap"],
                 100 * rows[-1]["pct_zero"], rows[-1]["n_distinct_lift_values"], rows[-1]["lift_std_over_faiss_std"])
    return pd.DataFrame(rows)


def main() -> None:
    parser = get_arg_parser("Feedback calibration + lift saturation diagnostics")
    parser.add_argument("--protocols", nargs="*", default=["conditioned", "blind"])
    parser.add_argument("--sim-model", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--saturation-queries", type=int, default=319, help="dev queries used for Part B (0 to skip)")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("calibration")
    paths = ProjectPaths()
    out_dir = paths.results / "feedback_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("feedback_calibration")

    df = load_dataset()
    sim = ReplySimilarity(df, args.sim_model)
    split = load_split(args.seed, args.regime)
    report = {"sim_model": args.sim_model, "protocols": {}}
    inputs = [paths.dataset_parquet, split_path(args.seed, args.regime)]
    for protocol in args.protocols:
        db = paths.data_processed / f"feedback_{protocol}.db"
        if not db.exists():
            log.warning("missing %s", db)
            continue
        inputs.append(db)
        rows = load_feedback_rows(db)
        report["protocols"][protocol] = part_a(rows, sim, protocol, out_dir, log)

    if args.saturation_queries > 0:
        db = paths.data_processed / "feedback_conditioned.db"
        bundle = FeedbackBundle(load_feedback_rows(db, exclude_query_ids=set(split["dev"])))
        qids = split["dev"][: args.saturation_queries]
        log.info("Part B: lift saturation on %d dev queries x 100-candidate pools", len(qids))
        sat = part_b(paths, df, bundle, qids, log)
        sat.to_csv(out_dir / "saturation.csv", index=False)
        report["saturation"] = sat.to_dict("records")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = write_run_manifest(out_dir, script="experiments/10_feedback_calibration.py", args=args, inputs=inputs, run_id=run_id)
    g = report["protocols"].get("conditioned", {})
    append_registry(run_id=run_id, phase="P1.4-calibration", script="10_feedback_calibration.py", out_dir=out_dir, manifest=manifest,
                    headline_metric="AUC(score->same_reply) conditioned",
                    headline_value=g.get("auc_of_score", {}).get("same_reply"))
    print("\n=== FEEDBACK CALIBRATION ===")
    for p, r in report["protocols"].items():
        d = r["score_distribution"]
        print(f"[{p}] n={d['n_rows']} mean={d['mean']:.3f} %<=0.40={100*d['pct_le_0.40']:.1f} %>=0.80={100*d['pct_ge_0.80']:.1f} "
              f"AUC same_reply={r['auc_of_score']['same_reply']} group={r['auc_of_score']['same_group']} "
              f"rho(sim)={r['spearman_score_vs_usefulness_sim']:.3f} monotone={r['reliability_monotone_in_usefulness']} "
              f"global p_bar={r['scope_priors']['global_mean']:.3f} team p_bar range={r['scope_priors']['team_mean_range']}")
    if "saturation" in report:
        print(pd.DataFrame(report["saturation"])[["lift", "routing", "pct_at_minus_cap", "pct_at_plus_cap", "pct_zero",
                                                    "n_distinct_lift_values", "lift_std_over_faiss_std"]].to_string(index=False))
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
