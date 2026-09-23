#!/usr/bin/env python3
"""
11_retriever_ladder.py — P2
============================
Retrieval-only grid over retrievers x feedback routings x lift variants, scored
with the offline proxy (no API calls). Produces the feedback-gain-vs-retriever-
strength curve and the feedback-coverage confound diagnostics.

    python experiments/11_retriever_ladder.py --split dev
    python experiments/11_retriever_ladder.py --split dev --retrievers dense_minilm bm25 hybrid_rrf ce_rerank

Grid (defaults)
  retrievers : dense_minilm bm25 hybrid_rrf dense_bge hybrid_bge_rrf ce_rerank
  routings   : none M1_global M2_team M3_class M4_intersection M5_backoff
  lifts      : laplace | laplace_eb (kappa 2, 10)
  scaling    : absolute (cosine retrievers only) | pool_std lambda in {0.5, 1, 2}

Per cell: mean proxy deltas (minilm + bge reply similarity, same-reply and
same-group hits), bootstrap CI + Wilcoxon of the primary proxy delta, % top-1
changed, feedback coverage of the pool per scope.

Outputs: results/retriever_ladder/{grid.csv, per_ticket.parquet, ladder_curve.csv,
         pools/<retriever>_<split>_seed<seed>.parquet, manifest.json}
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiments.utils import (setup_logging, load_dataset, load_split, split_path, get_arg_parser,
                               write_run_manifest, append_registry, make_run_id, set_seeds)
from src.config import ProjectPaths, LiftConfig, RoutingConfig, RETRIEVERS
from src.evaluation.ladder import collect_pools, build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.evaluation.reporting import bootstrap_ci
from src.feedback.loader import load_feedback_bundle
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.retrieval.retrievers import build_retriever
from src.retrieval.semantic import TicketTextSimilarity

ROUTINGS = {
    "none": RoutingConfig(name="none"),
    "M1_global": RoutingConfig.global_(),
    "M2_team": RoutingConfig.team_only(),
    "M3_class": RoutingConfig.class_only(),
    "M4_intersection": RoutingConfig.intersection(),
    "M5_backoff": RoutingConfig.backoff(),
}
COSINE_RETRIEVERS = {"dense_minilm", "dense_bge"}
DEFAULT_SEMANTIC_TAUS = [0.5, 0.6, 0.7, 0.8]


def build_routings(semantic_taus: list[float]) -> dict[str, RoutingConfig]:
    """Base routings plus one semantic-filter variant per tau (P5)."""
    out = dict(ROUTINGS)
    for tau in semantic_taus:
        out[f"semantic_intersection_tau{tau:g}"] = RoutingConfig.semantic_intersection(tau)
        out[f"semantic_backoff_tau{tau:g}"] = RoutingConfig.semantic_backoff(tau, min_evidence=2.0)
    return out


def lift_variants(scale_modes: list[str], lambdas: list[float], kappas: list[float],
                  tanh_sensitivities: list[float], lcb_ks: list[float]) -> dict[str, LiftConfig]:
    out = {}
    for sm in scale_modes:
        lams = lambdas if sm == "pool_std" else [1.0]
        for lam in lams:
            tag = "abs" if sm == "absolute" else f"std{lam:g}"
            out[f"laplace|{tag}"] = LiftConfig(name="laplace", scale_mode=sm, pool_lambda=lam)
            for k in kappas:
                out[f"laplace_eb_k{k:g}|{tag}"] = LiftConfig.laplace_eb(prior_strength=k, scale_mode=sm, pool_lambda=lam)
            for s in tanh_sensitivities:
                out[f"tanh_s{s:g}|{tag}"] = LiftConfig(name="tanh", multiplier=0.80, cap=0.20,
                                                       sensitivity=s, scale_mode=sm, pool_lambda=lam)
            for k in lcb_ks:
                out[f"bayesian_lcb_k{k:g}|{tag}"] = LiftConfig(name="bayesian_lcb", multiplier=0.80, cap=0.20,
                                                               lcb_k=k, scale_mode=sm, pool_lambda=lam)
    return out


def main() -> None:
    parser = get_arg_parser("Retriever ladder (retrieval-only, proxy-scored)")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--retrievers", nargs="*", default=["dense_minilm", "bm25", "hybrid_rrf", "dense_bge", "hybrid_bge_rrf", "ce_rerank"])
    parser.add_argument("--routings", nargs="*", default=list(ROUTINGS.keys()))
    parser.add_argument("--scale-modes", nargs="*", default=["absolute", "pool_std"])
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--kappas", nargs="*", type=float, default=[2.0, 10.0])
    parser.add_argument("--tanh-sensitivities", nargs="*", type=float, default=[3.5],
                        help="tanh lift sensitivity values (lift-formula ablation)")
    parser.add_argument("--lcb-ks", nargs="*", type=float, default=[1.0],
                        help="Bayesian-LCB k values (lift-formula ablation)")
    parser.add_argument("--sim-models", nargs="*", default=["minilm", "bge"])
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--feedback-protocol", default="conditioned")
    parser.add_argument("--semantic-taus", nargs="*", type=float, default=DEFAULT_SEMANTIC_TAUS,
                        help="Taus for the semantic relevance-filter routings")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("ladder")
    set_seeds(args.seed)
    for r in args.retrievers:
        if r not in RETRIEVERS:
            raise SystemExit(f"unknown retriever {r}")

    routings = build_routings(args.semantic_taus)
    unknown = [r for r in args.routings if r not in routings]
    if unknown:
        raise SystemExit(f"unknown routing(s) {unknown}; available: {sorted(routings)}")
    needs_semantic = any(name.startswith("semantic") for name in args.routings)

    paths = ProjectPaths()
    out_dir = paths.results / ("retriever_ladder" + (f"_{args.tag}" if args.tag else ""))
    pools_dir = paths.results / "retriever_ladder" / "pools"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("retriever_ladder")

    df = load_dataset()
    split = load_split(args.seed, args.regime)
    qids = split[args.split][: args.limit] if args.limit else split[args.split]
    queries = df.set_index("seq_id").loc[qids].reset_index()
    sims = {m: ReplySimilarity(df, m) for m in args.sim_models}
    primary = args.sim_models[0]

    fb_db = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    bundle = load_feedback_bundle(fb_db, exclude_query_ids=set(split[args.split]))
    per_query_loo = args.split == "train"

    idx = FAISSIndex(dim=384)
    idx.load(paths.data_processed / "faiss_index" / "faiss.index", paths.data_processed / "faiss_index" / "faiss_metadata.parquet")
    enc = TicketEncoder("all-MiniLM-L6-v2")
    text_sim = TicketTextSimilarity(df, encoder=enc) if needs_semantic else None

    lifts = lift_variants(args.scale_modes, args.lambdas, args.kappas,
                          args.tanh_sensitivities, args.lcb_ks)
    grid_rows, ticket_frames, curve_rows = [], [], []
    for rname in args.retrievers:
        log.info("=== retriever: %s ===", rname)
        retr = build_retriever(rname, paths, idx, enc, ce_pool=args.search_k)
        pool_path = pools_dir / f"{rname}_{args.split}_seed{args.seed}{'_' + args.regime if args.regime != 'random' else ''}.parquet"
        pools = collect_pools(retr, queries, enc, args.search_k, cache_path=pool_path, id_to_index=idx.id_to_index)
        T = build_pool_tensor(pools, queries, bundle, sims, per_query_loo=per_query_loo, text_sim=text_sim)
        cov = T.coverage()
        base_top = T.baseline_top(args.top_k)
        base_m = {tag: T.metrics(base_top, tag) for tag in sims}
        curve_rows.append({"retriever": rname, "split": args.split, "n": len(T.query_ids),
                           **{f"base_{tag}_{k}": float(v.mean()) for tag, m in base_m.items() for k, v in m.items()},
                           **{f"coverage_{s}": float(cov[:, j].mean()) for j, s in enumerate(("global", "class", "team", "intersection"))}})
        for (route_name, lift_name) in itertools.product(args.routings, lifts):
            lcfg = lifts[lift_name]
            if route_name == "none" and lift_name != next(iter(lifts)):
                continue
            if lcfg.scale_mode == "absolute" and rname not in COSINE_RETRIEVERS and route_name != "none":
                continue  # absolute cosine-unit lifts are meaningless for BM25 / RRF / CE scores
            routing = routings[route_name]
            lift = T.routed_lift(lcfg, routing) if route_name != "none" else np.zeros_like(T.score)
            top = T.rerank(lift, T.scale_factor(lcfg), args.top_k)
            rec = {"retriever": rname, "routing": route_name, "lift": lift_name, "split": args.split, "n": len(T.query_ids)}
            tf = pd.DataFrame({"ticket_id": T.query_ids, "retriever": rname, "routing": route_name, "lift": lift_name,
                               "expected_class": T.query_class, "expected_team": T.query_team,
                               "top1_changed": (top[:, 0] != base_top[:, 0]).astype(float),
                               "coverage_team": cov[:, 2], "coverage_intersection": cov[:, 3],
                               "base_score_top1": T.score[np.arange(len(T.query_ids)), base_top[:, 0]]})
            for tag in sims:
                m = T.metrics(top, tag)
                for k, v in m.items():
                    d = v - base_m[tag][k]
                    tf[f"{tag}_d_{k}"] = d
                    rec[f"{tag}_d_{k}"] = float(d.mean())
                    rec[f"{tag}_fb_{k}"] = float(v.mean())
            d = tf[f"{primary}_d_proxy_top1"].to_numpy()
            ci = bootstrap_ci(d, n_resamples=2000)
            rec.update({"primary_ci_lo": ci["ci_lower"], "primary_ci_hi": ci["ci_upper"],
                        "primary_wilcoxon_p": float(wilcoxon(d)[1]) if np.any(d != 0) else None,
                        "pct_top1_changed": float(tf["top1_changed"].mean()),
                        "pct_improved": float(np.mean(d > 0)), "pct_worsened": float(np.mean(d < 0))})
            grid_rows.append(rec)
            ticket_frames.append(tf)
            if lift_name.startswith("laplace|") or route_name == "none":
                log.info("  %-16s %-22s d_top1=%+.4f  hit1=%+.4f  changed=%.0f%%", route_name, lift_name,
                         rec[f"{primary}_d_proxy_top1"], rec[f"{primary}_d_same_reply_hit1"], 100 * rec["pct_top1_changed"])

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "grid.csv", index=False)
    pd.concat(ticket_frames, ignore_index=True).to_parquet(out_dir / "per_ticket.parquet", index=False)
    curve = pd.DataFrame(curve_rows)
    # best feedback gain per retriever (primary proxy) -> the ladder curve
    best = grid[grid["routing"] != "none"].sort_values(f"{primary}_d_proxy_top1", ascending=False).groupby("retriever").head(1)
    curve = curve.merge(best[["retriever", "routing", "lift", f"{primary}_d_proxy_top1", "primary_ci_lo", "primary_ci_hi", "primary_wilcoxon_p"]]
                        .rename(columns={"routing": "best_routing", "lift": "best_lift", f"{primary}_d_proxy_top1": "best_gain"}), on="retriever", how="left")
    for rn in ("M2_team", "M4_intersection", "M5_backoff"):
        sub = grid[(grid["routing"] == rn)].sort_values(f"{primary}_d_proxy_top1", ascending=False).groupby("retriever").head(1)
        curve = curve.merge(sub[["retriever", f"{primary}_d_proxy_top1"]].rename(columns={f"{primary}_d_proxy_top1": f"best_gain_{rn}"}), on="retriever", how="left")
    curve.to_csv(out_dir / "ladder_curve.csv", index=False)

    manifest = write_run_manifest(out_dir, script="experiments/11_retriever_ladder.py", args=args,
                                  inputs=[paths.dataset_parquet, split_path(args.seed, args.regime), fb_db],
                                  extra={"n_cells": len(grid), "primary_proxy": f"{primary}_d_proxy_top1"}, run_id=run_id)
    append_registry(run_id=run_id, phase="P2-ladder", script="11_retriever_ladder.py", out_dir=out_dir, manifest=manifest,
                    headline_metric="n_cells", headline_value=len(grid))

    pd.set_option("display.width", 250)
    print("\n=== LADDER CURVE (retriever strength vs best feedback gain; primary proxy = %s reply-sim of top-1) ===" % primary)
    cols = ["retriever", f"base_{primary}_proxy_top1", f"base_{primary}_same_reply_hit1", "coverage_team", "coverage_intersection",
            "best_routing", "best_lift", "best_gain", "primary_ci_lo", "primary_ci_hi", "best_gain_M2_team", "best_gain_M4_intersection", "best_gain_M5_backoff"]
    print(curve[[c for c in cols if c in curve]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== TOP-15 CELLS ===")
    show = ["retriever", "routing", "lift", f"{primary}_d_proxy_top1", "primary_ci_lo", "primary_ci_hi", "primary_wilcoxon_p",
            f"{primary}_d_same_reply_hit1", "pct_top1_changed"]
    if "bge" in sims:
        show.insert(4, "bge_d_proxy_top1")
    print(grid[grid["routing"] != "none"].sort_values(f"{primary}_d_proxy_top1", ascending=False).head(15)[show]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
