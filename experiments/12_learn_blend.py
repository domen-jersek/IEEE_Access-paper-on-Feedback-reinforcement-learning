#!/usr/bin/env python3
"""
12_learn_blend.py — P3
=======================
Learns granularity parameters on the TRAIN split (per-query leave-one-out
feedback) using the offline proxy, then evaluates them on DEV. No API calls.

Learned objects
  1. blend weights  w = (w_global, w_class, w_team, w_intersection) on a grid,
     objective = mean proxy_top1 delta (reply similarity of the top-1 candidate),
  2. backoff threshold min_evidence,
  3. laplace_eb prior strength kappa (and pool-std lambda when --scale-mode pool_std),
  4. an "adaptive" variant: separate weights for queries with / without
     intersection-scope evidence in the baseline top-5,
  5. bootstrap stability of the chosen weights (train resamples).

Also emits per-query gate feature tables (retrieval-only features + proxy labels)
for train and dev, consumed by 05_gate_cv.py --features-parquet.

Outputs: results/blend/{learned_weights.json, grid_train.csv, dev_eval.csv,
         bootstrap_weights.csv, gate_features_train.parquet, gate_features_dev.parquet, manifest.json}
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

from experiments.utils import (setup_logging, load_dataset, load_split, split_path, get_arg_parser,
                               write_run_manifest, append_registry, make_run_id, set_seeds)
from src.config import ProjectPaths, LiftConfig, RoutingConfig
from src.evaluation.ladder import collect_pools, build_pool_tensor, PoolTensor, SCOPE_IDX
from src.evaluation.proxy import ReplySimilarity
from src.evaluation.reporting import bootstrap_ci
from src.feedback.loader import load_feedback_bundle
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.retrieval.retrievers import build_retriever

SCOPES = ("global", "class", "team", "intersection")


def objective(T: PoolTensor, lift: np.ndarray, factor: np.ndarray, base_top: np.ndarray, sim: str, k: int, metric: str) -> np.ndarray:
    top = T.rerank(lift, factor, k)
    return T.metrics(top, sim)[metric] - T.metrics(base_top, sim)[metric]


def weight_grid(step: float) -> list[tuple]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    return [w for w in itertools.product(vals, repeat=4) if sum(w) > 0]


def gate_features(T: PoolTensor, L: np.ndarray, lift: np.ndarray, top: np.ndarray, base_top: np.ndarray, sim: str) -> pd.DataFrame:
    Q = len(T.query_ids)
    rows = np.arange(Q)[:, None]
    sc = np.where(T.mask, T.score, np.nan)
    b5 = T.score[rows, base_top]
    n = T.n
    f = {
        "ticket_id": T.query_ids, "expected_class": T.query_class, "expected_team": T.query_team,
        "top1_score": b5[:, 0], "top5_mean_score": b5.mean(1), "top5_min_score": b5.min(1), "top5_std_score": b5.std(1),
        "retrieval_margin": b5[:, 0] - b5[:, 1], "pool_std": np.nanstd(sc, axis=1),
        "retrieval_overlap": np.array([len(set(top[i]) & set(base_top[i])) for i in range(Q)], float),
        "top1_changed": (top[:, 0] != base_top[:, 0]).astype(float),
        "max_lift": lift.max(1), "mean_lift_top5": lift[rows, top].mean(1),
        "n_positive_lifts_top5": (lift[rows, top] > 0).sum(1), "n_negative_lifts_top5": (lift[rows, top] < 0).sum(1),
        "lift_spread_pool": lift.max(1) - lift.min(1),
    }
    for s in SCOPES:
        j = SCOPE_IDX[s]
        f[f"evidence_{s}_top5"] = n[rows, base_top][..., j].sum(1)
        f[f"coverage_{s}"] = ((n[..., j] > 0) & T.mask).sum(1) / np.maximum(T.mask.sum(1), 1)
        f[f"max_lift_{s}"] = L[..., j].max(1)
    m_fb, m_b = T.metrics(top, sim), T.metrics(base_top, sim)
    f["proxy_delta_top1"] = m_fb["proxy_top1"] - m_b["proxy_top1"]
    f["proxy_label_improved"] = (f["proxy_delta_top1"] > 0).astype(float)
    f["proxy_label_harmed"] = (f["proxy_delta_top1"] < 0).astype(float)
    f["base_proxy_top1"] = m_b["proxy_top1"]
    return pd.DataFrame(f)


def main() -> None:
    parser = get_arg_parser("Learn blend / backoff parameters on train, evaluate on dev (proxy)")
    parser.add_argument("--retriever", default="dense_minilm")
    parser.add_argument("--sim-model", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--metric", default="proxy_top1", choices=["proxy_top1", "proxy_best5", "same_reply_hit1"])
    parser.add_argument("--lift", default="laplace", choices=["laplace", "laplace_eb"])
    parser.add_argument("--kappas", nargs="*", type=float, default=[2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--scale-mode", default="absolute", choices=["absolute", "pool_std"])
    parser.add_argument("--lambdas", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--min-evidences", nargs="*", type=float, default=[1, 2, 3, 5, 8])
    parser.add_argument("--step", type=float, default=0.25, help="blend weight grid step")
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--feedback-protocol", default="conditioned")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("learn_blend")
    set_seeds(args.seed)

    paths = ProjectPaths()
    out_dir = paths.results / ("blend" + (f"_{args.tag}" if args.tag else ""))
    pools_dir = paths.results / "retriever_ladder" / "pools"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("learn_blend")

    df = load_dataset()
    split = load_split(args.seed, args.regime)
    sims = {args.sim_model: ReplySimilarity(df, args.sim_model)}
    sim = args.sim_model
    idx = FAISSIndex(dim=384)
    idx.load(paths.data_processed / "faiss_index" / "faiss.index", paths.data_processed / "faiss_index" / "faiss_metadata.parquet")
    enc = TicketEncoder("all-MiniLM-L6-v2")
    retr = build_retriever(args.retriever, paths, idx, enc, ce_pool=args.search_k)
    fb_db = paths.data_processed / f"feedback_{args.feedback_protocol}.db"

    tensors, base_tops = {}, {}
    for sp in ("train", "dev"):
        queries = df.set_index("seq_id").loc[split[sp]].reset_index()
        pool_path = pools_dir / f"{args.retriever}_{sp}_seed{args.seed}{'_' + args.regime if args.regime != 'random' else ''}.parquet"
        pools = collect_pools(retr, queries, enc, args.search_k, cache_path=pool_path, id_to_index=idx.id_to_index)
        # TRAIN: the DB contains train queries' own judgements -> per-query LOO subtraction.
        # DEV : exclude dev queries at load time (same guard as 04_evaluate.py).
        bundle = load_feedback_bundle(fb_db, exclude_query_ids=set(split["dev"]) if sp == "dev" else set())
        tensors[sp] = build_pool_tensor(pools, queries, bundle, sims, per_query_loo=(sp == "train"))
        base_tops[sp] = tensors[sp].baseline_top(args.top_k)
        log.info("%s tensor: %d queries x %d candidates", sp, *tensors[sp].score.shape)

    Ttr, Tdv = tensors["train"], tensors["dev"]
    lift_cfgs = {}
    lams = args.lambdas if args.scale_mode == "pool_std" else [1.0]
    for lam in lams:
        if args.lift == "laplace":
            lift_cfgs[f"laplace|{args.scale_mode}{lam:g}"] = LiftConfig(name="laplace", scale_mode=args.scale_mode, pool_lambda=lam)
        else:
            for k in args.kappas:
                lift_cfgs[f"laplace_eb_k{k:g}|{args.scale_mode}{lam:g}"] = LiftConfig.laplace_eb(k, args.scale_mode, lam)

    grid_rows = []
    fixed = {"M1_global": RoutingConfig.global_(), "M2_team": RoutingConfig.team_only(),
             "M3_class": RoutingConfig.class_only(), "M4_intersection": RoutingConfig.intersection()}
    W = weight_grid(args.step)
    log.info("Searching %d lift cfgs x (%d fixed + %d backoff + %d blend weights)", len(lift_cfgs), len(fixed), len(args.min_evidences), len(W))
    for lname, lcfg in lift_cfgs.items():
        f_tr, f_dv = Ttr.scale_factor(lcfg), Tdv.scale_factor(lcfg)
        L_tr, L_dv = Ttr.lifts(lcfg), Tdv.lifts(lcfg)
        def ev(routing_name, lift_tr, lift_dv, **extra):
            d_tr = objective(Ttr, lift_tr, f_tr, base_tops["train"], sim, args.top_k, args.metric)
            d_dv = objective(Tdv, lift_dv, f_dv, base_tops["dev"], sim, args.top_k, args.metric)
            grid_rows.append({"lift": lname, "routing": routing_name, **extra,
                              "train_gain": float(d_tr.mean()), "dev_gain": float(d_dv.mean()),
                              "dev_pct_improved": float(np.mean(d_dv > 0)), "dev_pct_worsened": float(np.mean(d_dv < 0))})
        for rn, rc in fixed.items():
            ev(rn, Ttr.routed_lift(lcfg, rc), Tdv.routed_lift(lcfg, rc))
        for me in args.min_evidences:
            rc = RoutingConfig.backoff(min_evidence=me)
            ev("M5_backoff", Ttr.routed_lift(lcfg, rc), Tdv.routed_lift(lcfg, rc), min_evidence=me)
        for w in W:
            wv = np.array(w)
            ev("M6_blend", L_tr @ wv, L_dv @ wv, w_global=w[0], w_class=w[1], w_team=w[2], w_intersection=w[3])
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "grid_train.csv", index=False)

    # --- select on TRAIN only ---
    best_blend = grid[grid["routing"] == "M6_blend"].sort_values("train_gain", ascending=False).iloc[0]
    best_backoff = grid[grid["routing"] == "M5_backoff"].sort_values("train_gain", ascending=False).iloc[0]
    best_fixed = grid[grid["routing"].isin(fixed)].sort_values("train_gain", ascending=False).iloc[0]
    lcfg = lift_cfgs[best_blend["lift"]]
    w_best = np.array([best_blend[f"w_{s}"] for s in SCOPES])

    # --- adaptive: separate weights by "intersection evidence present in baseline top-5" ---
    def bucket(T, base_top):
        rows = np.arange(len(T.query_ids))[:, None]
        return (T.n[rows, base_top][..., SCOPE_IDX["intersection"]].sum(1) > 0)
    b_tr, b_dv = bucket(Ttr, base_tops["train"]), bucket(Tdv, base_tops["dev"])
    L_tr, L_dv = Ttr.lifts(lcfg), Tdv.lifts(lcfg)
    f_tr, f_dv = Ttr.scale_factor(lcfg), Tdv.scale_factor(lcfg)
    adaptive = {}
    lift_dv_adapt = np.zeros_like(Tdv.score)
    for flag in (True, False):
        best_w, best_v = None, -np.inf
        for w in W:
            d = objective(Ttr, L_tr @ np.array(w), f_tr, base_tops["train"], sim, args.top_k, args.metric)
            v = d[b_tr == flag].mean() if (b_tr == flag).any() else -np.inf
            if v > best_v:
                best_v, best_w = v, w
        adaptive["with_intersection_evidence" if flag else "without_intersection_evidence"] = {
            "weights": dict(zip(SCOPES, map(float, best_w))), "train_gain_bucket": float(best_v),
            "n_train": int((b_tr == flag).sum()), "n_dev": int((b_dv == flag).sum())}
        lift_dv_adapt[b_dv == flag] = (L_dv @ np.array(best_w))[b_dv == flag]
    d_adapt = objective(Tdv, lift_dv_adapt, f_dv, base_tops["dev"], sim, args.top_k, args.metric)

    # --- bootstrap stability of blend weights (train resamples, coarse grid) ---
    rng = np.random.default_rng(args.seed)
    coarse = weight_grid(0.5)
    boot = []
    Q = len(Ttr.query_ids)
    per_w = {w: objective(Ttr, L_tr @ np.array(w), f_tr, base_tops["train"], sim, args.top_k, args.metric) for w in coarse}
    for _ in range(args.bootstrap):
        ix = rng.integers(0, Q, Q)
        w_b = max(coarse, key=lambda w: per_w[w][ix].mean())
        boot.append(dict(zip(SCOPES, w_b)))
    boot_df = pd.DataFrame(boot)
    boot_df.to_csv(out_dir / "bootstrap_weights.csv", index=False)

    # --- dev evaluation of the selected configs ---
    dev_rows = []
    def dev_eval(name, lift_dv, extra=None):
        d = objective(Tdv, lift_dv, f_dv, base_tops["dev"], sim, args.top_k, args.metric)
        ci = bootstrap_ci(d, n_resamples=5000)
        dev_rows.append({"config": name, "dev_gain": float(d.mean()), "ci_lo": ci["ci_lower"], "ci_hi": ci["ci_upper"],
                         "pct_improved": float(np.mean(d > 0)), "pct_worsened": float(np.mean(d < 0)), **(extra or {})})
        return d
    for rn, rc in fixed.items():
        dev_eval(rn, Tdv.routed_lift(lcfg, rc))
    dev_eval(f"M5_backoff(min_ev={best_backoff['min_evidence']:g})", Tdv.routed_lift(lcfg, RoutingConfig.backoff(min_evidence=float(best_backoff["min_evidence"]))))
    d_blend = dev_eval("M6_blend(learned)", L_dv @ w_best, {"weights": json.dumps(dict(zip(SCOPES, map(float, w_best))))})
    dev_eval("M6_blend_adaptive", lift_dv_adapt)
    dev_df = pd.DataFrame(dev_rows)
    dev_df.to_csv(out_dir / "dev_eval.csv", index=False)

    learned = {
        "retriever": args.retriever, "sim_model": sim, "metric": args.metric, "selected_on": "train (per-query LOO feedback)",
        "lift": lcfg.to_dict() | {"name": lcfg.name, "prior_strength": lcfg.prior_strength, "scale_mode": lcfg.scale_mode, "pool_lambda": lcfg.pool_lambda},
        "weights": dict(zip(SCOPES, map(float, w_best))),
        "train_gain": float(best_blend["train_gain"]), "dev_gain": float(d_blend.mean()),
        "best_fixed_on_train": {"routing": best_fixed["routing"], "lift": best_fixed["lift"], "train_gain": float(best_fixed["train_gain"]), "dev_gain": float(best_fixed["dev_gain"])},
        "best_backoff_on_train": {"min_evidence": float(best_backoff["min_evidence"]), "lift": best_backoff["lift"], "train_gain": float(best_backoff["train_gain"]), "dev_gain": float(best_backoff["dev_gain"])},
        "adaptive": adaptive, "adaptive_dev_gain": float(d_adapt.mean()),
        "bootstrap_weight_means": boot_df.mean().to_dict(), "bootstrap_weight_mode_freq": {s: float(boot_df[s].value_counts(normalize=True).iloc[0]) for s in SCOPES},
    }
    (out_dir / "learned_weights.json").write_text(json.dumps(learned, indent=2), encoding="utf-8")

    # --- gate feature tables (train + dev) for 05_gate_cv.py ---
    for sp, T, bt in (("train", Ttr, base_tops["train"]), ("dev", Tdv, base_tops["dev"])):
        L = T.lifts(lcfg)
        lift = L @ w_best
        top = T.rerank(lift, T.scale_factor(lcfg), args.top_k)
        gf = gate_features(T, L, lift, top, bt, sim)
        gf["routing"] = "M6_blend(learned)"
        gf.to_parquet(out_dir / f"gate_features_{sp}.parquet", index=False)
        # also for the best fixed routing (comparison target for the gate)
        lift2 = T.routed_lift(lcfg, fixed[best_fixed["routing"]])
        top2 = T.rerank(lift2, T.scale_factor(lcfg), args.top_k)
        gf2 = gate_features(T, L, lift2, top2, bt, sim)
        gf2["routing"] = best_fixed["routing"]
        gf2.to_parquet(out_dir / f"gate_features_{sp}_{best_fixed['routing']}.parquet", index=False)

    manifest = write_run_manifest(out_dir, script="experiments/12_learn_blend.py", args=args,
                                  inputs=[paths.dataset_parquet, split_path(args.seed, args.regime), fb_db],
                                  extra={"n_grid": len(grid)}, run_id=run_id)
    append_registry(run_id=run_id, phase="P3-blend", script="12_learn_blend.py", out_dir=out_dir, manifest=manifest,
                    headline_metric=f"dev_gain M6_blend ({args.metric})", headline_value=round(float(d_blend.mean()), 5))

    pd.set_option("display.width", 200)
    print("\n=== LEARNED ON TRAIN, EVALUATED ON DEV (metric: %s delta, sim=%s, retriever=%s) ===" % (args.metric, sim, args.retriever))
    print(dev_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nblend weights: {learned['weights']}   lift: {best_blend['lift']}")
    print(f"bootstrap weight means: { {k: round(v, 2) for k, v in learned['bootstrap_weight_means'].items()} }")
    print(f"adaptive: {json.dumps(adaptive, indent=1)}")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
