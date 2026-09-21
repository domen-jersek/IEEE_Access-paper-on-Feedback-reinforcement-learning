#!/usr/bin/env python3
"""
13_feedback_volume_curve.py — P3 extension
===========================================
Measures feedback-gain as a function of how much feedback is available:
how the gain per routing scope grows when only a subset of the judged
training queries are used.

A real system starts with zero feedback. Practitioners need to know:
  * How many judged queries before each scope delivers a visible gain?
  * Does intersection collapse at low volume (practically zero evidence per cell)?
  * At what volume does M5 backoff overtake M4 intersection?
  * Does global pooling ever help?

For each volume fraction f (1%, 2%, 5%, 10%, 25%, 50%, 75%, 100%)
and for multiple random seeds per f, the training-query set is sub-sampled,
a FeedbackBundle is built from only those queries' rows, and the cached
dev candidate pools (from dense_minilm) are re-scored with the same routing /
lift configurations as the ladder. All proxy-scored, zero API cost.

Outputs: results/feedback_volume/{curve.csv, per_seed.csv, summary.csv, manifest_<run_id>.json}
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
from src.evaluation.ladder import build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.feedback.loader import load_feedback_rows, FeedbackBundle
DEFAULT_SEEDS = list(range(1000, 1010))
DEFAULT_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
SCOPES = ("global", "class", "team", "intersection")


def eval_configs():
    out = {}
    for lcfg, lname in [(LiftConfig.laplace(), "laplace"),
                         (LiftConfig.laplace_eb(2.0), "laplace_eb")]:
        for rname, rc in [("M1_global", RoutingConfig.global_()),
                          ("M2_team", RoutingConfig.team_only()),
                          ("M4_intersection", RoutingConfig.intersection()),
                          ("M5_backoff2", RoutingConfig.backoff(min_evidence=2))]:
            out[(lname, rname)] = lcfg, rc
    return out


def _parse_csv_values(raw: str, cast, name: str) -> list:
    try:
        values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --{name}: {raw}") from exc
    if not values:
        raise ValueError(f"--{name} must contain at least one value")
    return values


def summarise_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarise each lift/routing/fraction cell with seed uncertainty."""
    keys = ["fraction", "n_keep", "lift", "routing"]
    metrics = ["mean_delta_top1", "mean_delta_best5", "pct_top1_changed", *[f"coverage_{s}" for s in SCOPES]]
    summary = per_seed.groupby(keys, as_index=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = ["_".join(part for part in col if part) for col in summary.columns.to_flat_index()]
    summary = summary.rename(columns={f"{key}_": key for key in keys})
    for metric in metrics:
        se = summary[f"{metric}_std"] / np.sqrt(summary[f"{metric}_count"])
        summary[f"{metric}_ci_low"] = summary[f"{metric}_mean"] - 1.96 * se
        summary[f"{metric}_ci_high"] = summary[f"{metric}_mean"] + 1.96 * se
    return summary


def main():
    parser = get_arg_parser("Feedback-volume learning curve on dev pools (proxy)")
    parser.add_argument("--sim-model", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--retriever", default="dense_minilm")
    parser.add_argument("--feedback-protocol", default="conditioned")
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated subsampling seeds")
    parser.add_argument("--fractions", default=",".join(map(str, DEFAULT_FRACTIONS)),
                        help="Comma-separated feedback fractions in (0, 1]")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("feedback_volume")
    set_seeds(42)
    seeds = _parse_csv_values(args.seeds, int, "seeds")
    fractions = _parse_csv_values(args.fractions, float, "fractions")
    if any(f <= 0.0 or f > 1.0 for f in fractions):
        parser.error("--fractions values must be in (0, 1]")
    paths = ProjectPaths()
    out_dir = paths.results / ("feedback_volume" + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("feedback_volume")

    df = load_dataset()
    sim = ReplySimilarity(df, args.sim_model)
    sims = {args.sim_model: sim}
    split = load_split(42)
    dev_qids = split["dev"]
    queries = df.set_index("seq_id").loc[dev_qids].reset_index()
    pool_path = paths.results / "retriever_ladder" / "pools" / f"{args.retriever}_dev_seed42.parquet"
    if not pool_path.exists():
        raise FileNotFoundError(f"Run 11_retriever_ladder.py first to produce {pool_path}")

    fb_db = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    all_rows = load_feedback_rows(fb_db)
    train_qids = set(split["train"])
    all_rows = all_rows[all_rows["query_id"].isin(train_qids)].copy()
    n_train = len(train_qids)
    log.info("Full train feedback: %d queries, %d rows", n_train, len(all_rows))

    base_bundle = FeedbackBundle(all_rows)
    T_full = build_pool_tensor(pd.read_parquet(pool_path), queries, base_bundle, sims)
    base_top = T_full.baseline_top(args.top_k)

    configs = eval_configs()
    rows, per_seed_rows = [], []
    sorted_ids = sorted(train_qids)
    pool = pd.read_parquet(pool_path)
    base_metrics = T_full.metrics(base_top, args.sim_model)

    for fraction in fractions:
        n_keep = max(1, int(fraction * n_train))
        accum = {"fraction": fraction, "n_train_queries": n_keep}
        for seed in seeds:
            rng_sub = np.random.default_rng(seed)
            chosen = sorted(rng_sub.choice(sorted_ids, size=n_keep, replace=False))
            sub_rows = all_rows[all_rows["query_id"].isin(chosen)]
            bundle = FeedbackBundle(sub_rows, mode="continuous")
            T = build_pool_tensor(pool, queries, bundle, sims)
            cov = T.coverage()
            for j, s in enumerate(SCOPES):
                accum.setdefault(f"coverage_{s}", 0.0)
                accum[f"coverage_{s}"] += float(cov[:, j].mean()) / len(seeds)
            for (lname, rname), (lcfg, rc) in configs.items():
                lift = T.routed_lift(lcfg, rc)
                top = T.rerank(lift, T.scale_factor(lcfg), args.top_k)
                m = T.metrics(top, args.sim_model)
                d = {k: v - base_metrics[k] for k, v in m.items()}
                kk = f"{lname}_{rname}_d_proxy_top1"
                accum.setdefault(kk, 0.0)
                accum[kk] += float(d["proxy_top1"].mean()) / len(seeds)
                per_seed_rows.append({
                    "fraction": fraction, "n_keep": n_keep, "seed": seed,
                    "lift": lname, "routing": rname,
                    "mean_delta_top1": float(d["proxy_top1"].mean()),
                    "mean_delta_best5": float(d["proxy_best5"].mean()),
                    "pct_top1_changed": float((top[:, 0] != base_top[:, 0]).mean()),
                    **{f"coverage_{s}": float(cov[:, j].mean()) for j, s in enumerate(SCOPES)},
                })
            log.info("f=%.2f n=%d seed=%d M4_intersection=%.4f cov_int=%.3f",
                     fraction, n_keep, seed,
                     accum.get("laplace_M4_intersection_d_proxy_top1", 0) * len(seeds),
                     cov[:, 3].mean())
        rows.append(accum)

    curve = pd.DataFrame(rows)
    per_seed = pd.DataFrame(per_seed_rows)
    summary = summarise_per_seed(per_seed)
    curve.to_csv(out_dir / "curve.csv", index=False)
    per_seed.to_csv(out_dir / "per_seed.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest = write_run_manifest(out_dir, script="experiments/13_feedback_volume_curve.py", args=args,
                                  inputs=[paths.dataset_parquet, pool_path, fb_db, split_path(42)],
                                   extra={"n_fractions": len(fractions), "n_seeds": len(seeds),
                                          "fractions": fractions, "seeds": seeds}, run_id=run_id)
    append_registry(run_id=run_id, phase="P3-volume", script="13_feedback_volume_curve.py",
                    out_dir=out_dir, manifest=manifest,
                     headline_metric="fractions", headline_value=len(fractions))

    log.info("=== VOLUME CURVE (dense_minilm, %s proxy top-1) ===", args.sim_model)
    for h in ("laplace", "laplace_eb"):
        print(f"\n--- {h} ---")
        for sc in ["M1_global", "M2_team", "M4_intersection", "M5_backoff2"]:
            pts = []
            for _, r in curve.iterrows():
                v = r.get(f"{h}_{sc}_d_proxy_top1")
                if v is not None:
                    pts.append(f"f={r['fraction']:.2f}:{v:+.4f}")
            print(f"  {sc:18s}: {' | '.join(pts[:6])}")
    print(f"\noutputs: {out_dir}")


if __name__ == "__main__":
    main()
