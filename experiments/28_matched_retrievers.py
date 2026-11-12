#!/usr/bin/env python3
"""28_matched_retrievers.py — retriever comparison with one feedback configuration.

R1-m4 / R2-10: retriever strength must not be confounded with different lift
settings. Here every retriever uses the same evidence, the same empirical-Bayes
lift and the same routing, in two scaling modes:

* ``absolute`` — the published finalist; meaningful for cosine retrievers.
* ``pool_std`` — lift expressed in the retriever pool's own score units, which is
  the scale-matched comparison across retrieval families.

Retrieval-level only; no generation, no API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, make_run_id, setup_logging
from src.config import LiftConfig, ProjectPaths, RoutingConfig
from src.data.artifacts import write_once_json
from src.data.groups_v2 import normalize_signature
from src.evaluation.ladder import build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.evaluation.revision_panels import (
    bootstrap,
    collect_pools,
    judge_bundle,
    load_feedback_rows,
    p_value,
    panel_specs,
    resolve_panel,
    ticket_embeddings,
)
from src.retrieval.history import build_history_retriever

PANELS = ("legacy_random", "disjoint_prefix")
RETRIEVERS = ("dense_minilm", "bm25", "hybrid_rrf")
SCALES = (("absolute", LiftConfig.laplace_eb(prior_strength=2.0)),
          ("pool_std", LiftConfig.laplace_eb(prior_strength=2.0,
                                             scale_mode="pool_std", pool_lambda=1.0)))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", default=list(PANELS))
    parser.add_argument("--retrievers", nargs="+", default=list(RETRIEVERS))
    parser.add_argument("--source", default="conditioned", choices=["conditioned", "blind"])
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)
    embeddings = ticket_embeddings(dataset, paths)
    specs = panel_specs(paths, 42)
    raw = load_feedback_rows(paths.data_processed / f"feedback_{args.source}.db")

    rows = []
    for panel_name in args.panels:
        context = resolve_panel(specs[panel_name], paths, partition="eval")
        queries = dataset.set_index("seq_id").loc[list(context.query_ids)].reset_index()
        group_map = dict(context.group_ids)
        reply_full = dict(zip(dataset["seq_id"], dataset["first_reply"].map(normalize_signature)))
        similarity = ReplySimilarity(dataset, "minilm", reply_hash_override=reply_full,
                                     group_hash_override=group_map)
        bundle = judge_bundle(context, raw, args.source)[0]
        for retriever_name in args.retrievers:
            retriever = build_history_retriever(retriever_name, dataset,
                                                set(context.history_ids))
            pools = collect_pools(context, dataset, embeddings, retriever, args.search_k)
            tensor = build_pool_tensor(pools, queries, bundle, {"minilm": similarity})
            base_top = tensor.baseline_top(5)
            base_metrics = tensor.metrics(base_top, "minilm")
            for scale_name, lift in SCALES:
                if scale_name == "absolute" and retriever_name != "dense_minilm":
                    # Absolute lift units are not comparable outside cosine scores;
                    # still evaluated for completeness and labelled as such.
                    pass
                values = tensor.routed_lift(lift, RoutingConfig.intersection())
                top = tensor.rerank(values, tensor.scale_factor(lift), 5)
                metrics = tensor.metrics(top, "minilm")
                delta = metrics["proxy_top1"] - base_metrics["proxy_top1"]
                stats = bootstrap(delta)
                rows.append({
                    "panel": panel_name, "source": args.source, "retriever": retriever_name,
                    "scale_mode": scale_name, "n_queries": len(delta),
                    "baseline_top1": float(base_metrics["proxy_top1"].mean()),
                    "delta_top1": stats["mean"],
                    "ci_lower": stats["mean_ci_lower"], "ci_upper": stats["mean_ci_upper"],
                    "wilcoxon_p": p_value(delta),
                    "delta_mean5": float((metrics["proxy_mean5"]
                                          - base_metrics["proxy_mean5"]).mean()),
                    "pct_top1_changed": float((top[:, 0] != base_top[:, 0]).mean()),
                })
            print(f"[{panel_name}/{retriever_name}] queries={len(context.query_ids)} "
                  f"pool_median={pools.groupby('query_id').size().median():.0f}")

    run_id = make_run_id(f"matched_retrievers_{args.source}")
    out_dir = paths.results / "revision" / "matched_retrievers" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "matched_retrievers.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "source": args.source,
        "lift": {"name": "laplace_eb", "prior_strength": 2.0, "routing": "intersection"},
        "dataset": {"sha256": file_sha256(paths.dataset_parquet)},
        "git": git_info(),
    })
    print(table[["panel", "retriever", "scale_mode", "baseline_top1", "delta_top1",
                 "ci_lower", "ci_upper", "wilcoxon_p"]].round(4).to_string(index=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
