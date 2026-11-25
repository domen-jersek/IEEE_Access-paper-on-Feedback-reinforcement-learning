#!/usr/bin/env python3
"""34_lift_diagnostics.py — vote distribution, saturation and scope priors.

Reproduces the numbers needed for the revised equations and the saturation
argument:

* empirical judge-score distribution on the allowed history-only pairs;
* Laplace lift symmetry for (pos, neg) pairs;
* share of candidate lifts at the positive/negative cap for n votes, obtained by
  resampling the empirical score distribution;
* ratio of lift standard deviation to retrieval-score standard deviation on the
  frozen strict and random panels, for the legacy and calibrated lift;
* per-scope base rates π_s on the allowed evidence.

No API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, make_run_id, setup_logging
from src.config import LiftConfig, ProjectPaths, RoutingConfig
from src.data.artifacts import write_once_json
from src.evaluation.ladder import build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.evaluation.revision_panels import (
    collect_pools, judge_bundle, load_feedback_rows, panel_specs, resolve_panel,
    saturation, ticket_embeddings,
)
from src.feedback.loader import compute_scope_priors
from src.retrieval.history import build_history_retriever


def _laplace(pos: float, neg: float, m: float = 0.8, cap: float = 0.2) -> float:
    n = pos + neg
    p = (pos + 1.0) / (n + 2.0)
    return float(np.clip((p - 0.5) * min(1.0, n / 2.0) * m, -cap, cap))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="disjoint_prefix")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)
    specs = panel_specs(paths, 42)
    context = resolve_panel(specs[args.panel], paths, partition="eval")
    train = set(context.history_ids)

    distribution_rows, prior_rows, symmetry_rows = [], [], []
    vote_rows, pool_rows = [], []
    embeddings = ticket_embeddings(dataset, paths)
    retriever = build_history_retriever("dense_minilm", dataset, train)
    pools = collect_pools(context, dataset, embeddings, retriever, 100)
    queries = dataset.set_index("seq_id").loc[list(context.query_ids)].reset_index()
    similarity = ReplySimilarity(dataset, "minilm")

    rng = np.random.default_rng(42)
    for protocol in ("conditioned", "blind"):
        rows = load_feedback_rows(paths.data_processed / f"feedback_{protocol}.db")
        allowed, audit = context.select_feedback(rows, protocol=protocol)
        scores = allowed["score"].to_numpy(float)
        distribution_rows.append({
            "protocol": protocol, "rows": len(scores), "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p10": float(np.percentile(scores, 10)), "p90": float(np.percentile(scores, 90)),
            "pct_negative": float((scores <= 0.4).mean()),
            "pct_positive": float((scores >= 0.8).mean()),
            "pct_neutral": float(((scores > 0.4) & (scores < 0.8)).mean()),
            "pct_exact_half": float((scores == 0.5).mean()),
            "covered_queries": int(allowed.query_id.nunique()),
            "covered_candidates": int(allowed.candidate_id.nunique()),
        })

        priors = compute_scope_priors(allowed)
        for scope, values in priors.items():
            prior_rows.append({"protocol": protocol, "scope": scope,
                               "mean": values["mean"], "n": values["n"]})

        # Sample n votes with replacement from the empirical distribution, compute
        # the Laplace lift, and record the cap shares.
        for n in (1, 2, 5, 10):
            draws = rng.choice(scores, size=(args.bootstrap, n), replace=True)
            pos = draws.sum(axis=1)
            neg = n - pos
            p = (pos + 1) / (n + 2)
            lift = np.clip((p - 0.5) * min(1.0, n / 2.0) * 0.8, -0.2, 0.2)
            saturation_rows = {
                "protocol": protocol, "n_votes": n,
                "mean_lift": float(lift.mean()),
                "pct_at_positive_cap": float((lift >= 0.19999).mean()),
                "pct_at_negative_cap": float((lift <= -0.19999).mean()),
                "pct_zero": float((lift == 0).mean()),
                "pct_positive": float((lift > 0).mean()),
                "pct_negative": float((lift < 0).mean()),
            }
            vote_rows.append(saturation_rows)

        bundle = judge_bundle(context, rows, protocol)[0]
        tensor = build_pool_tensor(pools, queries, bundle, {"minilm": similarity})
        for label, lift in (("laplace_legacy", LiftConfig.laplace()),
                            ("laplace_eb_k2", LiftConfig.laplace_eb(prior_strength=2.0)),
                            ("laplace_eb_k2_poolstd",
                             LiftConfig.laplace_eb(prior_strength=2.0,
                                                   scale_mode="pool_std", pool_lambda=1.0))):
            stats = saturation(tensor, lift, RoutingConfig.intersection())
            pool_rows.append({"protocol": protocol, "config": label, **stats})

    for pos, neg in ((0, 1), (1, 0), (0, 2), (2, 0), (0, 5), (5, 0), (1, 1), (2, 2)):
        symmetry_rows.append({"pos": pos, "neg": neg,
                              "laplace": _laplace(pos, neg),
                              "empirical_note": "symmetric by construction"})

    run_id = make_run_id(f"lift_diagnostics_{args.panel}")
    out_dir = paths.results / "revision" / "lift_diagnostics" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distribution_rows).to_csv(out_dir / "score_distribution.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / "scope_priors.csv", index=False)
    pd.DataFrame(symmetry_rows).to_csv(out_dir / "lift_symmetry.csv", index=False)
    pd.DataFrame(vote_rows).to_csv(out_dir / "vote_saturation.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(out_dir / "pool_saturation.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "panel": args.panel,
        "dataset": {"sha256": file_sha256(paths.dataset_parquet)},
        "feedback": {"conditioned": file_sha256(paths.data_processed / "feedback_conditioned.db"),
                     "blind": file_sha256(paths.data_processed / "feedback_blind.db")},
        "bootstrap": args.bootstrap,
    })
    print(pd.DataFrame(distribution_rows).round(4).to_string(index=False))
    print()
    print(pd.DataFrame(vote_rows).round(4).to_string(index=False))
    print()
    print(pd.DataFrame(pool_rows).round(4).to_string(index=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
