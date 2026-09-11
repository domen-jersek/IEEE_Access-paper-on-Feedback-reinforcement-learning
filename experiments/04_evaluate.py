#!/usr/bin/env python3
"""
04_evaluate.py — Phase 3
==========================
Runs the LOO evaluation pipeline on dev queries (or train queries for gate training).
Supports M1-M4 methods + baseline.

Usage:
    python experiments/04_evaluate.py --method M1_global --split dev --seed 42
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths, DEFAULT_METHODS, EvalConfig
from experiments.utils import setup_logging, load_dataset, load_split, get_arg_parser
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.feedback.loader import load_feedback_as_scores
from src.evaluation.runner import EvaluationRunner


async def main_async() -> None:
    parser = get_arg_parser("Evaluation runner")
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline", "M1_global", "M2_team", "M3_class", "M4_intersection"])
    parser.add_argument("--split", type=str, default="dev",
                        choices=["train", "dev", "eval"])
    parser.add_argument("--feedback-protocol", type=str, default="conditioned",
                        choices=["conditioned", "blind"])
    parser.add_argument("--agg-mode", type=str, default="continuous",
                        choices=["continuous", "binary"],
                        help="How to aggregate raw judge scores into pos/neg lifts")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    log = logging.getLogger("evaluate")

    paths = ProjectPaths()
    log.info("=== Loading dataset + split ===")
    df = load_dataset()
    split = load_split(seed=args.seed, regime=args.regime)
    query_ids = split[args.split]
    log.info("Evaluating %s: %d queries (agg: %s)", args.split, len(query_ids), args.agg_mode)

    log.info("=== Loading FAISS index ===")
    index_dir = paths.data_processed / "faiss_index"
    faiss_idx = FAISSIndex(dim=384)
    faiss_idx.load(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")

    log.info("=== Loading encoder ===")
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    log.info("=== Loading feedback scores ===")
    fb_db_path = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    if not fb_db_path.exists():
        log.warning("Feedback DB not found at %s, proceeding with zero feedback (baseline-only)", fb_db_path)
        feedback_scores = {}
    else:
        feedback_scores = load_feedback_as_scores(
            fb_db_path, exclude_query_ids=set(query_ids), mode=args.agg_mode
        )
        log.info("Loaded feedback scores for %d candidates", len(feedback_scores))

    log.info("=== Configuring evaluation ===")
    base_cfg = DEFAULT_METHODS.get(args.method)
    if base_cfg is None:
        log.error("Unknown method: %s", args.method)
        return
    config = EvalConfig(
        experiment_id=f"{args.method}_{args.split}_{args.feedback_protocol}_{args.agg_mode}",
        lift=base_cfg.lift,
        routing=base_cfg.routing,
        gating=base_cfg.gating,
        generator_model=base_cfg.generator_model,
        judge_model=base_cfg.judge_model,
        regime=args.regime,
        seed=args.seed,
    )

    results_dir = paths.results / config.experiment_id
    runner = EvaluationRunner(
        config=config,
        faiss_index=faiss_idx,
        dataset=df,
        encoder=encoder,
        feedback_scores=feedback_scores,
        results_dir=results_dir,
        concurrency=4,
        interim_every=10,
    )

    log.info("=== Running evaluation ===")
    summary = await runner.run(query_ids)
    log.info("Summary: mean_delta=%.4f, pct_improved=%.1f%%",
             summary.get("mean_delta_cosine", 0), summary.get("pct_improved", 0) * 100)
    log.info("=== DONE ===")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()