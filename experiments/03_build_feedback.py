#!/usr/bin/env python3
"""
03_build_feedback.py — Phase 2
================================
Builds feedback databases (conditioned + blind) from train queries only.
Uses FAISS top-K retrieval + LLM judge with continuous score aggregation.

Two modes:
  --validate   Run a diagnostic on 20 random queries first, produce a report,
               then prompt before continuing to the full build.
  (default)    Full build on all 878 train queries (skips validation).

Requires: OPENROUTER_API_KEY env var.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from experiments.utils import setup_logging, load_dataset, load_split
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.feedback.judge import FeedbackJudge
from src.feedback.builder import FeedbackBuilder, validate_feedback_pipeline, save_validation_report
from src.feedback.loader import FeedbackDB

TOP_K = 75
JUDGE_MODEL = "openai/gpt-luna-latest"


async def build_protocol(
    protocol: str,
    db_path: Path,
    train_ids: list[str],
    faiss_idx,
    dataset,
    encoder,
) -> int:
    log = logging.getLogger(f"feedback.{protocol}")
    cache_path = Path("data/processed") / f"judge_cache_{protocol}.db"

    judge = FeedbackJudge(
        model=JUDGE_MODEL,
        protocol=protocol,
        prompt_version="v1",
        cache_path=cache_path,
        max_concurrency=8,
    )
    db = FeedbackDB(db_path)
    builder = FeedbackBuilder(faiss_idx, dataset, judge, db, top_k=TOP_K, batch_size=10)

    log.info("Building %s feedback from %d train queries (top-%d per query)...",
             protocol, len(train_ids), TOP_K)
    n = await builder.build(train_ids, encoder)
    log.info("Done: %d rows, %d distinct queries, %d distinct candidates",
             db.count(), db.distinct_queries(), db.distinct_candidates())
    return n


async def main_async() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build feedback databases")
    parser.add_argument("--validate", action="store_true",
                        help="Run a diagnostic on 20 queries first, then ask before full build")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run only the diagnostic, skip the full build")
    parser.add_argument("--n-validate", type=int, default=20,
                        help="Number of queries for the validation run (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regime", type=str, default="random")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("build_feedback")
    paths = ProjectPaths()

    log.info("=== Loading dataset + split ===")
    df = load_dataset()
    split = load_split(seed=args.seed, regime=args.regime)
    train_ids = split["train"]
    log.info("Train queries: %d", len(train_ids))

    log.info("=== Loading FAISS index ===")
    index_dir = paths.data_processed / "faiss_index"
    faiss_idx = FAISSIndex(dim=384)
    faiss_idx.load(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")

    log.info("=== Loading encoder ===")
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    log.info("=== Judge model: %s ===", JUDGE_MODEL)
    log.info("=== Retrieval depth: top-%d ===", TOP_K)

    if args.validate or args.validate_only:
        log.info("=" * 60)
        log.info("VALIDATION RUN: %d queries × 2 protocols", args.n_validate)
        log.info("=" * 60)

        report = await validate_feedback_pipeline(
            faiss_index=faiss_idx,
            dataset=df,
            encoder=encoder,
            judge_model=JUDGE_MODEL,
            top_k=TOP_K,
            n_queries=args.n_validate,
        )
        report_path = paths.data_processed / "feedback_validation_report.json"
        save_validation_report(report, report_path)

        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        for proto in ["conditioned", "blind"]:
            p = report["protocols"].get(proto, {})
            sd = p.get("score_distribution", {})
            bb = p.get("binary_breakdown", {})
            print(f"\n  [{proto}]")
            print(f"    Rows: {p.get('n_rows', '?')}")
            print(f"    Mean score: {sd.get('mean', 0):.3f}  (median: {sd.get('median', 0):.3f})")
            print(f"    Positive (>=0.80): {100 * bb.get('pct_positive', 0):.1f}%")
            print(f"    Negative (<=0.40): {100 * bb.get('pct_negative', 0):.1f}%")
            print(f"    Neutral (0.40..0.80): {100 * bb.get('pct_neutral', 0):.1f}%")

        cv = report.get("conditioned_vs_blind", {})
        if cv:
            print(f"\n  Conditioned-vs-blind Pearson r: {cv.get('pearson_r', 0):.3f}")
            print(f"  (across {cv.get('n_common_pairs', 0)} common pairs)")

        print("\n" + "=" * 60)

        if args.validate_only:
            log.info("--validate-only: stopping after diagnostic.")
            return

        resp = input("\nProceed with full build on all %d train queries? [y/N] " % len(train_ids))
        if resp.strip().lower() != "y":
            log.info("Aborted.")
            return

    log.info("=" * 60)
    log.info("FULL FEEDBACK BUILD")
    log.info("=" * 60)

    for protocol in ["conditioned", "blind"]:
        db_path = paths.data_processed / f"feedback_{protocol}.db"
        n = await build_protocol(protocol, db_path, train_ids, faiss_idx, df, encoder)
        log.info("Written: %s (%d rows)", db_path, n)

    log.info("=== DONE ===")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()