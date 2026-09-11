#!/usr/bin/env python3
"""
03_build_feedback.py — Phase 2
================================
Builds feedback databases (conditioned + blind) from train queries only.
Uses FAISS top-100 retrieval + LLM judge.

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
from src.feedback.builder import FeedbackBuilder
from src.feedback.loader import FeedbackDB


async def build_protocol(
    protocol: str,
    judge_model: str,
    db_path: Path,
    train_ids: list[str],
    faiss_idx,
    dataset,
    encoder,
) -> int:
    log = logging.getLogger(f"feedback.{protocol}")
    cache_path = Path("data/processed") / f"judge_cache_{protocol}.db"

    judge = FeedbackJudge(
        model=judge_model,
        protocol=protocol,
        prompt_version="v1",
        cache_path=cache_path,
        max_concurrency=8,
    )
    db = FeedbackDB(db_path)
    builder = FeedbackBuilder(faiss_idx, dataset, judge, db, top_k=100, batch_size=10)

    log.info("Building %s feedback from %d train queries...", protocol, len(train_ids))
    n = await builder.build(train_ids, encoder)
    log.info("Built %s feedback: %d rows, %d queries, %d candidates",
             protocol, db.count(), db.distinct_queries(), db.distinct_candidates())
    return n


async def main_async() -> None:
    setup_logging()
    log = logging.getLogger("build_feedback")
    paths = ProjectPaths()

    log.info("=== Loading dataset + split ===")
    df = load_dataset()
    split = load_split(seed=42, regime="random")
    train_ids = split["train"]
    log.info("Train queries: %d", len(train_ids))

    log.info("=== Loading FAISS index ===")
    index_dir = paths.data_processed / "faiss_index"
    faiss_idx = FAISSIndex(dim=384)
    faiss_idx.load(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")

    log.info("=== Loading encoder ===")
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    JUDGE_MODEL = "openai/gpt-4o-mini-2024-07-18"

    for protocol in ["conditioned", "blind"]:
        db_path = paths.data_processed / f"feedback_{protocol}.db"
        n = await build_protocol(protocol, JUDGE_MODEL, db_path, train_ids, faiss_idx, df, encoder)
        log.info("Feedback DB %s: %d rows written to %s", protocol, n, db_path)

    log.info("=== DONE ===")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()