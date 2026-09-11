#!/usr/bin/env python3
"""
02_build_index.py — Phase 1 Step 2
====================================
Builds FAISS index over all tickets and computes per-ticket baseline retrieval
difficulty (top-1 FAISS to the most similar other ticket).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from experiments.utils import setup_logging, load_dataset
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex


def main() -> None:
    setup_logging()
    log = logging.getLogger("build_index")
    paths = ProjectPaths()

    log.info("=== Loading dataset ===")
    df = load_dataset()
    log.info("Loaded %d tickets", len(df))

    log.info("=== Building SentenceTransformer encoder ===")
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    log.info("=== Encoding all tickets ===")
    embeddings = encoder.encode_batch(
        df["Title_anon"].tolist(),
        df["Description_anon"].fillna("").tolist(),
    )
    log.info("Embeddings: %s", embeddings.shape)

    index_dir = paths.data_processed / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Building FAISS index ===")
    faiss_idx = FAISSIndex(dim=encoder.dim)
    faiss_idx.build(embeddings, df)

    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "faiss_metadata.parquet"
    faiss_idx.save(index_path, meta_path)

    log.info("=== Computing baseline retrieval difficulty (top-1 FAISS) ===")
    top1_sims = FAISSIndex.compute_ticket_similarity(encoder, df, k=1)
    df["baseline_top1_similarity"] = top1_sims

    difficulty_path = paths.data_processed / "baseline_difficulty.csv"
    df[["seq_id", "baseline_top1_similarity"]].to_csv(difficulty_path, index=False)
    log.info("Difficulty stats: mean=%.3f median=%.3f", top1_sims.mean(), top1_sims.median())

    log.info("=== DONE ===")
    log.info("Index: %s (%d vectors)", index_path, faiss_idx.size)
    log.info("Metadata: %s", meta_path)
    log.info("Difficulty: %s", difficulty_path)


if __name__ == "__main__":
    main()