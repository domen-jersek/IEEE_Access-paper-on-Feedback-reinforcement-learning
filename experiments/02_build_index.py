#!/usr/bin/env python3
"""
02_build_index.py — Phase 1 Step 2
====================================
Builds FAISS index over all tickets, computes per-ticket baseline retrieval
difficulty, and determines the optimal feedback judging depth.

The judging depth is the number of FAISS candidates that must be rated by the
LLM judge per query so that any candidate the lift formula could theoretically
promote into the RAG top-5 has a feedback score.

Rationale: the lift cap is +/-0.20.  A candidate at FAISS rank R can enter the
generator's top-5 context iff  FAISS[R] + 0.20 >= FAISS[4]  (rank 4 = 5th-best
by raw FAISS).  We search top-200 per query, find the last rank where this
holds, and recommend the 95th percentile across all queries.

Output:  data/processed/recommended_judging_depth.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from experiments.utils import setup_logging, load_dataset, load_split
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex, compute_recommended_judging_depth


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

    log.info("=== Computing optimal feedback judging depth ===")
    random_split = load_split(seed=42, regime="random")
    train_ids = random_split["train"]
    log.info("Sampling %d train queries for depth computation", len(train_ids))

    depth_report = compute_recommended_judging_depth(
        faiss_index=faiss_idx,
        encoder=encoder,
        query_ids=train_ids,
        lift_cap=0.20,
        sample_depth=200,
    )

    depth_path = paths.data_processed / "recommended_judging_depth.json"
    depth_path.write_text(json.dumps(depth_report, indent=2), encoding="utf-8")

    log.info("Max promotable rank distribution (across %d queries):", depth_report["n_queries_sampled"])
    log.info("  min=%d  p50=%d  p75=%d  p90=%d  p95=%d  p99=%d  max=%d",
             depth_report["min_rank"],
             depth_report["p50"],
             depth_report["p75"],
             depth_report["p90"],
             depth_report["p95"],
             depth_report["p99"],
             depth_report["max_rank"])
    log.info("  Recommended judging depth (95th pctl): %d",
             depth_report["recommended_judging_depth"])

    log.info("=== DONE ===")
    log.info("Index: %s (%d vectors)", index_path, faiss_idx.size)
    log.info("Metadata: %s", meta_path)
    log.info("Difficulty: %s", difficulty_path)
    log.info("Judging depth: %s", depth_path)


def build_alt_index(tag: str) -> None:
    """P2: build an additional FAISS index with an alternative embedder (e.g. --embedder bge).
    Writes data/processed/faiss_index_<tag>/ and never touches the default MiniLM index."""
    from src.config import ALT_EMBEDDERS
    from experiments.utils import write_run_manifest, append_registry, make_run_id
    log = logging.getLogger("build_index")
    paths = ProjectPaths()
    model_name = ALT_EMBEDDERS[tag]
    df = load_dataset()
    # Row order must match the default index metadata (shared LOO exclusion by position).
    default_meta = paths.data_processed / "faiss_index" / "faiss_metadata.parquet"
    if default_meta.exists():
        import pandas as pd
        ref_ids = pd.read_parquet(default_meta)["seq_id"].tolist()
        if ref_ids != df["seq_id"].tolist():
            raise RuntimeError("dataset.parquet row order differs from faiss_metadata.parquet; refusing to build misaligned index")
    log.info("=== Encoding all tickets with %s ===", model_name)
    encoder = TicketEncoder(model_name)
    embeddings = encoder.encode_batch(df["Title_anon"].tolist(), df["Description_anon"].fillna("").tolist())
    index_dir = paths.data_processed / f"faiss_index_{tag}"
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss_idx = FAISSIndex(dim=encoder.dim)
    faiss_idx.build(embeddings, df)
    faiss_idx.save(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")
    run_id = make_run_id(f"build_index_{tag}")
    manifest = write_run_manifest(index_dir, script="experiments/02_build_index.py", args={"embedder": tag, "model": model_name},
                                  inputs=[paths.dataset_parquet], extra={"dim": encoder.dim, "n": len(df)}, run_id=run_id)
    append_registry(run_id=run_id, phase="P2-index", script="02_build_index.py", out_dir=index_dir, manifest=manifest,
                    headline_metric="n_vectors", headline_value=faiss_idx.size)
    log.info("=== DONE === %s (%d vectors, dim=%d)", index_dir, faiss_idx.size, encoder.dim)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build FAISS index (default MiniLM pipeline, or --embedder <tag> for an extra index)")
    ap.add_argument("--embedder", type=str, default=None, help="alternative embedder tag from src.config.ALT_EMBEDDERS (e.g. bge)")
    a = ap.parse_args()
    if a.embedder and a.embedder != "minilm":
        setup_logging()
        build_alt_index(a.embedder)
    else:
        main()