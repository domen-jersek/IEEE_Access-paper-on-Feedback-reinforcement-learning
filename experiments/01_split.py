#!/usr/bin/env python3
"""
01_split.py — Phase 0 Step 1
=============================
Produces stratified train/dev/eval splits (random + disjoint regimes)
with 5 random seeds.
Outputs:
  data/processed/splits/split_seed{42..1024}.json
  data/processed/splits/split_seed{42}_disjoint.json
  data/processed/splits/*_report.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from src.data.canonical import load_dataset
from src.data.split import stratified_triple_split, validate_splits


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("split")
    paths = ProjectPaths()

    log.info("=== Phase 0 Step 1: Loading dataset ===")
    df = load_dataset(paths.dataset_parquet)
    log.info("Loaded %d tickets from %s", len(df), paths.dataset_parquet)

    log.info("=== Phase 0 Step 1: Random stratified splits (5 seeds) ===")
    all_splits = stratified_triple_split(
        df,
        paths.splits_dir,
        seeds=[42, 123, 456, 789, 1024],
    )
    validate_splits(all_splits)
    for seed, split in all_splits.items():
        log.info("Seed %d: train=%d dev=%d eval=%d",
                 seed,
                 len(split["train"]),
                 len(split["dev"]),
                 len(split["eval"]))

    log.info("=== Phase 0 Step 1: Disjoint (group-aware) split ===")
    disjoint_splits = stratified_triple_split(
        df,
        paths.splits_dir,
        seeds=[42],
        group_col="combined_group_hash",
    )
    validate_splits(disjoint_splits)
    for seed, split in disjoint_splits.items():
        log.info("Disjoint seed %d: train=%d dev=%d eval=%d",
                 seed,
                 len(split["train"]),
                 len(split["dev"]),
                 len(split["eval"]))

    log.info("=== Phase 0 Step 1: DONE ===")
    log.info("Split files in %s:", paths.splits_dir)
    for p in sorted(paths.splits_dir.glob("*.json")):
        log.info("  %s", p.name)


if __name__ == "__main__":
    main()