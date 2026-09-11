#!/usr/bin/env python3
"""
00_canonicalize.py — Phase 0 Step 0
====================================
Reads the raw tickets CSV, canonicalizes it, builds the taxonomy,
detects near-duplicate procedure groups, and writes frozen artifacts.

Outputs:
  data/processed/dataset.parquet
  data/processed/dataset_manifest.json
  data/processed/taxonomy.csv
  data/processed/procedure_groups.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from src.data.canonical import canonicalize
from src.data.taxonomy import build_taxonomy, get_intent_class_names
from src.data.groups import build_groups


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    paths = ProjectPaths()
    paths.data_processed.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("canonicalize")
    log.info("=== Phase 0 Step 0: Canonicalization ===")
    df = canonicalize(paths.source_csv, paths.dataset_parquet, paths.manifest_json)
    log.info("Dataset: %d tickets written", len(df))

    log.info("=== Phase 0 Step 0: Taxonomy ===")
    df = build_taxonomy(df, paths.taxonomy_csv)
    class_names = get_intent_class_names()
    log.info("Intent classes (%d): %s", len(class_names), class_names)

    log.info("=== Phase 0 Step 0: Procedure Groups ===")
    df = build_groups(df, paths.groups_json)

    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, paths.dataset_parquet, compression="zstd")
    log.info("Dataset re-saved with taxonomy + groups: %d columns", len(df.columns))

    log.info("=== Phase 0 Step 0: DONE ===")
    log.info("Artifacts written:")
    for p in [paths.dataset_parquet, paths.manifest_json, paths.taxonomy_csv, paths.groups_json]:
        log.info("  %s (%d bytes)", p, p.stat().st_size if p.exists() else 0)


if __name__ == "__main__":
    main()