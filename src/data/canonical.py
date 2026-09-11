"""
Canonicalize the raw CSV into a clean, versioned Parquet dataset.
Handles: deduplication, Ref collisions, text normalization, manifest generation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import ProjectPaths

log = logging.getLogger(__name__)


def _hash_file(path: Path, blocksize: int = 65536) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            sha.update(block)
    return sha.hexdigest()


def canonicalize(csv_path: Path, output_path: Path, manifest_path: Path) -> pd.DataFrame:
    log.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    source_hash = _hash_file(csv_path)
    raw_rows = len(df)
    log.info("Raw rows: %d", raw_rows)

    df = df[df["Ref"].notna()].copy()
    log.info("After dropping missing Ref: %d rows", len(df))

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df["Title_anon"] = df["Title_anon"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["Description_anon"] = df["Description_anon"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["first_reply"] = df["first_reply"].fillna("").str.replace(r"\r\n", "\n", regex=True).str.replace(r"\n{3,}", "\n\n", regex=True).str.strip()

    ref_counts = df["Ref"].value_counts()
    dup_refs = ref_counts[ref_counts > 1]
    log.info("Duplicate Refs (%d groups): %s", len(dup_refs), list(dup_refs.index))

    keep_mask = pd.Series(True, index=df.index)
    for ref in dup_refs.index:
        rows = df[df["Ref"] == ref]
        canonical = rows.iloc[0]
        is_exact_dupe = (
            (rows["Title_anon"] == canonical["Title_anon"]) &
            (rows["Team->Name"] == canonical["Team->Name"]) &
            (rows["Service subcategory->Name"] == canonical["Service subcategory->Name"]) &
            (rows["first_reply"] == canonical["first_reply"])
        )
        if is_exact_dupe.all():
            idxs_to_drop = rows.index[1:]
            keep_mask.loc[idxs_to_drop] = False

    df = df[keep_mask].copy()
    log.info("After dedup: %d rows", len(df))

    seen_ref = {}
    for idx in df.index:
        ref = df.at[idx, "Ref"]
        count = seen_ref.get(ref, 0) + 1
        seen_ref[ref] = count
        if count > 1:
            df.at[idx, "Ref"] = f"{ref}_b"

    df.reset_index(drop=True, inplace=True)
    df["seq_id"] = [f"R-{i + 1}" for i in range(len(df))]

    df["reply_len"] = df["first_reply"].str.len()
    df["desc_len"] = df["Description_anon"].str.len()
    df["title_len"] = df["Title_anon"].str.len()

    df["has_reply"] = df["first_reply"] != ""
    df["has_desc"] = df["Description_anon"] != ""

    log.info("Final rows: %d, unique seq_id: %d", len(df), df["seq_id"].nunique())

    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, output_path, compression="zstd")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv_hash": source_hash,
        "source_rows": raw_rows,
        "output_rows": len(df),
        "unique_refs_before_dedup": raw_rows,
        "unique_refs_after_dedup": df["Ref"].nunique(),
        "unique_seq_ids": int(df["seq_id"].nunique()),
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_rows": {
            "first_reply": int(df["first_reply"].isna().sum() + (df["first_reply"] == "").sum()),
            "title": int(df["Title_anon"].isna().sum() + (df["Title_anon"] == "").sum()),
            "description": int(df["Description_anon"].isna().sum() + (df["Description_anon"] == "").sum()),
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Manifest written to %s", manifest_path)

    return df


def load_dataset(parquet_path: Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    paths = ProjectPaths()
    df = canonicalize(paths.source_csv, paths.dataset_parquet, paths.manifest_json)
    print(df[["seq_id", "Ref", "Title_anon", "Team->Name", "reply_len"]].head(10))
    print("\nTeam distribution preview:")
    print(df["Team->Name"].value_counts().head(15))