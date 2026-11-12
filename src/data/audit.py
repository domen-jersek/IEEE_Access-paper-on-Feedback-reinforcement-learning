"""Aggregate overlap audits between split partitions (no ticket text emitted)."""
from __future__ import annotations

import pandas as pd

from .groups_v2 import normalize_signature

PARTITION_PAIRS = (("train", "dev"), ("train", "eval"), ("dev", "eval"))


def overlap_rows(mapping: dict[str, str], split: dict, *, label: str, family: str) -> list[dict]:
    rows = []
    for first, second in PARTITION_PAIRS:
        ids_first = set(split[first])
        ids_second = set(split[second])
        first_values = {mapping[i] for i in ids_first if i in mapping}
        second_values = {mapping[i] for i in ids_second if i in mapping}
        shared = first_values & second_values
        affected = sum(1 for i in ids_second if mapping.get(i) in shared)
        rows.append({
            "split": label,
            "family": family,
            "pair": f"{first}-{second}",
            "shared_values": len(shared),
            "tickets_in_pairs": affected,
            "second_partition_tickets": len(ids_second),
            "fraction_second_affected": affected / max(len(ids_second), 1),
        })
    return rows


def signature_mappings(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Independent signature families (not the union rule)."""
    title = df["Title_anon"].map(normalize_signature)
    subcategory = df["Service subcategory->Name"].map(normalize_signature)
    title_subcategory = (title + " | " + subcategory).where(~((title == "") & (subcategory == "")), "")
    families = {
        "full_reply": df["first_reply"].map(normalize_signature),
        "reply_prefix300": df["first_reply"].fillna("").astype(str).str[:300].map(normalize_signature),
        "title_subcategory": title_subcategory,
    }
    return {
        name: dict(zip(df["seq_id"].astype(str), values))
        for name, values in families.items()
    }


def split_overlap_table(df: pd.DataFrame, splits: dict[str, dict],
                        group_mappings: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    """Long table: for every split, every signature family and union group rule."""
    frames = []
    families = signature_mappings(df)
    for label, split in splits.items():
        for family, mapping in families.items():
            frames.extend(overlap_rows(mapping, split, label=label, family=family))
        for rule_id, mapping in (group_mappings or {}).items():
            frames.extend(overlap_rows(mapping, split, label=label, family=f"group:{rule_id}"))
    return pd.DataFrame(frames)
