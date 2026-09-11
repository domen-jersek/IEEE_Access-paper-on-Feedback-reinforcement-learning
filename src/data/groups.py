"""
Detect near-duplicate procedure groups for condition-disjoint split regime.
Groups by: reply_hash (normalized first 300 chars of first_reply) and title_subcat_hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _hash_text(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def build_groups(df: pd.DataFrame, output_json: Path) -> pd.DataFrame:
    log.info("Building procedure groups for condition-disjoint split")

    reply_sig = df["first_reply"].fillna("").str[:300].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    df["reply_hash"] = reply_sig.apply(lambda x: _hash_text(x))

    title_sig = df["Title_anon"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    subcat_sig = df["Service subcategory->Name"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    df["title_subcat_hash"] = (title_sig + " | " + subcat_sig).apply(lambda x: _hash_text(x))

    df["combined_group_hash"] = (reply_sig + " | " + title_sig + " | " + subcat_sig).apply(lambda x: _hash_text(x))

    reply_groups = df.groupby("reply_hash").size()
    title_groups = df.groupby("title_subcat_hash").size()
    combined_groups = df.groupby("combined_group_hash").size()

    stats = {
        "total_tickets": int(len(df)),
        "reply_hash_unique": int(len(reply_groups)),
        "reply_hash_multi": int((reply_groups > 1).sum()),
        "reply_hash_multi_tickets": int(reply_groups[reply_groups > 1].sum()),
        "reply_hash_largest": int(reply_groups.max()),
        "title_subcat_hash_unique": int(len(title_groups)),
        "title_subcat_hash_multi": int((title_groups > 1).sum()),
        "title_subcat_hash_multi_tickets": int(title_groups[title_groups > 1].sum()),
        "title_subcat_hash_largest": int(title_groups.max()),
        "combined_hash_unique": int(len(combined_groups)),
        "combined_hash_multi": int((combined_groups > 1).sum()),
        "combined_hash_multi_tickets": int(combined_groups[combined_groups > 1].sum()),
    }

    output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Group stats written to %s: %d multi-ticket combined groups (%d tickets affected)",
             output_json, stats["combined_hash_multi"], stats["combined_hash_multi_tickets"])

    return df