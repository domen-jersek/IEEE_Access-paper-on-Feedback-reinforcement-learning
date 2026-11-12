"""Versioned structural duplicate/template groups for the revision.

Two rules, both connected components over ticket signatures:

``duplicate_template_v1``
    equal normalized full reply OR equal normalized title+subcategory.
``prefix_template_v1``
    equal normalized first 300 reply characters OR equal normalized
    title+subcategory. Conservative: includes the legacy prefix signature.

Signatures that are empty never link tickets. Group IDs are deterministic from
the sorted member IDs. These are *structural* groups: they guarantee that the
named signatures do not cross partitions. They do not prove that two tickets
describe the same procedure, and the artifact therefore records
``semantic_status = "structural_proxy"``. Domain validation is a separate step.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

GROUP_SCHEMA_VERSION = 1
PREFIX_CHARS = 300


def normalize_signature(value) -> str:
    text = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _reply_full(df: pd.DataFrame) -> pd.Series:
    return df["first_reply"].map(normalize_signature)


def _reply_prefix(df: pd.DataFrame) -> pd.Series:
    # Normalize first, then cut, so the prefix rule is a strict superset of the
    # full-reply rule (identical normalized replies always share a prefix).
    return df["first_reply"].map(normalize_signature).str[:PREFIX_CHARS]


def _reply_prefix_legacy(df: pd.DataFrame) -> pd.Series:
    """Exactly the submitted ``reply_hash`` definition: cut first, then normalize."""
    return df["first_reply"].fillna("").astype(str).str[:PREFIX_CHARS].map(normalize_signature)


def _title_subcategory(df: pd.DataFrame) -> pd.Series:
    title = df["Title_anon"].map(normalize_signature)
    subcategory = df["Service subcategory->Name"].map(normalize_signature)
    signature = title + " | " + subcategory
    # An empty title AND subcategory must not collapse unrelated tickets.
    return signature.where(~((title == "") & (subcategory == "")), "")


@dataclass(frozen=True)
class GroupRule:
    rule_id: str
    description: str
    signature_builders: tuple[Callable[[pd.DataFrame], pd.Series], ...]
    semantic_status: str = "structural_proxy"

    def signatures(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        return {builder.__name__.lstrip("_"): builder(df) for builder in self.signature_builders}


GROUP_RULES: Mapping[str, GroupRule] = {
    "duplicate_template_v1": GroupRule(
        rule_id="duplicate_template_v1",
        description="Equal normalized full reply OR equal normalized title+subcategory.",
        signature_builders=(_reply_full, _title_subcategory),
    ),
    "prefix_template_v1": GroupRule(
        rule_id="prefix_template_v1",
        description=(
            "Equal normalized first 300 reply characters (normalized-cut and the "
            "submitted raw-cut definition) OR equal normalized title+subcategory."
        ),
        signature_builders=(_reply_prefix, _reply_prefix_legacy, _title_subcategory),
    ),
}
CONTROL_RULE_IDS = ("duplicate_template_v1", "prefix_template_v1")


def _group_id(members: list[str]) -> str:
    payload = "\n".join(sorted(members))
    return "g" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_group_mapping(df: pd.DataFrame, rule_id: str) -> tuple[pd.Series, dict]:
    """Return (group_id per row, statistics) for one declared rule."""
    if rule_id not in GROUP_RULES:
        raise ValueError(f"Unknown group rule: {rule_id}; expected {sorted(GROUP_RULES)}")
    if df["seq_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate seq_id")
    rule = GROUP_RULES[rule_id]
    signatures = rule.signatures(df)

    parent = list(range(len(df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    linkage: dict[str, dict[str, int]] = {}
    for name, values in signatures.items():
        seen: dict[str, int] = {}
        empty = 0
        nonempty_values: list[str] = []
        for index, value in enumerate(values):
            if not value:
                empty += 1
                continue
            nonempty_values.append(value)
            if value in seen:
                union(index, seen[value])
            else:
                seen[value] = index
        counts = pd.Series(nonempty_values, dtype="object").value_counts()
        linkage[name] = {
            "nonempty": len(df) - empty,
            "empty": empty,
            "distinct_signatures": len(seen),
            "multi_ticket_groups": int((counts > 1).sum()) if len(counts) else 0,
            "tickets_in_multi_ticket_groups": int(counts[counts > 1].sum()) if len(counts) else 0,
            "largest_signature_group": int(counts.max()) if len(counts) else 0,
        }

    components: dict[int, list[str]] = {}
    for index, seq_id in enumerate(df["seq_id"].astype(str)):
        components.setdefault(find(index), []).append(seq_id)
    mapping = {}
    sizes = []
    for members in components.values():
        gid = _group_id(members)
        sizes.append(len(members))
        for seq_id in members:
            mapping[seq_id] = gid

    sizes_series = pd.Series(sizes)
    stats = {
        "rule_id": rule_id,
        "description": rule.description,
        "schema_version": GROUP_SCHEMA_VERSION,
        "n_tickets": int(len(df)),
        "n_groups": int(len(components)),
        "n_multi_ticket_groups": int((sizes_series > 1).sum()),
        "n_tickets_in_multi_ticket_groups": int(sizes_series[sizes_series > 1].sum()),
        "largest_group": int(sizes_series.max()),
        "largest_group_share": float(sizes_series.max() / len(df)),
        "group_size_quantiles": {
            str(q): float(sizes_series.quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "signature_linkage": linkage,
    }
    ordered = pd.Series([mapping[str(seq_id)] for seq_id in df["seq_id"]], index=df.index, name="group_id")
    return ordered, stats


def group_artifact(df: pd.DataFrame, rule_id: str) -> tuple[dict, dict]:
    """Return (artifact in the package-A schema, statistics)."""
    mapping, stats = build_group_mapping(df, rule_id)
    rule = GROUP_RULES[rule_id]
    artifact = {
        "schema_version": GROUP_SCHEMA_VERSION,
        "rule_id": rule_id,
        "semantic_status": rule.semantic_status,
        # Keyed by stable ticket ID, not by row position.
        "groups": {str(seq_id): str(gid) for seq_id, gid in zip(df["seq_id"], mapping)},
    }
    return artifact, stats


def write_group_artifacts(df: pd.DataFrame, output_dir: Path, rule_id: str,
                          overwrite: bool = False) -> dict:
    """Write ``<rule_id>.json`` (artifact) and ``<rule_id>_stats.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact, stats = group_artifact(df, rule_id)
    paths = {}
    for suffix, payload in ((".json", artifact), ("_stats.json", stats)):
        path = output_dir / f"{rule_id}{suffix}"
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") == text:
                paths[suffix] = path
                continue
            if not overwrite:
                raise FileExistsError(
                    f"{path} already exists with different content; pass --overwrite to regenerate"
                )
        path.write_text(text, encoding="utf-8")
        paths[suffix] = path
    return {
        "artifact": paths[".json"],
        "stats": paths["_stats.json"],
        "rule_id": rule_id,
        "n_groups": stats["n_groups"],
        "n_multi_ticket_groups": stats["n_multi_ticket_groups"],
        "n_tickets_in_multi_ticket_groups": stats["n_tickets_in_multi_ticket_groups"],
        "largest_group": stats["largest_group"],
    }
