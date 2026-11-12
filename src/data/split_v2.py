"""Group-aware three-way split for the revision (group atomicity enforced).

Groups are assigned as indivisible units. Within each stratification value the
largest groups are placed first into the partition with the largest relative
deficit, which keeps partition sizes close to the requested fractions without
splitting a group. Ordering uses a seeded permutation, so the split is
reproducible for a given dataset, group mapping and seed.

No outcome (retrieval score, feedback lift, metric) enters the assignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

PARTITIONS = ("train", "dev", "eval")


@dataclass(frozen=True)
class SplitAudit:
    n_tickets: int
    n_groups: int
    sizes: dict
    targets: dict
    largest_group: int
    max_abs_deviation: int

    def to_dict(self) -> dict:
        return {
            "n_tickets": self.n_tickets,
            "n_groups": self.n_groups,
            "sizes": self.sizes,
            "targets": self.targets,
            "largest_group": self.largest_group,
            "max_abs_deviation": self.max_abs_deviation,
        }


def _targets(n: int, train_frac: float, dev_frac: float) -> dict[str, int]:
    n_train = int(math.ceil(n * train_frac))
    n_dev = int(math.ceil(n * dev_frac))
    return {"train": n_train, "dev": n_dev, "eval": n - n_train - n_dev}


def _majority(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    return sorted(counts.index, key=lambda k: (-counts[k], k))[0]


def group_aware_split(
    df: pd.DataFrame,
    group_ids: pd.Series,
    *,
    seed: int = 42,
    train_frac: float = 0.55,
    dev_frac: float = 0.20,
    stratify_col: str = "intent_class",
) -> dict:
    """Return ``{"train": [...], "dev": [...], "eval": [...], "audit": {...}}``."""
    df = df.reset_index(drop=True)
    if len(group_ids) != len(df):
        raise ValueError("group_ids must align with df")
    if df["seq_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate seq_id")
    if not (0 < train_frac < 1 and 0 < dev_frac < 1 and train_frac + dev_frac < 1):
        raise ValueError("Invalid split fractions")
    if stratify_col not in df.columns:
        raise ValueError(f"Missing stratification column: {stratify_col}")

    frame = pd.DataFrame({
        "seq_id": df["seq_id"].astype(str),
        "group_id": group_ids.astype(str).to_numpy(),
        "stratum": df[stratify_col].astype(str).to_numpy(),
    })
    grouped = frame.groupby("group_id", sort=True)
    sizes = grouped.size()
    strata = grouped["stratum"].agg(_majority)

    rng = np.random.default_rng(seed)
    shuffled = list(sizes.index)
    rng.shuffle(shuffled)
    rank = {group: position for position, group in enumerate(shuffled)}
    # Random rank is the final tie-break, so the seed genuinely changes the split
    # while the same seed always reproduces it.
    order = sorted(shuffled, key=lambda g: (strata[g], -int(sizes[g]), rank[g]))

    targets = _targets(len(df), train_frac, dev_frac)
    counts = {p: 0 for p in PARTITIONS}
    assignment: dict[str, str] = {}
    for group in order:
        size = int(sizes[group])
        # Relative deficit keeps the three partitions close to their targets.
        choice = max(PARTITIONS, key=lambda p: (targets[p] - counts[p]) / max(targets[p], 1))
        assignment[group] = choice
        counts[choice] += size

    parts = {p: sorted(frame.loc[frame.group_id.map(assignment).eq(p), "seq_id"].tolist())
             for p in PARTITIONS}
    total = sum(len(parts[p]) for p in PARTITIONS)
    if total != len(df) or set().union(*(set(parts[p]) for p in PARTITIONS)) != set(frame.seq_id):
        raise RuntimeError("Group split did not assign every ticket exactly once")
    if any(not parts[p] for p in PARTITIONS):
        raise RuntimeError(f"Empty partition for seed {seed}")

    audit = SplitAudit(
        n_tickets=len(df),
        n_groups=int(len(sizes)),
        sizes={p: len(parts[p]) for p in PARTITIONS},
        targets=targets,
        largest_group=int(sizes.max()),
        max_abs_deviation=int(max(abs(len(parts[p]) - targets[p]) for p in PARTITIONS)),
    )
    return {**parts, "audit": audit.to_dict(), "seed": seed}


def validate_group_split(df: pd.DataFrame, group_ids: pd.Series, split: dict) -> None:
    """Raise if any group crosses partitions or coverage is incomplete."""
    frame = pd.DataFrame({"seq_id": df["seq_id"].astype(str), "group_id": group_ids.astype(str).to_numpy()})
    id_to_group = dict(zip(frame.seq_id, frame.group_id))
    seen: dict[str, str] = {}
    for partition in PARTITIONS:
        for seq_id in split[partition]:
            if seq_id not in id_to_group:
                raise ValueError(f"Unknown ticket in split: {seq_id}")
            group = id_to_group[seq_id]
            previous = seen.setdefault(group, partition)
            if previous != partition:
                raise ValueError(f"Group {group} crosses {previous}/{partition}")
    if set(seen) != set(id_to_group.values()):
        raise ValueError("Split does not cover every group")
    union = set().union(*(set(split[p]) for p in PARTITIONS))
    if union != set(frame.seq_id):
        raise ValueError("Split does not cover every ticket exactly once")
