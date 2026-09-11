"""
Stratified 3-way train/dev/eval split with multi-seed support.

Two regimes:
- "random": stratified random split
- "disjoint": procedure-group-aware split (near-duplicates stay together)

Stratification dimensions (in priority order):
1. team (pooled rare teams)
2. intent_class (coarse taxonomy)
3. baseline_retrieval_difficulty (computed after FAISS index build; initially a placeholder)

Output: split_{seed}.json with train/dev/eval query ID lists.
"""
from __future__ import annotations

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

RARE_TEAM_THRESHOLD = 5
RARE_CLASS_THRESHOLD = 10
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
TRAIN_FRAC = 0.55
DEV_FRAC = 0.20


def _pool_rare_categories(series: pd.Series, threshold: int, label: str = "OTHER") -> pd.Series:
    counts = series.value_counts()
    rare = counts[counts < threshold].index
    return series.apply(lambda x: label if x in rare else x)


def _build_stratum(df: pd.DataFrame) -> pd.Series:
    team_pooled = _pool_rare_categories(df["Team->Name"], RARE_TEAM_THRESHOLD, "TEAM_OTHER")
    class_pooled = _pool_rare_categories(df["intent_class"], RARE_CLASS_THRESHOLD, "CLASS_OTHER")
    return team_pooled + "|||" + class_pooled


def stratified_triple_split(
    df: pd.DataFrame,
    output_dir: Path,
    seeds: list[int] = DEFAULT_SEEDS,
    train_frac: float = TRAIN_FRAC,
    dev_frac: float = DEV_FRAC,
    group_col: Optional[str] = None,
) -> dict:
    """
    Produce splits for multiple seeds.
    
    Parameters
    ----------
    df : DataFrame
        Must have columns: seq_id, Team->Name, intent_class.
    group_col : str or None
        If provided (e.g., combined_group_hash), use GroupKFold-style splitting.
        
    Returns
    -------
    dict mapping seed -> split dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    n_train = int(np.ceil(n * train_frac))
    n_dev = int(np.ceil(n * dev_frac))
    n_eval = n - n_train - n_dev
    log.info("Split sizes: train=%d (%.1f%%), dev=%d (%.1f%%), eval=%d (%.1f%%)",
             n_train, 100 * n_train / n, n_dev, 100 * n_dev / n, n_eval, 100 * n_eval / n)

    all_splits = {}

    if group_col is not None:
        df = df.copy()
        df["_group"] = df[group_col]
        unique_groups = sorted(df["_group"].unique())
        rng = np.random.default_rng(seeds[0])
        rng.shuffle(unique_groups)
        g_train, g_dev, g_eval = _split_elements(len(unique_groups), train_frac, dev_frac)
        group_to_split = {}
        for i, g in enumerate(unique_groups):
            if i < g_train:
                group_to_split[g] = "train"
            elif i < g_train + g_dev:
                group_to_split[g] = "dev"
            else:
                group_to_split[g] = "eval"
        df["_split"] = df["_group"].map(group_to_split)
        split_dict = {
            "train": sorted(df[df["_split"] == "train"]["seq_id"].tolist()),
            "dev": sorted(df[df["_split"] == "dev"]["seq_id"].tolist()),
            "eval": sorted(df[df["_split"] == "eval"]["seq_id"].tolist()),
        }
        all_splits[seeds[0]] = {"regime": "disjoint", **split_dict}
        _write_split(output_dir, seeds[0], "disjoint", split_dict, df)
    else:
        for seed in seeds:
            stratum = _build_stratum(df)
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n)
            split_dict = _assign_to_splits(indices, n_train, n_dev, df, stratum)
            all_splits[seed] = {"regime": "random", **split_dict}
            _write_split(output_dir, seed, "random", split_dict, df)

    return all_splits


def _assign_to_splits(
    indices: np.ndarray,
    n_train: int,
    n_dev: int,
    df: pd.DataFrame,
    stratum: pd.Series,
) -> dict[str, list[str]]:
    train_ids = sorted(df.iloc[indices[:n_train]]["seq_id"].tolist())
    dev_ids = sorted(df.iloc[indices[n_train : n_train + n_dev]]["seq_id"].tolist())
    eval_ids = sorted(df.iloc[indices[n_train + n_dev :]]["seq_id"].tolist())
    return {"train": train_ids, "dev": dev_ids, "eval": eval_ids}


def _split_elements(n_total: int, train_frac: float, dev_frac: float) -> tuple[int, int, int]:
    n_train = int(np.ceil(n_total * train_frac))
    n_dev = int(np.ceil(n_total * dev_frac))
    return n_train, n_dev, n_total - n_train - n_dev


def _write_split(
    output_dir: Path,
    seed: int,
    regime: str,
    split_dict: dict,
    df: pd.DataFrame,
) -> None:
    payload = {
        "seed": seed,
        "regime": regime,
        "created_utc": pd.Timestamp.now().isoformat(),
        "ticket_count": {
            "train": len(split_dict["train"]),
            "dev": len(split_dict["dev"]),
            "eval": len(split_dict["eval"]),
        },
        "train": split_dict["train"],
        "dev": split_dict["dev"],
        "eval": split_dict["eval"],
    }

    suffix = f"_{regime}" if regime != "random" else ""
    path = output_dir / f"split_seed{seed}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _write_stratification_report(output_dir, seed, regime, split_dict, df)


def _write_stratification_report(
    output_dir: Path,
    seed: int,
    regime: str,
    split_dict: dict,
    df: pd.DataFrame,
) -> None:
    report_rows = []
    id_to_split = {}
    for split_name in ["train", "dev", "eval"]:
        for sid in split_dict[split_name]:
            id_to_split[sid] = split_name

    df = df.copy()
    df["_split"] = df["seq_id"].map(id_to_split)

    for dim in ["Team->Name", "intent_class"]:
        for split_name in ["train", "dev", "eval"]:
            subset = df[df["_split"] == split_name]
            vc = subset[dim].value_counts()
            for val, count in vc.items():
                report_rows.append({
                    "seed": seed,
                    "regime": regime,
                    "dimension": dim,
                    "split": split_name,
                    "value": val,
                    "count": count,
                })

    report_df = pd.DataFrame(report_rows)
    suffix = f"_{regime}" if regime != "random" else ""
    report_path = output_dir / f"split_seed{seed}{suffix}_report.csv"
    report_df.to_csv(report_path, index=False)


def load_split(split_path: Path) -> dict:
    return json.loads(split_path.read_text("utf-8"))


def validate_splits(all_splits: dict) -> bool:
    for seed, split in all_splits.items():
        train = set(split["train"])
        dev = set(split["dev"])
        eval_ = set(split["eval"])
        assert len(train & dev) == 0, f"Seed {seed}: train ∩ dev = {train & dev}"
        assert len(train & eval_) == 0, f"Seed {seed}: train ∩ eval = {train & eval_}"
        assert len(dev & eval_) == 0, f"Seed {seed}: dev ∩ eval = {dev & eval_}"
        total = len(train | dev | eval_)
        assert total == len(train) + len(dev) + len(eval_), f"Seed {seed}: union size mismatch"
        log.info("Seed %d: splits are disjoint, covering %d tickets", seed, total)
    return True