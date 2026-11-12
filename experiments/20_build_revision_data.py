#!/usr/bin/env python3
"""20_build_revision_data.py — versioned structural groups and group-aware splits.

No model, API or retrieval call. Outputs are written under
``data/processed/revision/`` (git-ignored; may contain local identifiers):

    groups/<rule>.json              package-A group artifact
    groups/<rule>_stats.json        component/signature statistics
    splits/split_<rule>_seed<N>.json  train/dev/eval ID lists
    audit/split_overlap.csv         overlap for legacy and revision splits
    manifest_<run_id>.json          input hashes and counts

Legacy ``data/processed/groups`` and ``data/processed/splits`` are not modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, setup_logging
from src.config import ProjectPaths
from src.data.audit import split_overlap_table
from src.data.groups_v2 import CONTROL_RULE_IDS, build_group_mapping, write_group_artifacts
from src.data.split_v2 import group_aware_split, validate_group_split


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        if not overwrite:
            raise FileExistsError(f"{path} exists with different content; pass --overwrite")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    dataset_path = paths.dataset_parquet
    df = pd.read_parquet(dataset_path)
    out = paths.data_processed / "revision"
    groups_dir = out / "groups"
    splits_dir = out / "splits"

    # 1. Versioned group artifacts (primary + conservative sensitivity rule).
    mappings: dict[str, pd.Series] = {}
    for rule_id in CONTROL_RULE_IDS:
        summary = write_group_artifacts(df, groups_dir, rule_id, overwrite=args.overwrite)
        mapping, stats = build_group_mapping(df, rule_id)
        mappings[rule_id] = mapping
        print(f"[groups] {rule_id}: groups={summary['n_groups']} "
              f"multi={summary['n_multi_ticket_groups']} "
              f"tickets_in_multi={summary['n_tickets_in_multi_ticket_groups']} "
              f"largest={summary['largest_group']}")

    # 2. Group-aware splits; group atomicity is validated for every seed.
    written = []
    for rule_id, mapping in mappings.items():
        for seed in args.seeds:
            split = group_aware_split(df, mapping, seed=seed)
            validate_group_split(df, mapping, split)
            payload = {
                "seed": seed,
                "regime": f"group_disjoint:{rule_id}",
                "group_rule_id": rule_id,
                "ticket_count": {k: len(split[k]) for k in ("train", "dev", "eval")},
                "train": split["train"],
                "dev": split["dev"],
                "eval": split["eval"],
            }
            path = splits_dir / f"split_{rule_id}_seed{seed}.json"
            _write_json(path, payload, args.overwrite)
            written.append(path)
            if seed == args.seeds[0]:
                print(f"[split] {rule_id} seed={seed}: sizes={payload['ticket_count']} "
                      f"max_dev={split['audit']['max_abs_deviation']}")

    # 3. Overlap audit: legacy random/disjoint versus revision splits.
    legacy = {
        "legacy_random_seed42": json.loads(
            (paths.splits_dir / "split_seed42.json").read_text(encoding="utf-8")),
        "legacy_disjoint_seed42": json.loads(
            (paths.splits_dir / "split_seed42_disjoint.json").read_text(encoding="utf-8")),
    }
    revision = {
        f"{rule_id}_seed{seed}": json.loads(
            (splits_dir / f"split_{rule_id}_seed{seed}.json").read_text(encoding="utf-8"))
        for rule_id in mappings for seed in args.seeds
    }
    table = split_overlap_table(
        df,
        {**legacy, **revision},
        {rule_id: dict(zip(df.seq_id.astype(str), mapping)) for rule_id, mapping in mappings.items()},
    )
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "split_overlap.csv"
    table.to_csv(audit_path, index=False)

    eval_overlap = table[
        table.pair.eq("train-eval")
        & table.split.isin(["legacy_random_seed42", "legacy_disjoint_seed42"])
    ]
    print("[audit] legacy train-eval overlap (fraction of eval tickets affected):")
    for row in eval_overlap.itertuples():
        print(f"  {row.split:26s} {row.family:34s} "
              f"{row.tickets_in_pairs:5d}/{row.second_partition_tickets} "
              f"({row.fraction_second_affected:.1%})")

    # 4. Data manifest.
    manifest = {
        "script": "experiments/20_build_revision_data.py",
        "git": git_info(),
        "dataset": {"path": str(dataset_path), "sha256": file_sha256(dataset_path)},
        "group_rules": {
            rule_id: {
                "artifact": str(groups_dir / f"{rule_id}.json"),
                "sha256": file_sha256(groups_dir / f"{rule_id}.json"),
            }
            for rule_id in mappings
        },
        "splits": {
            path.stem: {"sha256": file_sha256(path),
                        "ticket_count": json.loads(path.read_text("utf-8"))["ticket_count"]}
            for path in written
        },
        "overlap_audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
    }
    manifest_path = out / "manifest_build_revision_data.json"
    _write_json(manifest_path, manifest, args.overwrite)
    print(f"[done] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
