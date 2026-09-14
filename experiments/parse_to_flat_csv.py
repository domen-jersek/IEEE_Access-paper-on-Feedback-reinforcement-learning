#!/usr/bin/env python3
"""
parse_to_flat_csv.py
====================
Flattens all *_details.json from evaluation runs into a single tidy CSV,
one row per (ticket, method) pair — matching the SIKDD paper's format.

Produces: results/sikdd_replication.csv

Columns:
  experiment_id, ticket_id, method, model, team, intent_class,
  baseline_cosine, feedback_cosine, delta_cosine,
  baseline_rouge, feedback_rouge, delta_rouge,
  baseline_top1_faiss, retrieval_margin,
  retrieval_overlap, gate_active,
  feedback_protocol, agg_mode, lift_formula, routing_name,
  config_hash

Usage:
  python experiments/parse_to_flat_csv.py
  python experiments/parse_to_flat_csv.py --input-dir results/ --output results/sikdd_replication.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def flatten_one(record: dict, exp_id: str) -> dict:
    b = record.get("baseline", {})
    f = record.get("feedback", {})
    d = record.get("deltas", {})
    pf = record.get("pool_features", {})

    return {
        "experiment_id": exp_id,
        "ticket_id": record.get("ticket_id", ""),
        "expected_team": record.get("expected_team", ""),
        "expected_class": record.get("expected_class", ""),
        "baseline_cosine": b.get("metrics", {}).get("cosine"),
        "feedback_cosine": f.get("metrics", {}).get("cosine"),
        "delta_cosine": d.get("delta_cosine"),
        "baseline_rouge": b.get("metrics", {}).get("rouge_l"),
        "feedback_rouge": f.get("metrics", {}).get("rouge_l"),
        "delta_rouge": d.get("delta_rouge_l"),
        "baseline_top1_faiss": pf.get("top1_faiss"),
        "top5_faiss_mean": pf.get("top5_faiss_mean"),
        "retrieval_margin": pf.get("retrieval_margin"),
        "top5_spread": pf.get("top5_spread"),
        "retrieval_overlap": record.get("retrieval_overlap"),
        "gate_active": record.get("gate_active"),
    }


def parse_all(input_dir: Path) -> pd.DataFrame:
    rows = []
    for df_path in sorted(input_dir.rglob("*_details.json")):
        try:
            data = json.loads(df_path.read_text("utf-8"))
        except Exception:
            continue

        exp_id = df_path.parent.name

        for rec in data:
            rows.append(flatten_one(rec, exp_id))

    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} rows from {input_dir}")
    print(f"  Experiments: {df['experiment_id'].nunique()}")
    print(f"  Tickets: {df['ticket_id'].nunique()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten evaluation JSON to CSV")
    parser.add_argument("--input-dir", type=str, default="results",
                        help="Directory with *_details.json files")
    parser.add_argument("--output", type=str, default="results/sikdd_replication.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}")
        sys.exit(1)

    df = parse_all(input_dir)
    df.to_csv(args.output, index=False)
    print(f"Written to {args.output}")
    print(f"\nQuick summary:")
    if "delta_cosine" in df.columns:
        for exp in sorted(df["experiment_id"].unique()):
            sub = df[df["experiment_id"] == exp]
            d = sub["delta_cosine"].dropna()
            if len(d):
                print(f"  {exp:45s}  n={len(d):3d}  mean_delta={d.mean():+.4f}  pct_imp={100*(d>0).mean():.1f}%")


if __name__ == "__main__":
    main()