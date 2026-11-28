#!/usr/bin/env python3
"""38_generated_ablation.py — aggregate the optional generated ablation.

Reads the `_abl_*` dev run folders produced by 23_revision_evaluate.py and writes
paired cosine effects, bootstrap intervals and conditioned-minus-blind contrasts.
No model or API call: answers and metrics are already stored.

Example after running the six cells:
    python experiments/38_generated_ablation.py --panel disjoint_prefix --split dev
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, make_run_id, setup_logging
from src.config import ProjectPaths
from src.data.artifacts import write_once_json

TAGS = ("ebpool_m4", "ebpool_m2", "lap_backoff")
SOURCES = ("conditioned", "blind")


def _latest_details(directory: Path) -> Path:
    files = sorted(directory.glob("*_details.json"))
    if not files:
        raise FileNotFoundError(f"No details in {directory}")
    return files[-1]


def _delta(records: list[dict], metric: str = "delta_cosine") -> pd.Series:
    values = {}
    for record in records:
        value = record["deltas"].get(metric)
        if value is not None:
            values[record["ticket_id"]] = value
    return pd.Series(values, dtype=float)


def _bootstrap(values: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "median": float(np.median(values)),
    }


def _p_value(values: np.ndarray) -> float:
    if not np.any(values != 0):
        return float("nan")
    return float(stats.wilcoxon(values).pvalue)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="disjoint_prefix")
    parser.add_argument("--split", default="dev", choices=["dev", "eval"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    run_id = make_run_id("generated_ablation")
    out_dir = paths.results / "revision" / "generated_ablation" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, per_ticket, runs_meta = [], {}, {}
    for tag in TAGS:
        for source in SOURCES:
            folder = f"M4_intersection_{args.split}_{source}_rev_{args.panel}_abl_{tag}"
            run_dir = paths.results / folder
            details_path = _latest_details(run_dir)
            records = json.loads(details_path.read_text(encoding="utf-8"))
            values = _delta(records)
            per_ticket[(tag, source)] = values
            runs_meta[f"{tag}/{source}"] = {
                "run_dir": str(run_dir),
                "details": details_path.name,
                "details_sha256": file_sha256(details_path),
                "n_records": len(records),
            }
            block = _bootstrap(values.to_numpy())
            rows.append({
                "tag": tag, "source": source, "n": len(values), **block,
                "wilcoxon_p": _p_value(values.to_numpy()),
                "pct_improved": float((values > 0).mean()),
                "pct_worsened": float((values < 0).mean()),
                "pct_unchanged": float((values == 0).mean()),
                "mean_abs_delta": float(np.abs(values).mean()),
            })

    contrasts = []
    for tag in TAGS:
        joined = pd.concat([
            per_ticket[(tag, "conditioned")].rename("conditioned"),
            per_ticket[(tag, "blind")].rename("blind"),
        ], axis=1).dropna()
        if joined.empty:
            continue
        difference = (joined["conditioned"] - joined["blind"]).to_numpy()
        block = _bootstrap(difference)
        contrasts.append({
            "tag": tag, "contrast": "conditioned_minus_blind", "n": len(difference),
            **block, "wilcoxon_p": _p_value(difference),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "generated_ablation.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out_dir / "generated_ablation_contrast.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "panel": args.panel,
        "split": args.split,
        "metric": "delta_cosine (generated answer vs recorded reference reply)",
        "runs": runs_meta,
        "git": git_info(),
    })
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
