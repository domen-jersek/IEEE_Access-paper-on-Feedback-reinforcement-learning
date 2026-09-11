#!/usr/bin/env python3
"""
06_report.py — Phase 5
========================
Aggregates all evaluation results and produces final summary tables.
Reads *_details.json from all method runs and computes:
- method comparison table
- oracle decile analysis
- gate cutoff sweep
- oracle ceiling
- bootstrap CIs + Wilcoxon
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from experiments.utils import setup_logging, get_arg_parser
from src.evaluation.reporting import (
    bootstrap_ci,
    wilcoxon_paired,
    cohens_d,
    oracle_decile_analysis,
    gate_cutoff_sweep,
    oracle_ceiling_analysis,
    load_results_json,
)


def main() -> None:
    parser = get_arg_parser("Final report generator")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with *_details.json files (default: results/)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for report CSVs (default: results/report/)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    log = logging.getLogger("report")

    paths = ProjectPaths()
    input_dir = Path(args.input_dir) if args.input_dir else paths.results
    output_dir = Path(args.output_dir) if args.output_dir else (paths.results / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_files = sorted(input_dir.rglob("*_details.json"))
    log.info("Found %d detail files", len(detail_files))

    all_deltas = {}
    all_results = {}

    for df_path in detail_files:
        try:
            data = load_results_json(df_path)
            exp_id = df_path.parent.name if df_path.parent.name else df_path.stem.replace("_details", "")
            deltas = np.array([r.get("deltas", {}).get("delta_cosine", 0.0) for r in data if "deltas" in r])
            if len(deltas) > 0:
                all_deltas[exp_id] = deltas
                all_results[exp_id] = data
        except Exception as e:
            log.warning("Skipping %s: %s", df_path, e)

    method_rows = []
    for exp_id, deltas in all_deltas.items():
        ci = bootstrap_ci(deltas, n_resamples=5000)
        wc = wilcoxon_paired(deltas)
        method_rows.append({
            "experiment_id": exp_id,
            "n": len(deltas),
            "mean_delta": ci["mean"],
            "median_delta": ci["median"],
            "std_delta": ci["std"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "pct_improved": float(np.mean(deltas > 0)),
            "pct_worsened": float(np.mean(deltas < 0)),
            "wilcoxon_p": wc["p_value"],
            "wilcoxon_sig_005": wc["significant_005"],
            "cohens_d": cohens_d(deltas),
        })

    method_df = pd.DataFrame(method_rows)
    method_df.to_csv(output_dir / "method_comparison.csv", index=False)
    log.info("Method comparison: %d experiments", len(method_df))
    log.info(method_df[["experiment_id", "n", "mean_delta", "wilcoxon_p"]].to_string())

    for exp_id, data in all_results.items():
        try:
            oracle_df = oracle_decile_analysis(data)
            oracle_df.to_csv(output_dir / f"oracle_deciles_{exp_id}.csv", index=False)

            sweep_df = gate_cutoff_sweep(data, "top1_faiss")
            sweep_df.to_csv(output_dir / f"gate_sweep_{exp_id}.csv", index=False)
        except Exception as e:
            log.warning("Skipping analysis for %s: %s", exp_id, e)

    log.info("=== DONE ===")
    log.info("Reports written to %s", output_dir)


if __name__ == "__main__":
    main()