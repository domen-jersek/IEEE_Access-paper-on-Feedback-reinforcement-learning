#!/usr/bin/env python3
"""26_generated_panels.py — generated-answer effects and protocol contrasts.

Reads the completed revision runs (one directory per panel × evidence source),
computes per-ticket paired deltas of the generated answer against its reference
and reports bootstrap intervals, Wilcoxon p-values and changed/unchanged shares.

Also reports the direct conditioned-minus-blind contrast per panel (R2-6), which
requires the same tickets in both runs.

No model or API call: answers and metrics are already stored.
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

PANELS = ("legacy_random", "disjoint_duplicate", "disjoint_prefix")
SOURCES = ("conditioned", "blind", "reply_embed_bge")


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
    medians = np.median(values[draws], axis=1)
    return {
        "mean": float(values.mean()),
        "ci_lower": float(np.percentile(means, 2.5)),
        "ci_upper": float(np.percentile(means, 97.5)),
        "median": float(np.median(values)),
        "median_ci_lower": float(np.percentile(medians, 2.5)),
        "median_ci_upper": float(np.percentile(medians, 97.5)),
    }


def _p_value(values: np.ndarray) -> float:
    if not np.any(values != 0):
        return float("nan")
    return float(stats.wilcoxon(values).pvalue)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", default=list(PANELS))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    run_id = make_run_id("generated_panels")
    out_dir = paths.results / "revision" / "generated_panels" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, per_ticket, runs_meta = [], {}, {}
    metrics = ("delta_cosine", "delta_rouge_l")
    for panel in args.panels:
        for source in args.sources:
            run_dir = paths.results / f"M4_intersection_eval_{source}_rev_{panel}"
            details_path = _latest_details(run_dir)
            records = json.loads(details_path.read_text(encoding="utf-8"))
            deltas = _delta(records, "delta_cosine")
            per_ticket[(panel, source)] = deltas
            runs_meta[f"{panel}/{source}"] = {
                "run_dir": str(run_dir), "details": details_path.name,
                "details_sha256": file_sha256(details_path), "n_records": len(records),
            }
            for metric in metrics:
                values = _delta(records, metric)
                if values.empty:
                    continue
                stats_block = _bootstrap(values.to_numpy())
                rows.append({
                    "panel": panel, "source": source, "metric": metric, "n": len(values),
                    **stats_block,
                    "wilcoxon_p": _p_value(values.to_numpy()),
                    "pct_improved": float((values > 0).mean()),
                    "pct_worsened": float((values < 0).mean()),
                    "pct_unchanged": float((values == 0).mean()),
                    "mean_abs_delta": float(np.abs(values).mean()),
                    "std": float(values.std(ddof=1)),
                })

    contrasts = []
    for panel in args.panels:
        if (panel, "conditioned") in per_ticket and (panel, "blind") in per_ticket:
            joined = pd.concat([per_ticket[(panel, "conditioned")].rename("conditioned"),
                                per_ticket[(panel, "blind")].rename("blind")], axis=1).dropna()
            if joined.empty:
                continue
            difference = (joined["conditioned"] - joined["blind"]).to_numpy()
            stats_block = _bootstrap(difference)
            contrasts.append({
                "panel": panel, "contrast": "conditioned_minus_blind", "n": len(difference),
                **stats_block, "wilcoxon_p": _p_value(difference),
                "std": float(difference.std(ddof=1)),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "generated_deltas.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out_dir / "protocol_contrast.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "metric": "delta_cosine (generated answer vs recorded reference reply)",
        "runs": runs_meta,
        "git": git_info(),
    })
    print(summary[summary.metric.eq("delta_cosine")][
        ["panel", "source", "n", "mean", "ci_lower", "ci_upper", "median",
         "wilcoxon_p", "pct_improved", "pct_worsened", "pct_unchanged"]].round(4).to_string(index=False))
    if contrasts:
        print()
        print(pd.DataFrame(contrasts)[["panel", "n", "mean", "ci_lower", "ci_upper",
                                       "wilcoxon_p"]].round(4).to_string(index=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
