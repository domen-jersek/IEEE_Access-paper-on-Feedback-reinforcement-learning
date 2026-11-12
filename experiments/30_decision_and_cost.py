#!/usr/bin/env python3
"""30_decision_and_cost.py — abstention headroom and generation cost.

Two free analyses on the frozen generated runs:

* R1-9 / R2-11 / R2-12: for every panel and evidence source, report always-on,
  never-on and the oracle upper bound (apply feedback only when the per-ticket
  delta is positive). The oracle is not deployable; it bounds what any gate could
  recover. Headroom = oracle - always_on. A gate is valuable only where headroom
  is non-trivial and the always-on mean is not already neutral.
* R2-15: wall-clock duration, request count and seconds per request from the run
  manifests and summaries. Token counts are not logged; this is an end-to-end
  bound that includes model loading and metric computation.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import make_run_id, setup_logging
from src.config import ProjectPaths
from src.data.artifacts import write_once_json

PANELS = ("legacy_random", "disjoint_duplicate", "disjoint_prefix")
SOURCES = ("conditioned", "blind", "reply_embed_bge")


def _latest(directory: Path, suffix: str) -> Path:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No {suffix} in {directory}")
    return files[-1]


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    decision_rows, cost_rows = [], []
    for panel in PANELS:
        for source in SOURCES:
            run_dir = paths.results / f"M4_intersection_eval_{source}_rev_{panel}"
            records = json.loads(_latest(run_dir, "_details.json").read_text(encoding="utf-8"))
            deltas = np.array([record["deltas"]["delta_cosine"] for record in records])
            always = float(deltas.mean())
            oracle = float(np.maximum(deltas, 0).mean())
            decision_rows.append({
                "panel": panel, "source": source, "n": len(deltas),
                "never_on": 0.0, "always_on": always, "oracle_upper_bound": oracle,
                "headroom": oracle - always,
                "pct_helped": float((deltas > 0).mean()),
                "pct_harmed": float((deltas < 0).mean()),
                "mean_gain_when_helped": float(deltas[deltas > 0].mean()) if (deltas > 0).any() else 0.0,
                "mean_loss_when_harmed": float(deltas[deltas < 0].mean()) if (deltas < 0).any() else 0.0,
            })

            manifest = json.loads((run_dir / "manifest_latest.json").read_text(encoding="utf-8"))
            summary = json.loads(_latest(run_dir, "_summary.json").read_text(encoding="utf-8"))
            duration = (_ts(summary["generated_utc"]) - _ts(manifest["timestamp_utc"])).total_seconds()
            cache = summary.get("generation_cache", {})
            requests = int(cache.get("hits", 0)) + int(cache.get("misses", 0))
            cost_rows.append({
                "panel": panel, "source": source,
                "queries": int(summary.get("total_queries", 0)),
                "requests": requests,
                "cache_hits": int(cache.get("hits", 0)),
                "cache_misses": int(cache.get("misses", 0)),
                "duration_seconds": round(duration, 1),
                "seconds_per_query": round(duration / max(int(summary.get("total_queries", 1)), 1), 2),
                "seconds_per_request": round(duration / max(requests, 1), 2),
            })

    run_id = make_run_id("decision_cost")
    out_dir = paths.results / "revision" / "decision_cost" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = pd.DataFrame(decision_rows)
    cost = pd.DataFrame(cost_rows)
    decision.to_csv(out_dir / "decision_headroom.csv", index=False)
    cost.to_csv(out_dir / "generation_cost.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "note": ("Oracle applies feedback only when the per-ticket delta is positive; "
                 "it is an upper bound for any gate, not a deployable policy. Duration "
                 "includes model loading and metric computation; token counts are not logged."),
    })
    print(decision[["panel", "source", "always_on", "oracle_upper_bound", "headroom",
                    "pct_helped", "pct_harmed"]].round(4).to_string(index=False))
    print()
    print(cost[["panel", "source", "queries", "requests", "duration_seconds",
                "seconds_per_request"]].to_string(index=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
