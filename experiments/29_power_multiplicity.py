#!/usr/bin/env python3
"""29_power_multiplicity.py — pre-specified families, Holm correction and MDE.

Reads the generated panel deltas, fixes three test families before looking at
adjusted values, applies Holm within each family, and reports the minimum
detectable effect (80 % power, two-sided alpha .05) from the observed per-ticket
standard deviation.

Families:
  A (primary)   conditioned and blind, three panels          6 tests
  B (secondary) reply similarity, three panels               3 tests
  C (secondary) conditioned-minus-blind contrast, three panels 3 tests

MDE is a design sensitivity statement, not evidence of absence: a non-significant
result whose interval includes effects above the MDE is inconclusive, not null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import make_run_id, setup_logging
from src.config import ProjectPaths
from src.data.artifacts import write_once_json

Z_ALPHA = 1.959963984540054
Z_POWER = 0.8416212335729143


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=None,
                        help="generated_panels run directory (default: latest)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    run_dir = args.run
    if run_dir is None:
        run_dir = sorted((paths.results / "revision" / "generated_panels").glob("generated_panels_*"))[-1]
    deltas = pd.read_csv(run_dir / "generated_deltas.csv")
    contrasts = pd.read_csv(run_dir / "protocol_contrast.csv")

    rows = []
    families = {
        "A_primary": deltas[deltas.source.isin(["conditioned", "blind"])],
        "B_reply": deltas[deltas.source.eq("reply_embed_bge")],
        "C_contrast": contrasts.assign(source="conditioned_minus_blind",
                                       metric="delta_cosine"),
    }
    for family, frame in families.items():
        frame = frame[frame.metric.eq("delta_cosine")].copy()
        frame["p_raw"] = frame["wilcoxon_p"].astype(float)
        frame["p_holm"] = holm(frame["p_raw"].tolist())
        for _, row in frame.iterrows():
            sd = float(row["std"]) if "std" in row and not pd.isna(row["std"]) else float("nan")
            n = int(row["n"])
            mde = (Z_ALPHA + Z_POWER) * sd / np.sqrt(n) if np.isfinite(sd) and n else float("nan")
            rows.append({
                "family": family, "panel": row["panel"], "source": row["source"],
                "n": n, "mean": float(row["mean"]),
                "ci_lower": float(row["ci_lower"]), "ci_upper": float(row["ci_upper"]),
                "p_raw": float(row["p_raw"]), "p_holm": float(row["p_holm"]),
                "sd": sd, "mde_80": float(mde),
                "mean_over_mde": float(abs(row["mean"]) / mde) if mde else float("nan"),
                "ci_excludes_zero": bool(row["ci_lower"] * row["ci_upper"] > 0),
            })
    table = pd.DataFrame(rows).sort_values(["family", "panel", "source"])

    run_id = make_run_id("power_multiplicity")
    out_dir = paths.results / "revision" / "power_multiplicity" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "power_multiplicity.csv", index=False)
    write_once_json(out_dir / "summary.json", {
        "run_id": run_id,
        "source_run": str(run_dir),
        "families": {k: len(v[v.metric.eq("delta_cosine")]) for k, v in families.items()},
        "mde_formula": "(z_0.975 + z_0.80) * sd / sqrt(n)",
        "note": "MDE is design sensitivity, not evidence of absence.",
    })
    print(table[["family", "panel", "source", "n", "mean", "p_raw", "p_holm",
                 "sd", "mde_80", "mean_over_mde"]].round(4).to_string(index=False))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
