#!/usr/bin/env python3
"""33_human_eval.py — per-rater analysis and two-rater agreement.

Usage:
    python experiments/33_human_eval.py --package data/processed/revision/study/v1 \
        --ratings revision_ieee_access/02_human_study/evaluation_alberto/ratings.csv

    python experiments/33_human_eval.py --package ... --ratings rater1.csv --ratings rater2.csv

The same pipeline runs for every rater. With two raters it additionally writes
agreement (percent, kappa with bootstrap interval, PABAK, AC1, weighted kappa) and
a combined per-item effect. Outputs are aggregates; no ticket text is written.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import make_run_id, setup_logging
from src.data.artifacts import write_once_json
from src.evaluation.human_eval import (
    Study,
    agreement_tables,
    combined_effect,
    load_ratings,
    load_study,
    orient,
    rater_tables,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _figures(rater: str, df: pd.DataFrame, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = sorted(df["arm"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(arms), 1)))
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for color, arm in zip(colors, arms):
        group = df[df["arm"] == arm]
        human = (group["feedback_strict"] - group["baseline_strict"]).to_numpy()
        axes[0].scatter(group["auto_delta"], human + np.random.default_rng(1).normal(0, 0.02, len(group)),
                        s=22, alpha=0.75, color=color, label=arm)
        axes[1].scatter(group["len_diff"], human, s=22, alpha=0.75, color=color, label=arm)
    for ax in axes:
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xlabel("automatic cosine delta" if ax is axes[0] else "length difference (chars)")
        ax.set_ylabel("expert strict delta")
        ax.grid(alpha=0.2)
    axes[0].set_title("Expert vs automatic")
    axes[1].set_title("Expert vs answer length")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = out_dir / f"rater_{rater}_diagnostics.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/revision/human_eval"))
    parser.add_argument("--label", default="")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    study: Study = load_study(args.package)
    oriented = []
    rater_ids = []
    for path in args.ratings:
        ratings = load_ratings(path, study)
        raters = sorted(ratings["rater"].astype(str).unique())
        if len(raters) != 1:
            raise SystemExit(f"{path}: expected one rater, found {raters}")
        if raters[0] in rater_ids:
            raise SystemExit(f"Duplicate rater across files: {raters[0]}")
        rater_ids.append(raters[0])
        oriented.append(orient(ratings, study))

    run_id = make_run_id("human_eval" + (f"_{args.label}" if args.label else ""))
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = pd.DataFrame([
        {"arm": arm, "requested": meta.get("requested"), "n": meta.get("n"),
         "population": meta.get("population"), "changed": meta.get("changed"),
         "changed_share": meta.get("changed_share"), "eligible": meta.get("eligible"),
         "excluded_inconsistent": meta.get("excluded_inconsistent")}
        for arm, meta in study.manifest.get("sample", {}).items()
    ])
    overview.to_csv(out_dir / "study_overview.csv", index=False)

    per_rater = {}
    for rater, df in zip(rater_ids, oriented):
        tables = rater_tables(df, study)
        for name, table in tables.items():
            table.to_csv(out_dir / f"rater_{rater}_{name}.csv", index=False)
        item_level = pd.DataFrame({
            "arm": df["arm"],
            "item_hash": [hashlib.sha256(str(i).encode()).hexdigest()[:12] for i in df["item_id"]],
            "auto_delta": df["auto_delta"],
            "human_delta": df["feedback_strict"] - df["baseline_strict"],
            "len_diff": df["len_diff"],
            "preferred": df["preferred"],
            "baseline_strict": df["baseline_strict"],
            "feedback_strict": df["feedback_strict"],
        })
        item_level.to_csv(out_dir / f"rater_{rater}_item_level.csv", index=False)
        if not args.no_figures:
            _figures(rater, df, out_dir / "figures")
        per_rater[rater] = {
            "n_items": len(df),
            "arms": df["arm"].value_counts().to_dict(),
            "tables": {name: len(table) for name, table in tables.items()},
        }
        print(f"[{rater}] items={len(df)} arms={df['arm'].value_counts().to_dict()}")
        print(tables["summary"][tables["summary"].metric.eq("strict")][
            ["panel", "source", "n", "baseline_rate", "feedback_rate", "delta",
             "ci_lower", "ci_upper", "mcnemar_p"]].round(4).to_string(index=False))
        print()

    agreement = {}
    if len(oriented) == 2:
        tables = agreement_tables(oriented[0], oriented[1])
        for name, table in tables.items():
            table.to_csv(out_dir / f"agreement_{name}.csv", index=False)
        combined = combined_effect(oriented[0], oriented[1])
        combined.to_csv(out_dir / "combined_effect.csv", index=False)
        agreement = {name: table.to_dict(orient="records") for name, table in tables.items()}
        print("Agreement (preference):")
        print(tables["agreement"][["panel", "source", "n_items", "preference_agreement",
                                   "preference_kappa", "preference_kappa_ci_lower",
                                   "preference_kappa_ci_upper", "preference_ac1"]].round(3).to_string(index=False))
        print("\nCombined effect:")
        print(combined.round(4).to_string(index=False))
    else:
        print("One rater: agreement and combined effect are not computed.")

    summary = {
        "run_id": run_id,
        "package": str(args.package.resolve()),
        "package_id": study.package_id,
        "ratings": [{"path": str(p.resolve()), "sha256": _sha(p)} for p in args.ratings],
        "raters": rater_ids,
        "per_rater": per_rater,
        "agreement": agreement,
        "note": ("Descriptive expert study. Strict = correct only; lenient = correct or partially "
                 "correct. Ties, both-incorrect and cannot-judge are reported separately. "
                 "No ticket text is included."),
    }
    write_once_json(out_dir / "summary.json", summary)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
