#!/usr/bin/env python3
"""25_analyze_study.py — agreement and paired analysis of the expert ratings.

Input is the CSV exported by the study application. It contains the hidden side
mapping (`a_side`, `b_side`), so no method identity is needed from the app.

Outputs (private): per-stratum correctness and preference summaries with bootstrap
intervals, plus inter-rater agreement (percent agreement and Cohen's kappa) for
preference and for strict correctness.

Statistics are descriptive: two experts are not a population sample. Report ties,
both-incorrect and cannot-judge explicitly; do not collapse them into a preference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths

CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}
STRICT = {"correct": 1.0, "partially_correct": 0.0, "incorrect": 0.0}


def _side_value(row: pd.Series, prefix: str, side: str) -> str:
    column = "a" if row["a_side"] == side else "b"
    return str(row[f"{column}_{prefix}"])


def _orient(df: pd.DataFrame) -> pd.DataFrame:
    required = {"rater", "item_id", "panel", "ticket_id", "source", "block",
                "a_correct", "b_correct", "preference", "a_side", "b_side"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    out["baseline_correct"] = [_side_value(row, "correct", "baseline") for _, row in out.iterrows()]
    out["feedback_correct"] = [_side_value(row, "correct", "feedback") for _, row in out.iterrows()]
    out["baseline_score"] = out["baseline_correct"].map(CORRECTNESS_SCORE)
    out["feedback_score"] = out["feedback_correct"].map(CORRECTNESS_SCORE)
    out["baseline_strict"] = out["baseline_correct"].map(STRICT)
    out["feedback_strict"] = out["feedback_correct"].map(STRICT)

    def preferred(row) -> str:
        if row["preference"] in {"A", "B"}:
            return row["a_side"] if row["preference"] == "A" else row["b_side"]
        return row["preference"]

    out["preferred"] = out.apply(preferred, axis=1)
    out["baseline_better"] = out["preferred"].eq("baseline").astype(float)
    out["feedback_better"] = out["preferred"].eq("feedback").astype(float)

    for flag in ("unsafe", "missing"):
        a_column, b_column = f"a_{flag}", f"b_{flag}"
        if a_column in out and b_column in out:
            out[f"baseline_{flag}"] = np.where(out["a_side"].eq("baseline"),
                                               out[a_column], out[b_column])
            out[f"feedback_{flag}"] = np.where(out["a_side"].eq("feedback"),
                                               out[a_column], out[b_column])
    return out


def _bootstrap_mean(values: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> dict:
    if len(values) == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5))}


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    if len(labels) < 2:
        return float("nan")
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)))
    for left, right in zip(a, b):
        matrix[index[left], index[right]] += 1
    total = matrix.sum()
    observed = np.trace(matrix) / total
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / total ** 2)
    return float((observed - expected) / (1 - expected)) if expected < 1 else float("nan")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Study manifest; enables population extrapolation by changed share")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = ProjectPaths()
    out_dir = args.out_dir or (paths.results / "revision" / "study_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _orient(pd.read_csv(args.ratings))
    sample_meta = {}
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        sample_meta = manifest.get("sample", {})
    rows = []
    for keys, group in df.groupby(["panel", "source", "block"]):
        panel, source, block = keys
        delta_strict = group["feedback_strict"].to_numpy() - group["baseline_strict"].to_numpy()
        delta_score = group["feedback_score"].to_numpy() - group["baseline_score"].to_numpy()
        strict = _bootstrap_mean(delta_strict)
        meta = sample_meta.get(f"{panel}/{source}", {})
        share = meta.get("changed_share")
        rows.append({
            "panel": panel, "source": source, "block": block, "n": len(group),
            "baseline_strict_rate": group["baseline_strict"].mean(),
            "feedback_strict_rate": group["feedback_strict"].mean(),
            "strict_delta": strict["mean"],
            "strict_delta_ci_lower": strict["ci_lower"],
            "strict_delta_ci_upper": strict["ci_upper"],
            "changed_share": share,
            "population_delta": strict["mean"] * share if share is not None else None,
            "population_ci_lower": strict["ci_lower"] * share if share is not None else None,
            "population_ci_upper": strict["ci_upper"] * share if share is not None else None,
            "score_delta": delta_score.mean(),
            "feedback_better_rate": group["feedback_better"].mean(),
            "baseline_better_rate": group["baseline_better"].mean(),
            "net_feedback_better": group["feedback_better"].mean() - group["baseline_better"].mean(),
            "equal_rate": group["preferred"].eq("equal").mean(),
            "both_incorrect_rate": group["preferred"].eq("both_incorrect").mean(),
            "cannot_judge_rate": group["preferred"].eq("cannot_judge").mean(),
            "baseline_unsafe_rate": (group["baseline_unsafe"].mean()
                                     if "baseline_unsafe" in group else float("nan")),
            "feedback_unsafe_rate": (group["feedback_unsafe"].mean()
                                     if "feedback_unsafe" in group else float("nan")),
            "baseline_missing_rate": (group["baseline_missing"].mean()
                                      if "baseline_missing" in group else float("nan")),
            "feedback_missing_rate": (group["feedback_missing"].mean()
                                      if "feedback_missing" in group else float("nan")),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "study_summary.csv", index=False)

    agreement = []
    raters = sorted(df["rater"].unique())
    for block in sorted(df["block"].unique()):
        left = df[(df.rater == raters[0]) & (df.block == block)]
        right = df[(df.rater == raters[1]) & (df.block == block)] if len(raters) > 1 else None
        if right is None:
            continue
        merged = left.merge(right, on="item_id", suffixes=("_1", "_2"))
        if merged.empty:
            continue
        preference_1 = merged["preference_1"].astype(str)
        preference_2 = merged["preference_2"].astype(str)
        strict_1 = merged["baseline_strict_1"].astype(str) + "|" + merged["feedback_strict_1"].astype(str)
        strict_2 = merged["baseline_strict_2"].astype(str) + "|" + merged["feedback_strict_2"].astype(str)
        agreement.append({
            "block": block, "n_items": len(merged),
            "preference_percent_agreement": float((preference_1 == preference_2).mean()),
            "preference_kappa": _cohen_kappa(preference_1.tolist(), preference_2.tolist()),
            "strict_pair_percent_agreement": float((strict_1 == strict_2).mean()),
            "strict_pair_kappa": _cohen_kappa(strict_1.tolist(), strict_2.tolist()),
        })
    pd.DataFrame(agreement).to_csv(out_dir / "study_agreement.csv", index=False)

    payload = {
        "raters": raters,
        "n_ratings": len(df),
        "n_items": int(df["item_id"].nunique()),
        "summary": summary.to_dict(orient="records"),
        "agreement": agreement,
        "note": ("Descriptive expert study. Ties, both-incorrect and cannot-judge are "
                 "reported separately; two experts are not a population sample."),
    }
    (out_dir / "study_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_dir} ({len(df)} ratings, {df['item_id'].nunique()} items)")


if __name__ == "__main__":
    main()
