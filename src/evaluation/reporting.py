"""
Statistical reporting: bootstrap CIs, Wilcoxon, effect sizes, method comparison tables.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


def bootstrap_ci(data: np.ndarray, n_resamples: int = 10000, alpha: float = 0.05) -> dict:
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)
    rng = np.random.default_rng(42)
    means = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)])
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "ci_lower": float(np.percentile(means, lo_pct)),
        "ci_upper": float(np.percentile(means, hi_pct)),
        "significant": float(np.percentile(means, lo_pct)) * float(np.percentile(means, hi_pct)) > 0,
    }


def wilcoxon_paired(delta: np.ndarray) -> dict:
    stat, p = stats.wilcoxon(delta) if len(delta) > 0 else (np.nan, np.nan)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant_005": bool(p < 0.05),
        "significant_001": bool(p < 0.01),
    }


def cohens_d(delta: np.ndarray) -> float:
    if len(delta) == 0:
        return 0.0
    return float(np.mean(delta) / max(np.std(delta), 1e-8))


def load_results_json(path: Path) -> list[dict]:
    return json.loads(path.read_text("utf-8"))


def method_comparison_table(results_dir: Path, method_names: list[str]) -> pd.DataFrame:
    rows = []
    for mn in method_names:
        details_files = sorted(results_dir.glob(f"*{mn}*_details.json"))
        if not details_files:
            continue
        for df in details_files:
            data = load_results_json(df)
            deltas = np.array([r["deltas"]["delta_cosine"] for r in data if "deltas" in r])
            if len(deltas) == 0:
                continue
            ci = bootstrap_ci(deltas, n_resamples=5000)
            wc = wilcoxon_paired(deltas)
            rows.append({
                "method": mn,
                "file": df.name,
                "n": len(deltas),
                **ci,
                "wilcoxon_p": wc["p_value"],
                "wilcoxon_sig": wc["significant_005"],
                "cohens_d": cohens_d(deltas),
            })
    return pd.DataFrame(rows)


def oracle_decile_analysis(results: list[dict]) -> pd.DataFrame:
    rows = sorted(results, key=lambda r: r["baseline"]["metrics"]["cosine"])
    n = len(rows)
    decile_size = max(1, n // 10)
    deciles = []
    for d in range(10):
        start = d * decile_size
        end = start + decile_size if d < 9 else n
        subset = rows[start:end]
        bl_mean = np.mean([r["baseline"]["metrics"]["cosine"] for r in subset])
        delta_mean = np.mean([r["deltas"]["delta_cosine"] for r in subset])
        pct_improved = np.mean([r["deltas"]["delta_cosine"] > 0 for r in subset])
        deciles.append({
            "decile": d + 1,
            "n": len(subset),
            "baseline_cosine_mean": float(bl_mean),
            "delta_mean": float(delta_mean),
            "pct_improved": float(pct_improved),
        })
    return pd.DataFrame(deciles)


def gate_cutoff_sweep(
    results: list[dict],
    signal_key: str = "top1_faiss",
    cutoffs: Optional[list[float]] = None,
) -> pd.DataFrame:
    if cutoffs is None:
        cutoffs = list(np.linspace(0.5, 1.0, 11))
    rows = []
    for cutoff in cutoffs:
        gated_deltas = []
        for r in results:
            signal = r.get("pool_features", {}).get(signal_key, 1.0)
            if signal < cutoff:
                gated_deltas.append(r["deltas"]["delta_cosine"])
            else:
                gated_deltas.append(0.0)
        rows.append({
            "cutoff": cutoff,
            "mean_delta": float(np.mean(gated_deltas)),
            "pct_open": float(np.mean([r.get("pool_features", {}).get(signal_key, 1.0) < cutoff for r in results])),
        })
    return pd.DataFrame(rows)


def oracle_ceiling_analysis(results: list[dict], always_on_delta: float) -> pd.DataFrame:
    deltas = np.array([r["deltas"]["delta_cosine"] for r in results if "deltas" in r])
    oracle_delta = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
    return pd.DataFrame([{
        "always_on_mean_delta": always_on_delta,
        "oracle_positive_mean": oracle_delta,
        "fraction_captured": always_on_delta / oracle_delta if oracle_delta > 0 else 0.0,
        "n_rescues": int(np.sum(deltas > 0)),
        "n_disruptions": int(np.sum(deltas < 0)),
    }])