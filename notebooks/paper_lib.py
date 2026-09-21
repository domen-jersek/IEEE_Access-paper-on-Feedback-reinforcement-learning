"""Shared, read-only analysis helpers for the IEEE Access paper notebooks."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGDIR = ROOT / "notebooks" / "figures"
FIGURE_INDEX = FIGDIR / "FIGURES.md"

ROUTING_PALETTE = {
    "M1_global": "#777777", "M2_team": "#2a6fbb", "M3_class": "#d17a22",
    "M4_intersection": "#16856b", "M5_backoff": "#8b4fb3", "M5_backoff2": "#8b4fb3",
}
RETRIEVER_PALETTE = {
    "bm25": "#d17a22", "dense_minilm": "#2a6fbb", "dense_bge": "#16856b",
    "hybrid_rrf": "#8b4fb3", "hybrid_bge_rrf": "#bd3f5f", "ce_rerank": "#555555",
}
LIFT_PALETTE = {
    "laplace": "#777777", "laplace_eb": "#16856b", "laplace_eb_k2": "#16856b",
    "laplace_eb_k10": "#8b4fb3", "none": "#222222",
}

matplotlib.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})
sns.set_context("talk", font_scale=0.90)
log = logging.getLogger("paper_lib")
_registry: Optional[pd.DataFrame] = None


class ArtifactMissingError(FileNotFoundError):
    """Raised when a notebook dependency has not been produced yet."""


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise ArtifactMissingError(f"Missing artifact: {path}\nRun `{command}` first.")
    return path


def _csv(path: Path, command: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(_require(path, command), **kwargs)


def _json(path: Path, command: str) -> Any:
    return json.loads(_require(path, command).read_text(encoding="utf-8"))


def optional(loader: Callable, *args, **kwargs):
    """Return ``None`` for a pending artifact while printing its exact run command."""
    try:
        return loader(*args, **kwargs)
    except ArtifactMissingError as exc:
        print(f"PENDING: {exc}")
        return None


def load_dataset(columns: Optional[list[str]] = None) -> pd.DataFrame:
    path = _require(DATA / "dataset.parquet", "python experiments/00_canonicalize.py")
    return pd.read_parquet(path, columns=columns)


def load_splits(seed: int = 42, regime: str = "random") -> dict:
    suffix = "" if regime == "random" else f"_{regime}"
    return _json(DATA / "splits" / f"split_seed{seed}{suffix}.json",
                 f"python experiments/01_split.py --seed {seed} --regime {regime}")


def load_split_report(seed: int = 42, regime: str = "random") -> pd.DataFrame:
    suffix = "" if regime == "random" else f"_{regime}"
    return _csv(DATA / "splits" / f"split_seed{seed}{suffix}_report.csv",
                f"python experiments/01_split.py --seed {seed} --regime {regime}")


def registry(refresh: bool = False) -> pd.DataFrame:
    global _registry
    if _registry is None or refresh:
        path = RESULTS / "registry.csv"
        _registry = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not _registry.empty and "timestamp_utc" in _registry:
            _registry = _registry.sort_values("timestamp_utc")
    return _registry.copy()


def latest_runs(all: bool = False) -> pd.DataFrame:
    """Return latest non-smoke run per script/output directory, or the full registry."""
    runs = registry()
    if runs.empty:
        return runs
    if not all:
        smoke = runs["phase"].fillna("").str.contains("smoke", case=False)
        smoke |= runs["run_id"].fillna("").str.contains("smoke", case=False)
        runs = runs.loc[~smoke]
        keys = [column for column in ("script", "out_dir") if column in runs]
        runs = runs.drop_duplicates(keys, keep="last")
    return runs.sort_values("timestamp_utc", ascending=False).reset_index(drop=True)


def run_stamp(run_id: str) -> str:
    runs = registry()
    match = runs[runs["run_id"] == run_id] if not runs.empty else pd.DataFrame()
    if match.empty:
        return run_id
    row = match.iloc[-1]
    sha = str(row.get("git_sha", ""))[:8] or "unknown"
    dirty = " dirty" if str(row.get("git_dirty", "")).lower() == "true" else ""
    return f"{run_id} ({sha}{dirty})"


def provenance_table(all: bool = False) -> pd.DataFrame:
    runs = latest_runs(all=all)
    columns = ["run_id", "phase", "script", "headline_metric", "headline_value", "git_sha", "git_dirty"]
    return runs[[column for column in columns if column in runs]].reset_index(drop=True)


def artifact_status() -> pd.DataFrame:
    checks = [
        ("Validity: proxy", RESULTS / "proxy_validation" / "report.json", "07_validate_proxy.py"),
        ("Validity: calibration", RESULTS / "feedback_calibration" / "report.json", "10_feedback_calibration.py"),
        ("Validity: rescoring", RESULTS / "rescored" / "method_comparison_v2.csv", "08_rescore.py"),
        ("Granularity: ladder", RESULTS / "retriever_ladder" / "grid.csv", "11_retriever_ladder.py"),
        ("Granularity: blind ladder", RESULTS / "retriever_ladder_blind" / "grid.csv", "11_retriever_ladder.py --feedback-protocol blind --tag blind"),
        ("Granularity: blend", RESULTS / "blend_eb" / "dev_eval.csv", "12_learn_blend.py --lift laplace_eb --tag eb"),
        ("Granularity: blind blend", RESULTS / "blend_eb_blind" / "dev_eval.csv", "12_learn_blend.py --lift laplace_eb --feedback-protocol blind --tag eb_blind"),
        ("Gate pilot: blind", RESULTS / "gate_pilot_blind" / "gate_pilot.csv", "16_gate_pilot.py --feedback-protocol blind --tag blind"),
        ("Volume", RESULTS / "feedback_volume" / "summary.csv", "13_feedback_volume_curve.py"),
        ("Held-out ladder", RESULTS / "retriever_ladder_eval" / "grid.csv", "11_retriever_ladder.py --split eval --tag eval"),
        ("Disjoint ladder", RESULTS / "retriever_ladder_disjoint" / "grid.csv", "11_retriever_ladder.py --split eval --regime disjoint --tag disjoint"),
        ("Gate", RESULTS / "gate_eb_m4" / "gate_cv_results.json", "05_gate_cv.py ... --tag eb_m4"),
        ("Pairwise judge", RESULTS / "answer_judge" / "summary.json", "09_llm_judge_pairwise.py ..."),
    ]
    return pd.DataFrame([{"section": name, "available": path.exists(), "path": str(path.relative_to(ROOT)), "producer": script}
                         for name, path, script in checks])


def load_sikdd_replication() -> pd.DataFrame:
    return _csv(RESULTS / "sikdd_replication.csv", "python experiments/06_report.py")


def load_method_comparison() -> pd.DataFrame:
    return _csv(RESULTS / "report" / "method_comparison.csv", "python experiments/06_report.py")


def load_gate_sweep(run: str) -> pd.DataFrame:
    """Post-hoc static-gate cutoff sweep for a generated run (top1_faiss rule)."""
    return _csv(RESULTS / "report" / f"gate_sweep_{run}.csv", "python experiments/06_report.py")


def load_oracle_deciles(run: str) -> pd.DataFrame:
    """Per-decile baseline difficulty and delta for a generated run."""
    return _csv(RESULTS / "report" / f"oracle_deciles_{run}.csv", "python experiments/06_report.py")


def load_run_summaries() -> pd.DataFrame:
    """Assemble every generated dev run's summary into one conditioned-vs-blind frame."""
    rows = []
    for path in RESULTS.glob("*_dev_*/*_summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        cfg = summary.get("config", {})
        name = path.parent.name
        rows.append({
            "run": name,
            "method": name.split("_dev_")[0],
            "protocol": "blind" if "blind" in name else ("conditioned" if "conditioned" in name else "?"),
            "agg": "binary" if "binary" in name else "continuous",
            "lift": cfg.get("lift", {}).get("name"),
            "mean_delta_cosine": metrics.get("mean_delta_cosine"),
            "median_delta_cosine": metrics.get("median_delta_cosine"),
            "pct_improved": metrics.get("pct_improved"),
            "pct_worsened": metrics.get("pct_worsened"),
            "n": summary.get("total_valid"),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["method", "protocol"]).reset_index(drop=True) if len(frame) else frame


def load_identical_prompt_audit() -> pd.DataFrame:
    return _csv(RESULTS / "identical_prompt_audit.csv", "python experiments/14_audit_identical_prompts.py")


def load_proxy_report() -> dict:
    return _json(RESULTS / "proxy_validation" / "report.json", "python experiments/07_validate_proxy.py")


def load_proxy_per_ticket() -> pd.DataFrame:
    frame = _csv(RESULTS / "proxy_validation" / "per_ticket.csv", "python experiments/07_validate_proxy.py")
    return frame[~frame["run"].fillna("").str.contains("smoke", case=False)].copy()


def load_calib_report() -> dict:
    return _json(RESULTS / "feedback_calibration" / "report.json", "python experiments/10_feedback_calibration.py")


def load_calib_saturation() -> pd.DataFrame:
    return _csv(RESULTS / "feedback_calibration" / "saturation.csv", "python experiments/10_feedback_calibration.py")


def load_calib_reliability(protocol: str = "conditioned") -> pd.DataFrame:
    return _csv(RESULTS / "feedback_calibration" / f"reliability_{protocol}.csv", "python experiments/10_feedback_calibration.py")


def load_calib_scope_priors(protocol: str = "conditioned") -> pd.DataFrame:
    return _csv(RESULTS / "feedback_calibration" / f"scope_priors_{protocol}.csv", "python experiments/10_feedback_calibration.py")


def load_rescore_comparison() -> pd.DataFrame:
    frame = _csv(RESULTS / "rescored" / "method_comparison_v2.csv", "python experiments/08_rescore.py --force")
    return frame[~frame["run"].fillna("").str.contains("smoke", case=False)].copy()


def _ladder_dir(tag: str = "") -> Path:
    return RESULTS / ("retriever_ladder" + (f"_{tag}" if tag else ""))


def _ladder_command(tag: str) -> str:
    if tag == "eval":
        return "python experiments/11_retriever_ladder.py --split eval --retrievers dense_minilm bm25 hybrid_rrf --tag eval"
    if tag == "disjoint":
        return "python experiments/11_retriever_ladder.py --split eval --regime disjoint --retrievers dense_minilm bm25 hybrid_rrf --tag disjoint"
    return "python experiments/11_retriever_ladder.py"


def load_ladder_grid(tag: str = "") -> pd.DataFrame:
    return _csv(_ladder_dir(tag) / "grid.csv", _ladder_command(tag))


def load_ladder_curve(tag: str = "") -> pd.DataFrame:
    return _csv(_ladder_dir(tag) / "ladder_curve.csv", _ladder_command(tag))


def load_ladder_per_ticket(tag: str = "") -> pd.DataFrame:
    path = _require(_ladder_dir(tag) / "per_ticket.parquet", _ladder_command(tag))
    return pd.read_parquet(path)


def load_blend_dev(tag: str = "blend_eb") -> pd.DataFrame:
    return _csv(RESULTS / tag / "dev_eval.csv", "python experiments/12_learn_blend.py --lift laplace_eb --tag eb")


def load_blend_grid(tag: str = "blend_eb") -> pd.DataFrame:
    return _csv(RESULTS / tag / "grid_train.csv", "python experiments/12_learn_blend.py --lift laplace_eb --tag eb")


def load_blend_bootstrap(tag: str = "blend_eb") -> pd.DataFrame:
    return _csv(RESULTS / tag / "bootstrap_weights.csv", "python experiments/12_learn_blend.py --lift laplace_eb --tag eb")


def load_blend_weights(tag: str = "blend_eb") -> dict:
    return _json(RESULTS / tag / "learned_weights.json", "python experiments/12_learn_blend.py --lift laplace_eb --tag eb")


def load_volume_curve(tag: str = "") -> pd.DataFrame:
    folder = RESULTS / ("feedback_volume" + (f"_{tag}" if tag else ""))
    return _csv(folder / "curve.csv", "python experiments/13_feedback_volume_curve.py")


def load_volume_per_seed(tag: str = "") -> pd.DataFrame:
    folder = RESULTS / ("feedback_volume" + (f"_{tag}" if tag else ""))
    return _csv(folder / "per_seed.csv", "python experiments/13_feedback_volume_curve.py")


def load_volume_summary(tag: str = "") -> pd.DataFrame:
    folder = RESULTS / ("feedback_volume" + (f"_{tag}" if tag else ""))
    return _csv(folder / "summary.csv", "python experiments/13_feedback_volume_curve.py")


def load_gate_cv_results(tag: str = "gate_eb_m4") -> dict:
    return _json(RESULTS / tag / "gate_cv_results.json", "python experiments/05_gate_cv.py ... --tag eb_m4")


def load_gate_pilot(tag: str = "blind") -> pd.DataFrame:
    return _csv(RESULTS / f"gate_pilot_{tag}" / "gate_pilot.csv",
                f"python experiments/16_gate_pilot.py --tag {tag} ...")


def load_gate_pilot_policy(tag: str = "blind") -> pd.DataFrame:
    return _csv(RESULTS / f"gate_pilot_{tag}" / "policy.csv",
                f"python experiments/16_gate_pilot.py --tag {tag} ...")


def load_gate_pilot_rules(tag: str = "blind") -> pd.DataFrame:
    return _csv(RESULTS / f"gate_pilot_{tag}" / "rules.csv",
                f"python experiments/16_gate_pilot.py --tag {tag} ...")


def load_gate_dev_eval(tag: str = "gate_eb_m4") -> dict:
    return _json(RESULTS / tag / "dev_eval.json", "python experiments/05_gate_cv.py ... --tag eb_m4")


def load_gate_policy(tag: str = "gate_eb_m4") -> pd.DataFrame:
    return _csv(RESULTS / tag / "dev_policy_value.csv", "python experiments/05_gate_cv.py ... --tag eb_m4")


def _newest(folder: str, suffix: str) -> Path:
    files = sorted((RESULTS / folder).glob(f"*{suffix}"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise ArtifactMissingError(f"No `{suffix}` artifact in {RESULTS / folder}.\nRun `python experiments/04_evaluate.py ...` first.")
    return files[-1]


def load_details(folder: str) -> list[dict]:
    return json.loads(_newest(folder, "_details.json").read_text(encoding="utf-8"))


def load_summary(folder: str) -> dict:
    return json.loads(_newest(folder, "_summary.json").read_text(encoding="utf-8"))


def load_rescored_per_ticket(folder: str) -> pd.DataFrame:
    return pd.read_csv(_newest(folder, "_details_rescored.csv"))


def load_answer_judge_summary() -> dict:
    return _json(RESULTS / "answer_judge" / "summary.json", "python experiments/09_llm_judge_pairwise.py ...")


def load_answer_judge_scores() -> pd.DataFrame:
    return _csv(RESULTS / "answer_judge" / "scores.csv", "python experiments/09_llm_judge_pairwise.py ...")


def bootstrap_ci(values: Iterable[float], confidence: float = 0.95, n_boot: int = 10_000,
                 seed: int = 42) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(np.quantile(means, [alpha, 1.0 - alpha]))


def wilcoxon_p(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values).pvalue)


def paired_sign_agreement(a: Iterable[float], b: Iterable[float], ignore_zeros: bool = True) -> float:
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if ignore_zeros:
        mask &= (a != 0.0) & (b != 0.0)
    return float((np.sign(a[mask]) == np.sign(b[mask])).mean()) if mask.any() else np.nan


def savefig(name: str, run_ids: Iterable[str] = (), dpi: int = 180) -> Path:
    """Save an IEEE figure and append its run provenance to FIGURES.md."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"ieee_{name}.png"
    plt.savefig(path, dpi=dpi)
    header = "# IEEE paper figures\n\n| Figure | Producing run(s) |\n|---|---|\n"
    if not FIGURE_INDEX.exists():
        FIGURE_INDEX.write_text(header, encoding="utf-8")
    line = f"| `{path.name}` | {', '.join(run_stamp(run) for run in run_ids) or 'derived from cited artifacts'} |\n"
    existing = FIGURE_INDEX.read_text(encoding="utf-8")
    filtered = "\n".join(row for row in existing.splitlines() if f"`{path.name}`" not in row) + "\n"
    FIGURE_INDEX.write_text(filtered + line, encoding="utf-8")
    return path


def sidebar(figsize: tuple[float, float] = (8, 5)):
    return plt.subplots(figsize=figsize)
