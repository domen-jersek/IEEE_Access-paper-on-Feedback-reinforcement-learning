#!/usr/bin/env python3
"""
08_rescore.py — P1.2
=====================
Re-scores the STORED answers of completed evaluation runs with metrics from an
independent model family (no generation, no API cost):

    cosine_bge     cosine under BAAI/bge-base-en-v1.5  (retriever + SIKDD metric are MiniLM)
    bertscore_f1   BERTScore F1 (roberta-large, baseline-rescaled)   [--no-bertscore to skip]

Writes, per run folder:  <details-stem>_rescored.csv  (one row per ticket with all
metrics/deltas) and updates results/rescored/method_comparison_v2.csv with mean
deltas, Wilcoxon p-values and bootstrap CIs under every metric.

Answers the question: do the M1..M4 conclusions survive a non-MiniLM metric?
"""
from __future__ import annotations

import glob
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.utils import (setup_logging, get_arg_parser, write_run_manifest, append_registry, make_run_id)
from src.config import ProjectPaths
from src.evaluation.metrics import batch_cosine, batch_bertscore
from src.evaluation.reporting import bootstrap_ci, wilcoxon_paired


def newest_details(folder: Path) -> Path | None:
    files = sorted(folder.glob("*_details.json"))
    return files[-1] if files else None


def rescore_run(details: Path, do_bertscore: bool, log) -> pd.DataFrame:
    recs = json.loads(details.read_text("utf-8"))
    refs = [r["reference_reply"] for r in recs]
    bl = [r["baseline"]["answer"] for r in recs]
    fb = [r["feedback"]["answer"] for r in recs]
    df = pd.DataFrame({
        "run": details.parent.name,
        "ticket_id": [r["ticket_id"] for r in recs],
        "expected_team": [r["expected_team"] for r in recs],
        "expected_class": [r["expected_class"] for r in recs],
        "baseline_cosine": [r["baseline"]["metrics"]["cosine"] for r in recs],
        "feedback_cosine": [r["feedback"]["metrics"]["cosine"] for r in recs],
        "delta_cosine": [r["deltas"]["delta_cosine"] for r in recs],
        "delta_rouge_l": [r["deltas"].get("delta_rouge_l", np.nan) for r in recs],
        "top1_changed": [float(r["baseline"]["retrieval"][0]["seq_id"] != r["feedback"]["retrieval"][0]["seq_id"]) for r in recs],
    })
    log.info("  cosine_bge ...")
    df["baseline_cosine_bge"] = batch_cosine(bl, refs, model="bge")
    df["feedback_cosine_bge"] = batch_cosine(fb, refs, model="bge")
    df["delta_cosine_bge"] = df["feedback_cosine_bge"] - df["baseline_cosine_bge"]
    # sanity: recompute minilm cosine in batch and compare to stored values
    df["baseline_cosine_recomputed"] = batch_cosine(bl, refs, model="minilm")
    if do_bertscore:
        log.info("  bertscore_f1 (roberta-large on CPU; a few minutes) ...")
        df["baseline_bertscore_f1"] = batch_bertscore(bl, refs)
        df["feedback_bertscore_f1"] = batch_bertscore(fb, refs)
        df["delta_bertscore_f1"] = df["feedback_bertscore_f1"] - df["baseline_bertscore_f1"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    out = {"run": df["run"].iloc[0], "n": int(len(df))}
    for m in ("delta_cosine", "delta_rouge_l", "delta_cosine_bge", "delta_bertscore_f1"):
        if m not in df or df[m].isna().all():
            continue
        d = df[m].to_numpy(float)
        ci = bootstrap_ci(d)
        w = wilcoxon_paired(d) if np.any(d != 0) else {"p_value": None}
        out[f"{m}_mean"] = float(d.mean())
        out[f"{m}_ci_lo"] = ci["ci_lower"]
        out[f"{m}_ci_hi"] = ci["ci_upper"]
        out[f"{m}_wilcoxon_p"] = w.get("p_value")
        out[f"{m}_pct_improved"] = float(np.mean(d > 0))
    out["max_abs_cosine_recompute_diff"] = float(np.abs(df["baseline_cosine_recomputed"] - df["baseline_cosine"]).max())
    return out


def main() -> None:
    parser = get_arg_parser("Re-score stored answers with independent metrics")
    parser.add_argument("--runs", nargs="*", default=["*_dev_*"], help="glob(s) of result folders under results/")
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute even if *_rescored.csv exists")
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("rescore")
    paths = ProjectPaths()
    out_dir = paths.results / "rescored"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("rescore")

    folders = []
    for pat in args.runs:
        folders += [Path(p) for p in glob.glob(str(paths.results / pat)) if Path(p).is_dir()]
    folders = sorted(set(folders))
    summaries, inputs = [], []
    for folder in folders:
        details = newest_details(folder)
        if details is None:
            continue
        out_csv = folder / (details.stem + "_rescored.csv")
        if out_csv.exists() and not args.force:
            log.info("%s: using cached %s", folder.name, out_csv.name)
            df = pd.read_csv(out_csv)
        else:
            log.info("%s: rescoring %d tickets", folder.name, len(json.loads(details.read_text('utf-8'))))
            df = rescore_run(details, not args.no_bertscore, log)
            df.to_csv(out_csv, index=False)
        inputs.append(details)
        summaries.append(summarize(df))

    comp = pd.DataFrame(summaries)
    comp_path = out_dir / "method_comparison_v2.csv"
    comp.to_csv(comp_path, index=False)
    manifest = write_run_manifest(out_dir, script="experiments/08_rescore.py", args=args, inputs=inputs,
                                  extra={"n_runs": len(summaries)}, run_id=run_id)
    append_registry(run_id=run_id, phase="P1.2-rescore", script="08_rescore.py", out_dir=out_dir, manifest=manifest,
                    headline_metric="n_runs", headline_value=len(summaries))

    cols = [c for c in comp.columns if c.endswith("_mean") or c.endswith("_wilcoxon_p") or c in ("run", "n")]
    pd.set_option("display.width", 200)
    print("\n=== RESCORED METHOD COMPARISON ===")
    print(comp[cols].to_string(index=False))
    print(f"\nmax |stored - recomputed minilm cosine| per run: {comp['max_abs_cosine_recompute_diff'].max():.2e}")
    print(f"outputs: {comp_path}")


if __name__ == "__main__":
    main()
