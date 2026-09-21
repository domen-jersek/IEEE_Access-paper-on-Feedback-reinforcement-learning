#!/usr/bin/env python3
"""Audit completed generation runs for artificial deltas on identical prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit_file(path: Path) -> dict:
    records = json.loads(path.read_text(encoding="utf-8"))
    raw, corrected, identical = [], [], []
    for record in records:
        delta = float(record.get("deltas", {}).get("delta_cosine", 0.0))
        hashes = record.get("prompt_sha256", {})
        if hashes:
            same = hashes.get("baseline") == hashes.get("feedback")
        else:
            baseline_ids = [row.get("seq_id", row.get("retrieved_id")) for row in record.get("baseline", {}).get("retrieval", [])]
            feedback_ids = [row.get("seq_id", row.get("retrieved_id")) for row in record.get("feedback", {}).get("retrieval", [])]
            same = bool(baseline_ids) and baseline_ids == feedback_ids
        raw.append(delta)
        corrected.append(0.0 if same else delta)
        identical.append(same)
    return {
        "run": path.parent.name,
        "details": str(path.relative_to(ROOT)),
        "n": len(records),
        "n_identical_prompts": int(sum(identical)),
        "pct_identical_prompts": float(np.mean(identical)) if records else np.nan,
        "raw_mean_delta_cosine": float(np.mean(raw)) if records else np.nan,
        "corrected_mean_delta_cosine": float(np.mean(corrected)) if records else np.nan,
        "correction": float(np.mean(corrected) - np.mean(raw)) if records else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="*", default=["*_dev_*"], help="Result-folder glob patterns")
    parser.add_argument("--output", default="results/identical_prompt_audit.csv")
    args = parser.parse_args()
    paths = []
    for pattern in args.folders:
        for folder in (ROOT / "results").glob(pattern):
            paths.extend(folder.glob("*_details.json"))
    newest_by_folder = {}
    for path in paths:
        if path.parent not in newest_by_folder or path.stat().st_mtime > newest_by_folder[path.parent].stat().st_mtime:
            newest_by_folder[path.parent] = path
    if not newest_by_folder:
        raise SystemExit("No details files matched --folders")
    report = pd.DataFrame(audit_file(path) for path in newest_by_folder.values()).sort_values("run")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
