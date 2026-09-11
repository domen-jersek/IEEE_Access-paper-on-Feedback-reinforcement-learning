#!/usr/bin/env python3
"""
05_gate_cv.py — Phase 4
=========================
Extracts gate features from eval results and runs nested cross-validation.
Uses train-set evaluation results to train/validate the learned gate.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from experiments.utils import setup_logging, load_dataset, get_arg_parser
from src.gate.features import extract_feature_matrix
from src.gate.model import train_evaluate_gate
from src.feedback.loader import load_feedback_as_scores


def main() -> None:
    parser = get_arg_parser("Gate CV training")
    parser.add_argument("--details-json", type=str, required=True,
                        help="Path to *_details.json from 04_evaluate.py (train split)")
    parser.add_argument("--feedback-protocol", type=str, default="conditioned")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    log = logging.getLogger("gate_cv")

    paths = ProjectPaths()
    details_path = Path(args.details_json)
    if not details_path.exists():
        log.error("Details JSON not found: %s", details_path)
        return

    log.info("=== Loading evaluation results ===")
    results = json.loads(details_path.read_text("utf-8"))
    log.info("Loaded %d ticket results", len(results))

    log.info("=== Loading feedback scores ===")
    fb_db_path = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    feedback_scores = load_feedback_as_scores(fb_db_path) if fb_db_path.exists() else {}

    log.info("=== Loading dataset for team/class lists ===")
    df = load_dataset()
    teams = sorted(df["Team->Name"].unique())
    classes = sorted(df["intent_class"].unique())

    log.info("=== Extracting feature matrix ===")
    X, y, feature_names = extract_feature_matrix(results, feedback_scores, teams, classes)
    log.info("Feature matrix: %s, class balance=%.2f", X.shape, float(np.mean(y)))

    if len(X) < 50:
        log.error("Too few samples (%d) for CV. Need at least 50 for meaningful results.", len(X))
        return

    log.info("=== Training gate with nested CV ===")
    output_dir = paths.results / "gate"
    results_cv = train_evaluate_gate(X, y, feature_names, output_dir)

    log.info("=== DONE ===")
    for k, v in results_cv.get("aggregate", {}).items():
        log.info("  %s: %s", k, v)


if __name__ == "__main__":
    main()