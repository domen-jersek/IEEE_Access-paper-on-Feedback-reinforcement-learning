#!/usr/bin/env python3
"""Validate a revision data contract locally; never import models or call APIs.

This is a data-boundary preflight, not an experiment runner. Without --output it
only reads inputs and prints aggregate information. Full local manifests include
IDs and paths; keep them private and outside public documentation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths
from src.data.artifacts import write_once_json
from src.data.protocol import load_data_protocol, resolve_data_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ProjectPaths().root,
                        help="Project root for all relative input paths")
    parser.add_argument("--output", type=Path,
                        help="Optional NEW local manifest path; existing files are never overwritten")
    args = parser.parse_args()
    config_path = args.data_protocol
    if not config_path.is_absolute():
        config_path = args.root / config_path
    config = load_data_protocol(config_path)
    context = resolve_data_context(config, root=args.root)
    manifest = context.to_manifest()
    print(json.dumps({k: manifest[k] for k in (
        "status", "context_id", "history_id", "partition_counts",
        "group_rule_id", "group_semantic_status",
    )}, indent=2))
    print(f"Scenario: {config.scenario}; queries: {config.query_partition}; metadata: {config.metadata_policy}")
    print("No feedback/index/model/generation validation performed yet; this is package A only.")
    if args.output is not None:
        output = args.output if args.output.is_absolute() else args.root / args.output
        write_once_json(output, manifest)
        print(f"Local data-context manifest: {output}")


if __name__ == "__main__":
    main()
