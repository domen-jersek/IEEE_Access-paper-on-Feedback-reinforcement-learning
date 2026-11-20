#!/usr/bin/env python3
"""32_public_study_manifest.py — sanitized, shareable study provenance.

Reads a private study package manifest and writes a version with no absolute
paths, item IDs or ticket IDs. The frozen package itself stays private because
items.json contains ticket text.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.artifacts import write_once_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    public = {
        "schema_version": manifest["schema_version"],
        "study_version": manifest["study_version"],
        "arms": manifest["arms"],
        "created_utc": manifest["created_utc"],
        "assignment_seed": manifest["assignment_seed"],
        "n_items": manifest["n_items"],
        "n_answer_items": manifest["n_answer_items"],
        "n_retrieval_items": manifest["n_retrieval_items"],
        "sample": manifest["sample"],
        "dataset": manifest["dataset"],
        "configs": manifest["configs"],
        "git": manifest.get("git"),
        "notes": manifest.get("notes", ""),
        "freeze_rule": manifest["freeze_rule"],
        "runs": {
            label: {
                "run_dir": Path(entry["run_dir"]).name,
                "details": entry["details"],
                "details_sha256": entry["details_sha256"],
                "config_hash": entry.get("config_hash"),
                "n_records": entry.get("n_records"),
                "generation_regime_id": (entry.get("generation_regime") or {}).get("regime_id"),
            }
            for label, entry in manifest["runs"].items()
        },
        "note": ("Sanitized provenance record. The package itself contains confidential "
                 "ticket text and is not distributed with the repository."),
    }
    write_once_json(args.out, public)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
