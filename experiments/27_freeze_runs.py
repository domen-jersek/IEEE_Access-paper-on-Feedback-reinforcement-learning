#!/usr/bin/env python3
"""27_freeze_runs.py — seal the generated revision runs before the human study.

Writes a write-once manifest with the hash of every details, summary and run
manifest file, the pinned dataset/config hashes and the generation regime. A
``--verify`` pass later re-hashes the files and fails if anything changed.

The manifest is a provenance record, not a copy of the answers. Answers and ticket
text stay in the private run directories.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, setup_logging
from src.config import ProjectPaths
from src.data.artifacts import content_id, write_once_json

RUN_PATTERN = "M4_intersection_eval_{source}_rev_{panel}"


def _latest(directory: Path, suffix: str) -> Path:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No {suffix} in {directory}")
    return files[-1]


def collect(paths: ProjectPaths, panels: list[str], sources: list[str]) -> dict:
    runs = {}
    for panel in panels:
        for source in sources:
            run_dir = paths.results / RUN_PATTERN.format(source=source, panel=panel)
            details = _latest(run_dir, "_details.json")
            summary_path = _latest(run_dir, "_summary.json")
            manifest_path = run_dir / "manifest_latest.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("total_valid") != summary.get("total_queries"):
                raise SystemExit(
                    f"Incomplete run {run_dir.name}: "
                    f"{summary.get('total_valid')}/{summary.get('total_queries')} valid"
                )
            manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                        if manifest_path.exists() else {})
            extra = manifest.get("extra", {})
            runs[f"{panel}/{source}"] = {
                "run_dir": str(run_dir),
                "details": details.name,
                "details_sha256": file_sha256(details),
                "summary": summary_path.name,
                "summary_sha256": file_sha256(summary_path),
                "manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else None,
                "n_records": summary.get("total_queries"),
                "config_hash": summary.get("config_hash"),
                "generation_regime": summary.get("generation_regime"),
                "context_id": extra.get("context_id"),
                "history_id": extra.get("history_id"),
                "feedback_prior_global": extra.get("feedback_prior_global"),
            }
    regimes = {str(entry["generation_regime"].get("regime_id"))
               if entry["generation_regime"] else None for entry in runs.values()}
    if len(regimes) != 1:
        raise SystemExit(f"Runs span multiple generation regimes: {sorted(map(str, regimes))}")
    return runs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v1")
    parser.add_argument("--panels", nargs="+", default=[
        "legacy_random", "disjoint_duplicate", "disjoint_prefix"])
    parser.add_argument("--sources", nargs="+", default=[
        "conditioned", "blind", "reply_embed_bge"])
    parser.add_argument("--verify", type=Path, default=None,
                        help="Verify an existing frozen manifest instead of creating one")
    parser.add_argument("--public", type=Path, default=None,
                        help="Also write a sanitized provenance record (no absolute paths)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    if args.verify is not None:
        frozen = json.loads(args.verify.read_text(encoding="utf-8"))
        failures = []
        for name, entry in frozen["runs"].items():
            for key, suffix in (("details_sha256", "details"), ("summary_sha256", "summary")):
                path = Path(entry["run_dir"]) / entry[suffix]
                if file_sha256(path) != entry[key]:
                    failures.append(f"{name}/{suffix}")
        if failures:
            raise SystemExit(f"Frozen artifacts changed: {failures}")
        print(f"verified {len(frozen['runs'])} runs; payload {frozen['frozen_id']}")
        return

    runs = collect(paths, args.panels, args.sources)
    payload = {
        "schema_version": 1,
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"sha256": file_sha256(paths.dataset_parquet)},
        "finalist": {"path": "configs/revision/finalists_v1.yaml",
                     "sha256": file_sha256(paths.root / "configs/revision/finalists_v1.yaml")},
        "generation_regime_id": next(iter(runs.values()))["generation_regime"]["regime_id"],
        "runs": runs,
        "git": git_info(),
        "freeze_rule": (
            "These generated artifacts are the source of the human study. Any change to "
            "retrieval, answers, evidence or configuration requires a new label and a new study."
        ),
    }
    payload["frozen_id"] = content_id({k: v for k, v in payload.items() if k != "created_utc"})
    out_dir = paths.results / "revision" / "frozen"
    out_path = out_dir / f"frozen_runs_{args.label}.json"
    if out_path.exists():
        # Idempotent: reuse the sealed record, but verify that nothing changed.
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        failures = [name for name, entry in existing["runs"].items()
                    if file_sha256(Path(entry["run_dir"]) / entry["details"]) != entry["details_sha256"]
                    or file_sha256(Path(entry["run_dir"]) / entry["summary"]) != entry["summary_sha256"]]
        if failures:
            raise SystemExit(f"Frozen artifacts changed: {failures}")
        payload = existing
        print(f"existing manifest verified: {out_path}")
    else:
        write_once_json(out_path, payload)
        print(f"frozen {len(runs)} runs -> {out_path}")
    if args.public is not None:
        public = {
            "schema_version": payload["schema_version"],
            "label": payload["label"],
            "created_utc": payload["created_utc"],
            "dataset": payload["dataset"],
            "finalist": payload["finalist"],
            "generation_regime_id": payload["generation_regime_id"],
            "frozen_id": payload["frozen_id"],
            "freeze_rule": payload["freeze_rule"],
            "runs": {
                name: {
                    "run_dir": Path(entry["run_dir"]).name,
                    "details": entry["details"],
                    "details_sha256": entry["details_sha256"],
                    "summary": entry["summary"],
                    "summary_sha256": entry["summary_sha256"],
                    "n_records": entry["n_records"],
                    "config_hash": entry["config_hash"],
                    "context_id": entry["context_id"],
                    "history_id": entry["history_id"],
                }
                for name, entry in runs.items()
            },
        }
        write_once_json(args.public, public)
        print(f"public provenance -> {args.public}")
    print(f"frozen {len(runs)} runs -> {out_path}")
    print(f"frozen_id={payload['frozen_id']}")


if __name__ == "__main__":
    main()
