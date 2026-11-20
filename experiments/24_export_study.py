#!/usr/bin/env python3
"""24_export_study.py — build the frozen, blinded human-evaluation package.

Reads generated revision runs and writes a study package with two blocks per
ticket: answer comparison and retrieval comparison. Baseline and feedback are
randomised to A/B independently per block, balanced within each stratum. Method
identity is written only to ``key.json``; the application serves ``items.json``.

Two modes:
    --panel X --sources conditioned blind      one panel, several evidence sources
    --arm X:source:N --arm Y:source:M          several arms with per-arm sample sizes

Pre-freeze checks (fail closed):
* runs of the same panel cover exactly the same query set when compared together;
* baseline answers and baseline retrieval lists are identical across sources of a
  panel (same prompts and cache regime);
* no missing or empty answers;
* every item receives exactly one A/B assignment.

Once exported, the package is immutable: use a new ``--version`` for another one.
The human study must cite the manifest, not a later run.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, setup_logging
from src.config import ProjectPaths

BLOCKS = ("answer", "retrieval")


def _latest_details(run_dir: Path) -> Path:
    files = sorted(run_dir.glob("*_details.json"))
    if not files:
        raise FileNotFoundError(f"No details JSON in {run_dir}")
    return files[-1]


def _load_run(run_dir: Path) -> tuple[dict, Path, dict]:
    details_path = _latest_details(run_dir)
    records = json.loads(details_path.read_text(encoding="utf-8"))
    summaries = sorted(run_dir.glob("*_summary.json"))
    summary = json.loads(summaries[-1].read_text(encoding="utf-8")) if summaries else {}
    return {record["ticket_id"]: record for record in records}, details_path, summary


def _opaque_id(seed: str, arm: str, ticket_id: str) -> str:
    """Blinded item stem: reveals neither method, panel nor ticket ID."""
    return hashlib.sha256(f"{seed}|{arm}|{ticket_id}".encode()).hexdigest()[:12]


def _balanced_assignment(item_ids: list[str], seed: str) -> dict[str, str]:
    """Deterministic and exactly balanced: half A=baseline, half A=feedback."""
    ordered = sorted(item_ids, key=lambda item: hashlib.sha256(f"{seed}|{item}".encode()).hexdigest())
    half = (len(ordered) + 1) // 2
    return {item: ("baseline" if index < half else "feedback") for index, item in enumerate(ordered)}


def _baseline_inconsistencies(sources: dict[str, dict], label: str) -> set[str]:
    """Tickets whose baseline answer differs across sources of one panel.

    Baseline retrieval must be identical; a differing answer is hosted-model
    nondeterminism on a cache miss. Such tickets are excluded from the study frame
    because the A/B comparison would otherwise use different baselines across arms.
    """
    names = list(sources)
    reference = names[0]
    inconsistent = set()
    for ticket_id, record in sources[reference].items():
        retrieval = [c["seq_id"] for c in record["baseline"]["retrieval"]]
        for other in names[1:]:
            if ticket_id not in sources[other]:
                continue
            if [c["seq_id"] for c in sources[other][ticket_id]["baseline"]["retrieval"]] != retrieval:
                raise SystemExit(
                    f"Baseline retrieval differs between {reference} and {other} for {ticket_id} in {label}"
                )
            if sources[other][ticket_id]["baseline"]["answer"] != record["baseline"]["answer"]:
                inconsistent.add(ticket_id)
    return inconsistent


def _changed(record: dict, stratum: str) -> bool:
    if stratum == "answer":
        return record["baseline"]["answer"] != record["feedback"]["answer"]
    return (record["baseline"]["retrieval"][0]["seq_id"]
            != record["feedback"]["retrieval"][0]["seq_id"])


def _allocate(groups: dict[str, list[str]], n: int) -> dict[str, int]:
    """Proportional allocation with largest remainders, capped by group size."""
    total = sum(len(ids) for ids in groups.values())
    if n >= total:
        return {name: len(ids) for name, ids in groups.items()}
    quotas = {name: n * len(ids) / total for name, ids in groups.items()}
    allocation = {name: min(len(groups[name]), int(quotas[name])) for name in groups}
    remaining = n - sum(allocation.values())
    order = sorted(groups, key=lambda name: (-(quotas[name] - int(quotas[name])), name))
    while remaining > 0:
        progressed = False
        for name in order:
            if remaining == 0:
                break
            if allocation[name] < len(groups[name]):
                allocation[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


def _sample(records: dict, sources: dict, seed: str, n: int | None, stratum: str,
            changed_only: bool, balance_by: str, exclude: set[str]):
    """Deterministic sample from the eligible frame; never selects on outcome sign."""
    frame = sorted(set(records) - set(exclude))
    eligible = [t for t in frame if (not changed_only) or _changed(records[t], stratum)]
    changed_count = sum(1 for t in frame if _changed(records[t], stratum))
    stats = {
        "frame": len(frame),
        "excluded_inconsistent": len(exclude),
        "changed": changed_count,
        "changed_share": changed_count / len(frame) if frame else None,
        "eligible": len(eligible),
    }
    if n is None or n >= len(eligible):
        return eligible, stats, {"all": len(eligible)}

    if balance_by == "none":
        groups = {"all": eligible}
    else:
        key = f"expected_{balance_by}"
        groups = {}
        for ticket_id in eligible:
            groups.setdefault(str(records[ticket_id].get(key, "unknown")), []).append(ticket_id)
    allocation = _allocate(groups, n)
    selected = []
    for name, ids in groups.items():
        ordered = sorted(ids, key=lambda t: hashlib.sha256(
            f"{seed}|sample|{t}".encode()).hexdigest())
        selected.extend(ordered[: allocation[name]])
    stats["strata_sizes"] = {name: len(ids) for name, ids in groups.items()}
    return sorted(selected), stats, allocation


def _parse_arms(args) -> list[dict]:
    if args.arm:
        arms = []
        for spec in args.arm:
            parts = spec.split(":")
            if len(parts) != 3:
                raise SystemExit(f"--arm must be panel:source:sample, got {spec!r}")
            panel, source, sample = parts
            arms.append({"panel": panel, "source": source,
                         "sample": int(sample) if sample else None})
        return arms
    return [{"panel": args.panel, "source": source, "sample": args.sample}
            for source in args.sources]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=None)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--arm", action="append", default=[],
                        help="panel:source:sample; repeat for multiple arms")
    parser.add_argument("--version", required=True, help="Study version label, e.g. v1")
    parser.add_argument("--seed", default="20261112", help="A/B assignment seed")
    parser.add_argument("--sample", type=int, default=None,
                        help="Per-source sample size (single-panel mode)")
    parser.add_argument("--changed-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Sample only tickets whose answer changed (pre-registered default)")
    parser.add_argument("--balance-by", choices=["team", "class", "none"], default="team",
                        help="Proportional stratification of the sample")
    parser.add_argument("--blocks", nargs="+", choices=["answer", "retrieval"], default=["answer"],
                        help="Which item blocks to generate")
    parser.add_argument("--stratum", choices=["answer", "retrieval"], default="answer",
                        help="What counts as a changed ticket for eligibility")
    parser.add_argument("--run", action="append", default=[],
                        help="Override a run directory: source=relative/or/absolute/path")
    parser.add_argument("--out-root", default="data/processed/revision/study",
                        help="Private output root; do not point this at a tracked directory")
    parser.add_argument("--notes", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)
    if not args.arm and (not args.panel or not args.sources):
        raise SystemExit("Provide either --arm or --panel with --sources")

    paths = ProjectPaths()
    overrides = {}
    for value in args.run:
        if "=" not in value:
            raise SystemExit(f"--run must be source=path, got {value!r}")
        source, path = value.split("=", 1)
        overrides[source] = Path(path)

    arms = _parse_arms(args)
    loaded: dict[str, dict] = {}
    run_meta = {}
    for arm in arms:
        label = f"{arm['panel']}/{arm['source']}"
        run_dir = overrides.get(arm["source"],
                                paths.results / f"M4_intersection_eval_{arm['source']}_rev_{arm['panel']}")
        if not run_dir.is_absolute():
            run_dir = paths.root / run_dir
        records, details_path, summary = _load_run(run_dir)
        loaded[label] = records
        run_meta[label] = {
            "run_dir": str(run_dir), "details": details_path.name,
            "details_sha256": file_sha256(details_path),
            "config_hash": summary.get("config_hash"),
            "generation_regime": summary.get("generation_regime"),
            "n_records": len(records),
        }

    # Same-panel arms must share one query set and one baseline.
    by_panel: dict[str, dict[str, dict]] = {}
    for arm in arms:
        by_panel.setdefault(arm["panel"], {})[arm["source"]] = loaded[f"{arm['panel']}/{arm['source']}"]
    excluded: dict[str, set[str]] = {}
    for panel, sources in by_panel.items():
        sets = {source: set(records) for source, records in sources.items()}
        reference = set(next(iter(sets.values())))
        if any(value != reference for value in sets.values()):
            raise SystemExit(f"Runs of panel {panel} do not cover the same query set")
        excluded[panel] = _baseline_inconsistencies(sources, panel)

    items, key, sample_meta = [], {}, {}
    for arm in arms:
        label = f"{arm['panel']}/{arm['source']}"
        records = loaded[label]
        selected, stats, allocation = _sample(
            records, {arm["source"]: records}, args.seed, arm["sample"],
            args.stratum, args.changed_only, args.balance_by, excluded[arm["panel"]])
        sample_meta[label] = {"requested": arm["sample"], "n": len(selected),
                              "stratum": args.stratum,
                              "changed_only": args.changed_only,
                              "balance_by": args.balance_by,
                              **stats, "allocation": allocation}
        for ticket_id in selected:
            record = records[ticket_id]
            if not str(record["feedback"]["answer"]).strip() or not str(record["baseline"]["answer"]).strip():
                raise SystemExit(f"Empty answer for {ticket_id}/{arm['source']}")
            stem = _opaque_id(args.seed, label, ticket_id)
            for block in args.blocks:
                item_id = f"{stem}-{'A' if block == 'answer' else 'R'}"
                items.append({"item_id": item_id, "block": block})
                key[item_id] = {"panel": arm["panel"], "ticket_id": ticket_id,
                                "source": arm["source"], "block": block, "A": None, "B": None}

    for block in args.blocks:
        stratum_ids = [item["item_id"] for item in items if item["block"] == block]
        assignment = _balanced_assignment(stratum_ids, f"{args.seed}|{block}")
        for item_id, side in assignment.items():
            key[item_id]["A"] = side
            key[item_id]["B"] = "feedback" if side == "baseline" else "baseline"

    # Attach content; method identity stays in key.json only.
    content = {}
    for arm in arms:
        label = f"{arm['panel']}/{arm['source']}"
        content[label] = loaded[label]
    for item in items:
        hidden = key[item["item_id"]]
        record = content[f"{hidden['panel']}/{hidden['source']}"][hidden["ticket_id"]]
        item["query_title"] = record["query_title"]
        item["query_description"] = record.get("query_description", "")
        if item["block"] == "answer":
            sides = {"baseline": record["baseline"]["answer"],
                     "feedback": record["feedback"]["answer"]}
            item["A"], item["B"] = sides[hidden["A"]], sides[hidden["B"]]
        else:
            def candidates(side):
                return [{"title": c["title"], "reply": c["reply"]} for c in record[side]["retrieval"]]
            item["A"], item["B"] = candidates(hidden["A"]), candidates(hidden["B"])

    out_root = paths.root / args.out_root / args.version
    if out_root.exists():
        raise SystemExit(f"Study package already exists: {out_root}; use a new --version")

    manifest = {
        "schema_version": 2,
        "study_version": args.version,
        "arms": arms,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assignment_seed": args.seed,
        "n_items": len(items),
        "n_answer_items": sum(1 for item in items if item["block"] == "answer"),
        "n_retrieval_items": sum(1 for item in items if item["block"] == "retrieval"),
        "sample": sample_meta,
        "runs": run_meta,
        "dataset": {"sha256": file_sha256(paths.dataset_parquet)},
        "configs": {"finalist": "configs/revision/finalists_v1.yaml",
                    "sha256": file_sha256(paths.root / "configs/revision/finalists_v1.yaml")},
        "git": git_info(),
        "notes": args.notes,
        "freeze_rule": (
            "This package is immutable. Any change to retrieval, answers, assignment or "
            "sample requires a new study version and a new expert evaluation."
        ),
    }
    out_root.mkdir(parents=True, exist_ok=False)
    for name, payload in (("items.json", items), ("key.json", key), ("manifest.json", manifest)):
        (out_root / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sides = {side: sum(1 for v in key.values() if v["A"] == side) for side in ("baseline", "feedback")}
    print(f"items={len(items)} tickets={sum(meta['n'] for meta in sample_meta.values())} A-sides={sides}")
    for label, meta in sample_meta.items():
        print(f"  {label}: {meta['n']} tickets, eligible={meta['eligible']}, "
              f"changed_share={meta['changed_share']:.3f}")
    print(f"package={out_root}")
    print("key.json is private: do not copy it into the application directory.")


if __name__ == "__main__":
    main()
