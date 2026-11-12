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


def _check_same_baseline(runs: dict[str, dict], label: str) -> None:
    sources = list(runs)
    reference = sources[0]
    for ticket_id, record in runs[reference].items():
        answer = record["baseline"]["answer"]
        retrieval = [c["seq_id"] for c in record["baseline"]["retrieval"]]
        for other in sources[1:]:
            if ticket_id not in runs[other]:
                continue
            if runs[other][ticket_id]["baseline"]["answer"] != answer:
                raise SystemExit(
                    f"Baseline answer differs between {reference} and {other} for {ticket_id} "
                    f"in {label}; the generated runs are not one comparable regime"
                )
            if [c["seq_id"] for c in runs[other][ticket_id]["baseline"]["retrieval"]] != retrieval:
                raise SystemExit(
                    f"Baseline retrieval differs between {reference} and {other} for {ticket_id} "
                    f"in {label}"
                )


def _sample(records: dict, sources: dict, seed: str, n: int | None, sources_list: list[str],
            changed_fraction: float = 0.7, stratum: str = "answer"):
    reference = sorted(records)
    if not n:
        return reference, {}, {}

    def changed(ticket_id: str) -> bool:
        for source in sources_list:
            record = sources[source][ticket_id]
            if stratum == "answer":
                if record["baseline"]["answer"] != record["feedback"]["answer"]:
                    return True
            elif (record["baseline"]["retrieval"][0]["seq_id"]
                  != record["feedback"]["retrieval"][0]["seq_id"]):
                return True
        return False

    groups = {"changed": [], "unchanged": []}
    for ticket_id in reference:
        groups["changed" if changed(ticket_id) else "unchanged"].append(ticket_id)
    wanted = min(n, len(reference))
    changed_n = min(len(groups["changed"]), int(round(wanted * changed_fraction)))
    unchanged_n = min(len(groups["unchanged"]), wanted - changed_n)
    if changed_n + unchanged_n < wanted:
        changed_n = min(len(groups["changed"]), wanted - unchanged_n)
    if changed_n + unchanged_n < wanted:
        unchanged_n = min(len(groups["unchanged"]), wanted - changed_n)
    allocation = {"changed": changed_n, "unchanged": unchanged_n}
    selected = []
    for name, ids in groups.items():
        ordered = sorted(ids, key=lambda t: hashlib.sha256(
            f"{seed}|sample|{t}".encode()).hexdigest())
        selected.extend(ordered[: allocation[name]])
    return sorted(selected), {name: len(ids) for name, ids in groups.items()}, allocation


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
    parser.add_argument("--changed-fraction", type=float, default=0.7,
                        help="Share of the sample drawn from tickets whose content changed")
    parser.add_argument("--stratum", choices=["answer", "retrieval"], default="answer",
                        help="What counts as a changed ticket for stratification")
    parser.add_argument("--run", action="append", default=[],
                        help="Override a run directory: source=relative/or/absolute/path")
    parser.add_argument("--out-root", default="revision_ieee_access/02_human_study")
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
    for panel, sources in by_panel.items():
        sets = {source: set(records) for source, records in sources.items()}
        reference = set(next(iter(sets.values())))
        if any(value != reference for value in sets.values()):
            raise SystemExit(f"Runs of panel {panel} do not cover the same query set")
        _check_same_baseline(sources, panel)

    items, key, sample_meta = [], {}, {}
    for arm in arms:
        label = f"{arm['panel']}/{arm['source']}"
        records = loaded[label]
        selected, strata, allocation = _sample(
            records, {arm["source"]: records}, args.seed, arm["sample"], [arm["source"]],
            changed_fraction=args.changed_fraction, stratum=args.stratum)
        sample_meta[label] = {"requested": arm["sample"], "n": len(selected),
                              "stratum": args.stratum,
                              "changed_fraction": args.changed_fraction,
                              "strata": strata, "allocation": allocation}
        for ticket_id in selected:
            record = records[ticket_id]
            if not str(record["feedback"]["answer"]).strip() or not str(record["baseline"]["answer"]).strip():
                raise SystemExit(f"Empty answer for {ticket_id}/{arm['source']}")
            stem = _opaque_id(args.seed, label, ticket_id)
            for block, item_id in (("answer", f"{stem}-A"), ("retrieval", f"{stem}-R")):
                items.append({"item_id": item_id, "block": block})
                key[item_id] = {"panel": arm["panel"], "ticket_id": ticket_id,
                                "source": arm["source"], "block": block, "A": None, "B": None}

    for block in BLOCKS:
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
        print(f"  {label}: {meta['n']} tickets, strata={meta['strata']}")
    print(f"package={out_root}")
    print("key.json is private: do not copy it into the application directory.")


if __name__ == "__main__":
    main()
