#!/usr/bin/env python3
"""
07_validate_proxy.py — P1.1
============================
Validates the offline retrieval-level proxy against the generated-answer deltas of
the already-completed evaluation runs (no API calls).

For every *_details.json under results/ (or those given with --runs) it computes,
per ticket, proxy deltas (retrieved-reply similarity to the reference reply under
two embedding families + embedding-free hash hits) and reports:
  * Spearman / Pearson of proxy delta vs generated delta_cosine (all tickets and
    only tickets whose top-1 changed),
  * sign agreement on changed tickets,
  * whether the method ordering by mean generated delta is reproduced by the proxy,
  * onboarding/offboarding subgroup reproduction.

Decision rule (printed at the end): adopt the proxy for zero-cost sweeps if
Spearman(changed) >= 0.5 for the primary proxy and the method ordering matches.

Outputs: results/proxy_validation/{report.json, per_ticket.csv, manifest.json}
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
from scipy.stats import pearsonr, spearmanr

from experiments.utils import (setup_logging, load_dataset, get_arg_parser, write_run_manifest,
                               append_registry, make_run_id)
from src.config import ProjectPaths
from src.evaluation.proxy import ReplySimilarity, proxy_frame_from_details

PRIMARY = "minilm_d_proxy_top1"
PROXIES = ["proxy_top1", "proxy_best5", "proxy_mean5", "proxy_rr", "same_reply_hit1", "same_reply_hit5", "group_hit1", "group_hit5"]


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "spearman": None, "pearson": None}
    return {"n": int(len(x)), "spearman": float(spearmanr(x, y)[0]), "spearman_p": float(spearmanr(x, y)[1]),
            "pearson": float(pearsonr(x, y)[0])}


def find_runs(paths: ProjectPaths, patterns: list[str]) -> list[Path]:
    out = []
    for pat in patterns:
        out += [Path(p) for p in glob.glob(str(paths.results / pat / "*_details.json"))]
    # keep newest details per folder
    by_dir = {}
    for p in sorted(out):
        by_dir[p.parent] = p
    return list(by_dir.values())


def main() -> None:
    parser = get_arg_parser("Validate offline retrieval proxy against generated deltas")
    parser.add_argument("--runs", nargs="*", default=["M*_dev_*"], help="glob(s) of result folders")
    parser.add_argument("--models", nargs="*", default=["minilm", "bge"], choices=["minilm", "bge"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("validate_proxy")

    paths = ProjectPaths()
    out_dir = paths.results / "proxy_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("proxy_validation")

    df = load_dataset()
    sims = {m: ReplySimilarity(df, m) for m in args.models}
    runs = find_runs(paths, args.runs)
    runs = [r for r in runs if "baseline" not in r.parent.name]
    log.info("Found %d method runs", len(runs))

    frames, report = [], {"runs": {}, "models": args.models, "k": args.k}
    for details in runs:
        recs = json.loads(details.read_text("utf-8"))
        f = proxy_frame_from_details(recs, sims, args.k)
        f.insert(0, "run", details.parent.name)
        frames.append(f)
        entry = {"details": str(details), "n": int(len(f)),
                 "mean_delta_cosine": float(f["delta_cosine"].mean()),
                 "pct_top1_changed": float(f["top1_changed"].mean()), "proxies": {}}
        ch = f["top1_changed"] == 1
        for tag in args.models:
            for p in PROXIES:
                col = f"{tag}_d_{p}"
                entry["proxies"][col] = {
                    "mean_proxy_delta": float(f[col].mean()),
                    "all": _corr(f[col], f["delta_cosine"]),
                    "changed_only": _corr(f.loc[ch, col], f.loc[ch, "delta_cosine"]),
                    "sign_agreement_changed": float(np.mean(np.sign(f.loc[ch, col]) == np.sign(f.loc[ch, "delta_cosine"]))) if ch.any() else None,
                }
        # onboarding / offboarding subgroup
        onb = f["expected_class"].astype(str).str.contains("onboard|offboard", case=False, regex=True)
        entry["onboarding_subgroup"] = {
            "n": int(onb.sum()),
            "mean_delta_cosine": float(f.loc[onb, "delta_cosine"].mean()) if onb.any() else None,
            "mean_primary_proxy": float(f.loc[onb, PRIMARY].mean()) if onb.any() and PRIMARY in f else None,
        }
        report["runs"][details.parent.name] = entry
        log.info("%s: gen=%.4f  proxy_top1=%.4f  rho(changed)=%s", details.parent.name,
                 entry["mean_delta_cosine"], entry["proxies"][PRIMARY]["mean_proxy_delta"],
                 entry["proxies"][PRIMARY]["changed_only"]["spearman"])

    per_ticket = pd.concat(frames, ignore_index=True)
    per_ticket.to_csv(out_dir / "per_ticket.csv", index=False)

    # Method ordering agreement
    order_gen = sorted(report["runs"], key=lambda r: report["runs"][r]["mean_delta_cosine"])
    ordering = {}
    for tag in args.models:
        for p in ("proxy_top1", "proxy_best5", "same_reply_hit1", "group_hit1"):
            col = f"{tag}_d_{p}"
            order_p = sorted(report["runs"], key=lambda r: report["runs"][r]["proxies"][col]["mean_proxy_delta"])
            rho = spearmanr([order_gen.index(r) for r in order_gen], [order_p.index(r) for r in order_gen])[0] if len(order_gen) > 2 else None
            ordering[col] = {"order_by_proxy": order_p, "rank_spearman_vs_generated": None if rho is None else float(rho)}
    report["method_order_by_generated_delta"] = order_gen
    report["method_ordering_by_proxy"] = ordering

    # Pooled correlation across runs
    ch = per_ticket["top1_changed"] == 1
    report["pooled"] = {}
    for tag in args.models:
        for p in PROXIES:
            col = f"{tag}_d_{p}"
            report["pooled"][col] = {"all": _corr(per_ticket[col], per_ticket["delta_cosine"]),
                                     "changed_only": _corr(per_ticket.loc[ch, col], per_ticket.loc[ch, "delta_cosine"])}

    prim = report["pooled"][PRIMARY]["changed_only"]["spearman"]
    prim_order = ordering[PRIMARY]["rank_spearman_vs_generated"]
    best_proxy = max(report["pooled"], key=lambda c: (report["pooled"][c]["changed_only"]["spearman"] or -1))
    report["decision"] = {
        "primary_proxy": PRIMARY,
        "pooled_spearman_changed": prim,
        "method_order_rank_spearman": prim_order,
        "best_per_ticket_proxy": best_proxy,
        "best_per_ticket_spearman_changed": report["pooled"][best_proxy]["changed_only"]["spearman"],
        # Use 1: choosing configurations (method-level means) — needs ordering agreement + a clearly positive per-ticket link.
        "adopt_for_config_selection": bool(prim is not None and prim >= 0.3 and (prim_order is None or prim_order >= 0.8)),
        # Use 2: per-ticket labels (gate training) — needs a stronger per-ticket link; otherwise validate the gate on generated labels.
        "adopt_for_per_ticket_labels": bool(prim is not None and prim >= 0.5),
        "rule": ("config selection: Spearman(changed) >= 0.3 and method-order rank-Spearman >= 0.8; "
                 "per-ticket labels: Spearman(changed) >= 0.5 (else train on proxy but report gate quality on generated labels)"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = write_run_manifest(out_dir, script="experiments/07_validate_proxy.py", args=args,
                                  inputs=[paths.dataset_parquet] + runs, extra={"n_runs": len(runs)}, run_id=run_id)
    append_registry(run_id=run_id, phase="P1.1-proxy", script="07_validate_proxy.py", out_dir=out_dir, manifest=manifest,
                    headline_metric="pooled_spearman_changed(primary)", headline_value=prim,
                    notes=f"config_selection={report['decision']['adopt_for_config_selection']} per_ticket={report['decision']['adopt_for_per_ticket_labels']}")

    print("\n=== PROXY VALIDATION ===")
    print(f"runs: {len(runs)}   tickets: {len(per_ticket)}   changed-top1 rows: {int(ch.sum())}")
    print("pooled Spearman (changed-only) per proxy:")
    for col, v in report["pooled"].items():
        print(f"  {col:32s} all={v['all']['spearman']!s:>8}  changed={v['changed_only']['spearman']!s:>8}")
    print(f"method order by generated delta : {order_gen}")
    for col in dict.fromkeys([PRIMARY, "bge_d_proxy_top1" if "bge" in args.models else PRIMARY]):
        print(f"method order by {col:24s}: {ordering[col]['order_by_proxy']}  rank-rho={ordering[col]['rank_spearman_vs_generated']}")
    print(f"DECISION: adopt_for_config_selection = {report['decision']['adopt_for_config_selection']}   "
          f"adopt_for_per_ticket_labels = {report['decision']['adopt_for_per_ticket_labels']}   "
          f"(best per-ticket proxy: {best_proxy})")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
