#!/usr/bin/env python3
"""22_revision_dev_lab.py — dev-only selection (no API calls).

Answers three reviewer requests on development data only:

* R1-M3 / R2-13: tune the Laplace multiplier and cap, then compare with the
  published empirical-Bayes lift under a matched search budget.
* R1-M2 / R2-13: isolate prior, scaling, scope and routing one factor at a time.
* R1-M5: replace the LLM judge with reply similarity (exact and embedding) on
  exactly the same historical pairs and the same lift.

Selection rule (declared before inspection, applied mechanically):

* primary metric ``proxy_top1`` (MiniLM reply cosine), tie-break ``proxy_mean5``;
* every candidate cell of the grid is evaluated; the legacy cell (m=0.8, cap=0.2,
  absolute, intersection) is included but not privileged;
* each lift family receives the same 5x5 budget per scope;
* selection uses dev only; eval panels are not read by this script.

Outputs (private, git-ignored): ``results/revision/dev_lab/<run_id>/``
    dev_grid.csv        every evaluated cell
    dev_selection.csv   best cell per panel x source x family x scope
    dev_ablation.csv    one-factor-at-a-time contrasts with bootstrap CIs
    summary.json        rule, counts and selected cells
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, make_run_id, setup_logging
from src.config import LiftConfig, ProjectPaths, RoutingConfig
from src.data.artifacts import write_once_json
from src.data.groups_v2 import normalize_signature
from src.evaluation.ladder import build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.evaluation.revision_panels import (
    bootstrap,
    collect_pools,
    judge_bundle,
    load_feedback_rows,
    p_value,
    panel_specs,
    resolve_panel,
    saturation,
    ticket_embeddings,
)
from src.feedback.loader import FeedbackBundle
from src.feedback.reply_similarity import with_reply_similarity
from src.retrieval.history import build_history_retriever

PANELS = ("legacy_random", "disjoint_duplicate", "disjoint_prefix")
SOURCES = ("conditioned", "blind", "reply_exact", "reply_embed_bge")
SCOPES = ("team", "intersection")

MULTIPLIERS = (0.2, 0.4, 0.6, 0.8, 1.0)
CAPS = (0.05, 0.10, 0.15, 0.20, 0.30)
KAPPAS = (2.0, 5.0, 10.0, 20.0, 50.0)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _routing(scope: str):
    return RoutingConfig.team_only() if scope == "team" else RoutingConfig.intersection()


def _laplace_absolute() -> list[dict]:
    return [
        {"family": "laplace_absolute", "scope": scope, "m": m, "cap": cap, "lambda": None,
         "kappa": None, "lift": LiftConfig(name="laplace", multiplier=m, cap=cap),
         "routing": _routing(scope)}
        for scope in SCOPES for m in MULTIPLIERS for cap in CAPS
    ]


def _laplace_pool_std() -> list[dict]:
    return [
        {"family": "laplace_pool_std", "scope": scope, "m": m, "cap": 0.20, "lambda": lam,
         "kappa": None,
         "lift": LiftConfig(name="laplace", multiplier=m, cap=0.20,
                            scale_mode="pool_std", pool_lambda=lam),
         "routing": _routing(scope)}
        for scope in SCOPES for m in MULTIPLIERS for lam in LAMBDAS
    ]


def _eb_grid() -> list[dict]:
    cells = []
    for scope in SCOPES:
        for kappa in KAPPAS:
            cells.append({"family": "eb_absolute", "scope": scope, "m": 0.80, "cap": 0.20,
                          "lambda": None, "kappa": kappa,
                          "lift": LiftConfig.laplace_eb(prior_strength=kappa),
                          "routing": _routing(scope)})
            for lam in LAMBDAS:
                cells.append({"family": "eb_pool_std", "scope": scope, "m": 0.80, "cap": 0.20,
                              "lambda": lam, "kappa": kappa,
                              "lift": LiftConfig.laplace_eb(prior_strength=kappa,
                                                            scale_mode="pool_std",
                                                            pool_lambda=lam),
                              "routing": _routing(scope)})
    return cells


def _cell_key(cell: dict) -> str:
    parts = [cell["family"], cell["scope"], f"m{cell['m']:g}", f"cap{cell['cap']:g}"]
    if cell["lambda"] is not None:
        parts.append(f"lam{cell['lambda']:g}")
    if cell["kappa"] is not None:
        parts.append(f"k{cell['kappa']:g}")
    return "|".join(parts)


def _evaluate(tensor, baseline: dict, base_top, lift: LiftConfig, routing) -> dict:
    lift_values = tensor.routed_lift(lift, routing)
    factor = tensor.scale_factor(lift)
    top = tensor.rerank(lift_values, factor, 5)
    metrics = tensor.metrics(top, "minilm")
    return {
        "proxy_top1": float((metrics["proxy_top1"] - baseline["proxy_top1"]).mean()),
        "proxy_mean5": float((metrics["proxy_mean5"] - baseline["proxy_mean5"]).mean()),
        "full_reply_hit5": float((metrics["same_reply_hit5"] - baseline["same_reply_hit5"]).mean()),
        "v2_group_hit5": float((metrics["group_hit5"] - baseline["group_hit5"]).mean()),
        "pct_top1_changed": float((top[:, 0] != base_top[:, 0]).mean()),
    }


def _bundle_for(context, dataset, rows, source: str, cache_dir: Path) -> FeedbackBundle:
    if source in {"conditioned", "blind"}:
        return judge_bundle(context, rows, source)[0]
    allowed, _ = context.select_feedback(rows, protocol="conditioned")
    kind = source.removeprefix("reply_")
    return FeedbackBundle(with_reply_similarity(allowed, dataset, kind, cache_dir=cache_dir))


def _build(panel_spec, partition, paths, dataset, embeddings, source, raw_rows, cache_dir, search_k):
    """Resolve a panel, build its history-only pools and one evidence tensor."""
    context = resolve_panel(panel_spec, paths, partition=partition)
    retriever = build_history_retriever("dense_minilm", dataset, set(context.history_ids))
    pools = collect_pools(context, dataset, embeddings, retriever, search_k)
    queries = dataset.set_index("seq_id").loc[list(context.query_ids)].reset_index()
    group_map = dict(context.group_ids)
    reply_full = dict(zip(dataset["seq_id"], dataset["first_reply"].map(normalize_signature)))
    similarity = ReplySimilarity(dataset, "minilm", reply_hash_override=reply_full,
                                 group_hash_override=group_map,
                                 reply_key_label="full_reply_v2",
                                 group_key_label=f"group:{context.group_rule_id}")
    rows = raw_rows[source if source in raw_rows else "conditioned"]
    bundle = _bundle_for(context, dataset, rows, source, cache_dir)
    tensor = build_pool_tensor(pools, queries, bundle, {"minilm": similarity})
    return tensor, context, bundle


def _lift_from_selection(row, family: str) -> LiftConfig:
    if family.startswith("laplace"):
        if family.endswith("pool_std"):
            return LiftConfig(name="laplace", multiplier=float(row.m), cap=float(row.cap),
                              scale_mode="pool_std", pool_lambda=float(row.pool_lambda))
        return LiftConfig(name="laplace", multiplier=float(row.m), cap=float(row.cap))
    return LiftConfig.laplace_eb(prior_strength=float(row.kappa),
                                 scale_mode="pool_std" if family.endswith("pool_std") else "absolute",
                                 pool_lambda=float(row.pool_lambda) if family.endswith("pool_std") else 1.0)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--panels", nargs="+", default=list(PANELS))
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--validate-eval", action="store_true",
                        help="Evaluate the selected dev cells on the eval partition")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)
    embeddings = ticket_embeddings(dataset, paths)
    specs = panel_specs(paths, 42)
    raw_rows = {
        "conditioned": load_feedback_rows(paths.data_processed / "feedback_conditioned.db"),
        "blind": load_feedback_rows(paths.data_processed / "feedback_blind.db"),
    }
    cache_dir = paths.root / "data" / "cache"

    grid_cells = _laplace_absolute() + _laplace_pool_std() + _eb_grid()
    grid_rows, selection_rows, saturation_rows = [], [], []
    tensors: dict[tuple[str, str], object] = {}
    baselines: dict[str, dict] = {}

    for panel_name in args.panels:
        for source in args.sources:
            tensor, context, bundle = _build(specs[panel_name], "dev", paths, dataset, embeddings,
                                             source, raw_rows, cache_dir, args.search_k)
            tensors[(panel_name, source)] = tensor
            base_top = tensor.baseline_top(args.top_k)
            base_metrics = tensor.metrics(base_top, "minilm")
            baselines[panel_name] = {k: v.mean() for k, v in base_metrics.items()}

            for cell in grid_cells:
                result = _evaluate(tensor, base_metrics, base_top, cell["lift"], cell["routing"])
                grid_rows.append({
                    "panel": panel_name, "source": source, "family": cell["family"],
                    "scope": cell["scope"], "m": cell["m"], "cap": cell["cap"],
                    "pool_lambda": cell["lambda"], "kappa": cell["kappa"],
                    **result,
                })
            print(f"[{panel_name}/{source}] dev queries={len(context.query_ids)} "
                  f"history={len(context.history_ids)} feedback_rows={len(bundle.rows)} "
                  f"prior={bundle.prior_mean('global'):.3f}")

    grid = pd.DataFrame(grid_rows)
    # Mechanical selection: highest dev proxy_top1, tie-break proxy_mean5.
    for (panel, source, family, scope), group in grid.groupby(["panel", "source", "family", "scope"]):
        best = group.sort_values(["proxy_top1", "proxy_mean5"], ascending=False).iloc[0]
        selection_rows.append({
            "panel": panel, "source": source, "family": family, "scope": scope,
            "m": best["m"], "cap": best["cap"], "pool_lambda": best["pool_lambda"],
            "kappa": best["kappa"], "delta_top1": best["proxy_top1"],
            "delta_mean5": best["proxy_mean5"],
        })
    selection = pd.DataFrame(selection_rows)
    validation_rows = []
    if args.validate_eval:
        for panel_name in args.panels:
            for source in args.sources:
                tensor, context, bundle = _build(specs[panel_name], "eval", paths, dataset,
                                                 embeddings, source, raw_rows, cache_dir,
                                                 args.search_k)
                base_top = tensor.baseline_top(args.top_k)
                base_metrics = tensor.metrics(base_top, "minilm")
                subset = selection[selection.panel.eq(panel_name) & selection.source.eq(source)]
                for family, group in subset.groupby("family"):
                    best = group.sort_values(["delta_top1", "delta_mean5"],
                                             ascending=False).iloc[0]
                    lift = _lift_from_selection(best, family)
                    routing = _routing(best["scope"])
                    values = tensor.routed_lift(lift, routing)
                    factor = tensor.scale_factor(lift)
                    top = tensor.rerank(values, factor, args.top_k)
                    metrics = tensor.metrics(top, "minilm")
                    delta = metrics["proxy_top1"] - base_metrics["proxy_top1"]
                    stats = bootstrap(delta)
                    validation_rows.append({
                        "panel": panel_name, "source": source, "family": family,
                        "scope": best["scope"], "m": best["m"], "cap": best["cap"],
                        "pool_lambda": best["pool_lambda"], "kappa": best["kappa"],
                        "dev_delta_top1": best["delta_top1"],
                        "eval_delta_top1": stats["mean"],
                        "eval_ci_lower": stats["mean_ci_lower"],
                        "eval_ci_upper": stats["mean_ci_upper"],
                        "eval_wilcoxon_p": p_value(delta),
                        "n_eval": int(len(delta)),
                    })
                    print(f"[validate {panel_name}/{source}/{family}] dev={best['delta_top1']:+.4f} "
                          f"eval={stats['mean']:+.4f} [{stats['mean_ci_lower']:+.4f}, "
                          f"{stats['mean_ci_upper']:+.4f}]")

    # One-factor-at-a-time contrasts on the two corrected panels, conditioned feedback.
    ablation_rows = []
    contrasts = []
    for panel_name in [p for p in args.panels if p != "legacy_random"]:
        tensor = tensors[(panel_name, "conditioned")]
        base_metrics = tensor.metrics(tensor.baseline_top(args.top_k), "minilm")
        tuned = selection[(selection.panel.eq(panel_name)) & (selection.source.eq("conditioned"))
                          & (selection.family.eq("laplace_absolute"))
                          & (selection.scope.eq("intersection"))].iloc[0]
        variants = [
            ("01_laplace_legacy", LiftConfig(name="laplace", multiplier=0.8, cap=0.2),
             RoutingConfig.intersection()),
            ("02_laplace_tuned", LiftConfig(name="laplace", multiplier=float(tuned.m),
                                             cap=float(tuned.cap)), RoutingConfig.intersection()),
            ("03_laplace_pool_std_lambda1", LiftConfig(name="laplace", multiplier=0.8, cap=0.2,
                                                       scale_mode="pool_std", pool_lambda=1.0),
             RoutingConfig.intersection()),
            ("04_eb_k2_absolute", LiftConfig.laplace_eb(prior_strength=2.0),
             RoutingConfig.intersection()),
            ("05_eb_k2_pool_std_lambda1", LiftConfig.laplace_eb(prior_strength=2.0,
                                                                scale_mode="pool_std",
                                                                pool_lambda=1.0),
             RoutingConfig.intersection()),
            ("06_eb_k2_pool_std_team", LiftConfig.laplace_eb(prior_strength=2.0,
                                                              scale_mode="pool_std",
                                                              pool_lambda=1.0),
             RoutingConfig.team_only()),
            ("07_eb_k2_pool_std_backoff", LiftConfig.laplace_eb(prior_strength=2.0,
                                                                 scale_mode="pool_std",
                                                                 pool_lambda=1.0),
             RoutingConfig.backoff()),
        ]
        for label, lift, routing in variants:
            lift_values = tensor.routed_lift(lift, routing)
            factor = tensor.scale_factor(lift)
            top = tensor.rerank(lift_values, factor, args.top_k)
            metrics = tensor.metrics(top, "minilm")
            delta = metrics["proxy_top1"] - base_metrics["proxy_top1"]
            stats = bootstrap(delta)
            ablation_rows.append({
                "panel": panel_name, "contrast": label, "scope": routing.name,
                "multiplier": lift.multiplier, "cap": lift.cap,
                "pool_lambda": lift.pool_lambda if lift.scale_mode == "pool_std" else None,
                "kappa": lift.prior_strength if lift.name == "laplace_eb" else None,
                "delta_top1": stats["mean"], "ci_lower": stats["mean_ci_lower"],
                "ci_upper": stats["mean_ci_upper"], "wilcoxon_p": p_value(delta),
                "n_queries": int(len(delta)),
                **saturation(tensor, lift, routing),
            })
            contrasts.append(label)
        # Legacy cell rank inside the Laplace absolute sweep at intersection.
        sweep = grid[(grid.panel.eq(panel_name)) & (grid.source.eq("conditioned"))
                     & (grid.family.eq("laplace_absolute")) & (grid.scope.eq("intersection"))]
        legacy = sweep[(sweep.m.eq(0.8)) & (sweep.cap.eq(0.2))].iloc[0]
        rank = int((sweep.proxy_top1 > legacy.proxy_top1).sum()) + 1
        ablation_rows.append({
            "panel": panel_name, "contrast": "08_legacy_cell_rank",
            "scope": "intersection", "multiplier": 0.8, "cap": 0.2,
            "pool_lambda": None, "kappa": None,
            "delta_top1": float(legacy.proxy_top1), "ci_lower": None, "ci_upper": None,
            "wilcoxon_p": None, "n_queries": len(sweep),
            "legacy_rank_in_sweep": rank, "sweep_cells": int(len(sweep)),
        })

    run_id = make_run_id("dev_lab")
    out_dir = paths.results / "revision" / "dev_lab" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "dev_grid.csv", index=False)
    selection.to_csv(out_dir / "dev_selection.csv", index=False)
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(out_dir / "dev_ablation.csv", index=False)
    if validation_rows:
        pd.DataFrame(validation_rows).to_csv(out_dir / "dev_to_eval.csv", index=False)

    summary = {
        "run_id": run_id,
        "selection_rule": "max dev mean proxy_top1; tie-break proxy_mean5; dev only",
        "grid_cells_per_scope_per_family": 25,
        "panels": list(args.panels),
        "sources": list(args.sources),
        "baselines": baselines,
        "n_grid_rows": int(len(grid)),
        "inputs": {
            "dataset": {"sha256": file_sha256(paths.dataset_parquet)},
            "feedback_conditioned": {
                "sha256": file_sha256(paths.data_processed / "feedback_conditioned.db")},
            "reply_embeddings": {
                name: {"sha256": file_sha256(paths.root / "data" / "cache" / name)}
                for name in ("reply_emb_bge.npy", "reply_emb_minilm.npy")
                if (paths.root / "data" / "cache" / name).exists()
            },
        },
        "git": git_info(),
    }
    write_once_json(out_dir / "summary.json", summary)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
