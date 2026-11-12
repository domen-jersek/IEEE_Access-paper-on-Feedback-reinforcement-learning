#!/usr/bin/env python3
"""21_revision_retrieval_panels.py — R1-M7 retrieval-level panels (no API calls).

Compares, on identical method settings, three data regimes:

* ``legacy_random``        submitted random split, candidates/evidence restricted
                           to its train partition (the revision's control).
* ``disjoint_duplicate``   group-aware split under ``duplicate_template_v1``
                           (equal full reply OR equal title+subcategory).
* ``disjoint_prefix``      conservative split under ``prefix_template_v1``
                           (the above plus equal reply opening).

Every panel uses history-only retrieval (dense index rebuilt from frozen
vectors), history-only feedback rows (both endpoints), no self-pairs, and priors
estimated from exactly those rows. Metrics are retrieval-level reply-similarity
proxies, not generated answers. Generated panels require separate approval.

Outputs (a private local directory; git-ignored):
    results/revision/retrieval_panels/<run_id>/
        panel_metrics.csv        per panel x config x metric
        saturation.csv           lift saturation diagnostics
        pool_overlap_audit.csv   structural overlap inside each retrieval pool
        summary.json             configuration, coverage, headline rows
        manifest_<run_id>.json   input hashes and environment
"""
from __future__ import annotations

import json
import hashlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import file_sha256, git_info, make_run_id, setup_logging
from src.config import DataProtocolConfig, GENERATOR_MODEL, LiftConfig, ProjectPaths, RoutingConfig
from src.data.artifacts import write_once_json
from src.data.groups_v2 import normalize_signature
from src.data.protocol import resolve_data_context
from src.evaluation.ladder import build_pool_tensor
from src.evaluation.proxy import ReplySimilarity
from src.feedback.loader import FeedbackBundle
from src.generation.prompts import SYSTEM_PROMPT, build_generation_prompt
from src.retrieval.history import build_history_retriever

SCOPES = ("global", "class", "team", "intersection")

# One compact grid: scope x lift x scaling, plus the calibrated baseline.
GRID = [
    ("M4_intersection", "laplace_legacy", LiftConfig.laplace(), RoutingConfig.intersection()),
    ("M4_intersection", "laplace_eb_k2", LiftConfig.laplace_eb(2.0), RoutingConfig.intersection()),
    ("M4_intersection", "laplace_eb_k2_poolstd",
     LiftConfig.laplace_eb(2.0, scale_mode="pool_std", pool_lambda=1.0), RoutingConfig.intersection()),
    ("M2_team", "laplace_legacy", LiftConfig.laplace(), RoutingConfig.team_only()),
    ("M2_team", "laplace_eb_k2_poolstd",
     LiftConfig.laplace_eb(2.0, scale_mode="pool_std", pool_lambda=1.0), RoutingConfig.team_only()),
    ("M1_global", "laplace_eb_k2_poolstd",
     LiftConfig.laplace_eb(2.0, scale_mode="pool_std", pool_lambda=1.0), RoutingConfig.global_()),
    ("M5_backoff", "laplace_eb_k2", LiftConfig.laplace_eb(2.0), RoutingConfig.backoff()),
]


def _panel_specs(paths: ProjectPaths, split_seed: int = 42) -> dict[str, dict]:
    revision = paths.data_processed / "revision"
    suffix = "" if split_seed == 42 else f"_seed{split_seed}"
    return {
        "legacy_random": {
            "split": paths.splits_dir / "split_seed42.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "group_rule": "duplicate_template_v1",
            "scenario": "familiar_procedure_possible",
        },
        "legacy_disjoint": {
            "split": paths.splits_dir / "split_seed42_disjoint.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "group_rule": "duplicate_template_v1",
            "scenario": "familiar_procedure_possible",
        },
        f"disjoint_duplicate{suffix}": {
            "split": revision / "splits" / f"split_duplicate_template_v1_seed{split_seed}.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "group_rule": "duplicate_template_v1",
            "scenario": "held_out_groups",
        },
        f"disjoint_prefix{suffix}": {
            "split": revision / "splits" / f"split_prefix_template_v1_seed{split_seed}.json",
            "groups": revision / "groups" / "prefix_template_v1.json",
            "group_rule": "prefix_template_v1",
            "scenario": "held_out_groups",
        },
    }


def _context(panel: dict, paths: ProjectPaths):
    config = DataProtocolConfig(
        dataset_path=str(paths.dataset_parquet.relative_to(paths.root)).replace("\\", "/"),
        dataset_sha256=file_sha256(paths.dataset_parquet),
        split_path=str(panel["split"].relative_to(paths.root)).replace("\\", "/"),
        split_sha256=file_sha256(panel["split"]),
        groups_path=str(panel["groups"].relative_to(paths.root)).replace("\\", "/"),
        groups_sha256=file_sha256(panel["groups"]),
        scenario=panel["scenario"],
        metadata_policy="posthoc_diagnostic",  # routing labels held fixed across panels
        query_partition="eval",
    )
    return resolve_data_context(config, root=paths.root)


def _ticket_embeddings(dataset: pd.DataFrame, paths: ProjectPaths) -> np.ndarray:
    cache = paths.root / "data" / "cache"
    ids = json.loads((cache / "ticket_text_emb_minilm_ids.json").read_text("utf-8"))
    if ids != dataset["seq_id"].astype(str).tolist():
        raise ValueError("Cached ticket embeddings are not aligned with the canonical dataset")
    matrix = np.load(cache / "ticket_text_emb_minilm.npy")
    if matrix.shape[0] != len(dataset):
        raise ValueError("Cached ticket embedding matrix has the wrong shape")
    return matrix.astype(np.float32)


def _collect_history_pools(context, dataset, embeddings, retriever, search_k: int) -> pd.DataFrame:
    position = {seq_id: i for i, seq_id in enumerate(dataset["seq_id"].astype(str))}
    rows = []
    for qid in context.query_ids:
        if qid in context.history_ids:
            raise AssertionError(f"Eval query {qid} is inside the allowed history")
        row = dataset.iloc[position[qid]]
        description = str(row["Description_anon"] or "")
        text = f"{row['Title_anon']}\n{description}" if description else str(row["Title_anon"])
        pool = retriever.search_pool(text, embeddings[position[qid]][None, :], search_k, set())
        for record in pool.itertuples(index=False):
            rows.append((qid, str(record.seq_id), float(record.faiss_score), int(record.faiss_rank)))
    pools = pd.DataFrame(rows, columns=["query_id", "seq_id", "faiss_score", "faiss_rank"])
    forbidden = set(pools["seq_id"]) - set(context.history_ids)
    if forbidden:
        raise AssertionError(f"Pool contains {len(forbidden)} candidates outside history")
    return pools


def _load_feedback(context, db_path: Path, protocol: str) -> tuple[FeedbackBundle, dict]:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = pd.read_sql_query(
            "SELECT query_id, candidate_id, query_class, query_team, score, protocol FROM feedback",
            connection,
        )
    allowed, audit = context.select_feedback(rows, protocol=protocol)
    return FeedbackBundle(allowed), audit


def _bootstrap(delta: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(n_resamples, len(delta)))
    sampled = delta[draws]
    means = sampled.mean(axis=1)
    medians = np.median(sampled, axis=1)
    return {
        "mean": float(delta.mean()),
        "mean_ci_lower": float(np.percentile(means, 2.5)),
        "mean_ci_upper": float(np.percentile(means, 97.5)),
        "median": float(np.median(delta)),
        "median_ci_lower": float(np.percentile(medians, 2.5)),
        "median_ci_upper": float(np.percentile(medians, 97.5)),
    }


def _p_value(delta: np.ndarray) -> float:
    if not np.any(delta != 0):
        return float("nan")
    return float(stats.wilcoxon(delta).pvalue)


def _pool_overlap(context, dataset, pools: pd.DataFrame, group_map: dict[str, str]) -> pd.DataFrame:
    frame = dataset.set_index("seq_id")
    title = frame["Title_anon"].map(normalize_signature)
    subcategory = frame["Service subcategory->Name"].map(normalize_signature)
    families = {
        "v2_group": group_map,
        "full_reply": dict(zip(frame.index, frame["first_reply"].map(normalize_signature))),
        "reply_prefix300": dict(zip(frame.index, frame["first_reply"].fillna("")
                                    .astype(str).str[:300].map(normalize_signature))),
        "title_subcategory": dict(zip(frame.index, (title + " | " + subcategory).where(
            ~((title == "") & (subcategory == "")), ""))),
    }
    rows = []
    grouped = {qid: g for qid, g in pools.groupby("query_id")}
    for family, mapping in families.items():
        affected = 0
        excess = 0
        for qid in context.query_ids:
            own = mapping.get(qid)
            matches = int(sum(1 for cid in grouped[qid]["seq_id"] if mapping.get(cid) == own))
            affected += matches > 0
            excess += matches
        rows.append({
            "family": family,
            "queries_with_shared_candidate": affected,
            "fraction_queries": affected / len(context.query_ids),
            "shared_candidates_total": excess,
            "pool_slots": int(len(pools)),
        })
    return pd.DataFrame(rows)


def _generation_cache_keys(paths: ProjectPaths) -> set[str]:
    path = paths.root / "data" / "cache" / "generation_cache.db"
    if not path.exists():
        return set()
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        return {row[0] for row in connection.execute("SELECT key FROM cache")}


def _prompt_keys(tensor, top: np.ndarray, index: pd.DataFrame) -> list[str]:
    """Exact generation prompt keys for a (Q, k) selection, without API calls."""
    keys = []
    for qi, qid in enumerate(tensor.query_ids):
        query = index.loc[qid]
        candidates = []
        for j in top[qi]:
            candidate_id = tensor.cand_ids[qi, j]
            candidate = index.loc[candidate_id]
            candidates.append({
                "Title_anon": candidate["Title_anon"],
                "first_reply": candidate["first_reply"],
                "faiss_score": float(tensor.score[qi, j]),
            })
        prompt = build_generation_prompt(
            str(query["Title_anon"]), str(query["Description_anon"] or ""), candidates)
        payload = {"model": GENERATOR_MODEL, "temperature": 0.0,
                   "prompt": prompt, "system": SYSTEM_PROMPT}
        keys.append(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
    return keys


def _saturation(tensor, lift_config: LiftConfig, routing: RoutingConfig) -> dict:
    lift = tensor.routed_lift(lift_config, routing)
    factor = tensor.scale_factor(lift_config)
    mask = tensor.mask
    effective = np.abs(lift * factor[:, None])
    cap = lift_config.cap
    at_cap = np.isclose(np.abs(lift), cap) & mask
    scores = np.where(mask, tensor.score, np.nan)
    score_std = np.nanstd(scores, axis=1, ddof=1)
    lift_std = np.nanstd(np.where(mask, lift * factor[:, None], np.nan), axis=1, ddof=1)
    ratio = lift_std / np.where(score_std > 0, score_std, np.nan)
    return {
        "pct_candidates_at_cap": float(at_cap.sum() / max(mask.sum(), 1)),
        "pct_candidates_zero": float((np.isclose(lift, 0.0) & mask).sum() / max(mask.sum(), 1)),
        "mean_abs_lift": float(np.nanmean(np.where(mask, effective, np.nan))),
        "mean_lift_std_over_score_std": float(np.nanmean(ratio)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="conditioned", choices=["conditioned", "blind"])
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--panels", nargs="+", default=[
        "legacy_random", "legacy_disjoint", "disjoint_duplicate", "disjoint_prefix"])
    parser.add_argument("--generation-budget", action="store_true",
                        help="Also count exact generation-prompt cache hits/misses (no API calls)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Group-aware split seed; legacy panels are seed-42 only")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)

    paths = ProjectPaths()
    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)
    embeddings = _ticket_embeddings(dataset, paths)
    specs = _panel_specs(paths, args.split_seed)
    if args.split_seed != 42:
        # Legacy splits exist only for seed 42; do not silently relabel them.
        args.panels = [name for name in args.panels if not name.startswith("legacy_")]
        args.panels = [name if name in specs else f"{name}_seed{args.split_seed}"
                       for name in args.panels]
    unknown = [name for name in args.panels if name not in specs]
    if unknown:
        raise ValueError(f"Unknown panels: {unknown}; available: {sorted(specs)}")
    feedback_path = paths.data_processed / f"feedback_{args.protocol}.db"

    metric_rows, saturation_rows, overlap_rows, coverage_rows = [], [], [], []
    budget_rows: list[dict] = []
    panel_summaries = {}

    for panel_name in args.panels:
        context = _context(specs[panel_name], paths)
        retriever = build_history_retriever("dense_minilm", dataset, set(context.history_ids))
        pools = _collect_history_pools(context, dataset, embeddings, retriever, args.search_k)
        bundle, feedback_audit = _load_feedback(context, feedback_path, args.protocol)

        group_map = dict(context.group_ids)
        reply_full = dict(zip(dataset["seq_id"], dataset["first_reply"].map(normalize_signature)))
        similarity = ReplySimilarity(
            dataset, "minilm",
            reply_hash_override=reply_full,
            group_hash_override=group_map,
            reply_key_label="full_reply_v2",
            group_key_label=f"group:{context.group_rule_id}",
        )
        queries = dataset.set_index("seq_id").loc[list(context.query_ids)].reset_index()
        tensor = build_pool_tensor(pools, queries, bundle, {"minilm": similarity})

        baseline_top = tensor.baseline_top(args.top_k)
        baseline_metrics = tensor.metrics(baseline_top, "minilm")
        cache_keys = _generation_cache_keys(paths) if args.generation_budget else set()
        dataset_index = dataset.set_index("seq_id")
        if args.generation_budget:
            baseline_unique = set(_prompt_keys(tensor, baseline_top, dataset_index))
            budget_rows.append({
                "panel": panel_name, "method": "", "config": "baseline",
                "kind": "baseline", "n_queries": len(context.query_ids),
                "unique_prompts": len(baseline_unique),
                "cache_hits": len(baseline_unique & cache_keys),
                "cache_misses": len(baseline_unique - cache_keys),
            })
        coverage = tensor.coverage()
        for j, scope in enumerate(SCOPES):
            coverage_rows.append({
                "panel": panel_name, "scope": scope,
                "mean_pool_coverage": float(coverage[:, j].mean()),
                "queries_with_zero_evidence": int((coverage[:, j] == 0).sum()),
                "n_queries": int(len(context.query_ids)),
            })

        overlap = _pool_overlap(context, dataset, pools, group_map)
        overlap.insert(0, "panel", panel_name)
        overlap_rows.append(overlap)

        relabel = {"same_reply_hit1": "full_reply_hit1", "same_reply_hit5": "full_reply_hit5",
                   "group_hit1": "v2_group_hit1", "group_hit5": "v2_group_hit5"}
        for method, config_name, lift_config, routing in GRID:
            lift = tensor.routed_lift(lift_config, routing)
            factor = tensor.scale_factor(lift_config)
            top = tensor.rerank(lift, factor, args.top_k)
            feedback_metrics = tensor.metrics(top, "minilm")
            if args.generation_budget:
                feedback_unique = baseline_unique | set(_prompt_keys(tensor, top, dataset_index))
                budget_rows.append({
                    "panel": panel_name, "method": method, "config": config_name,
                    "kind": "baseline+feedback", "n_queries": len(context.query_ids),
                    "unique_prompts": len(feedback_unique),
                    "cache_hits": len(feedback_unique & cache_keys),
                    "cache_misses": len(feedback_unique - cache_keys),
                })
            for metric, values in feedback_metrics.items():
                delta = values - baseline_metrics[metric]
                rows = _bootstrap(delta)
                metric_rows.append({
                    "panel": panel_name, "method": method, "config": config_name,
                    "metric": relabel.get(metric, metric),
                    "baseline_mean": float(baseline_metrics[metric].mean()),
                    "feedback_mean": float(values.mean()),
                    "delta_mean": rows["mean"],
                    "delta_mean_ci_lower": rows["mean_ci_lower"],
                    "delta_mean_ci_upper": rows["mean_ci_upper"],
                    "delta_median": rows["median"],
                    "delta_median_ci_lower": rows["median_ci_lower"],
                    "delta_median_ci_upper": rows["median_ci_upper"],
                    "wilcoxon_p": _p_value(delta),
                    "pct_top1_changed": float((top[:, 0] != baseline_top[:, 0]).mean()),
                    "n_queries": int(len(delta)),
                })
            saturation_rows.append({
                "panel": panel_name, "method": method, "config": config_name,
                **_saturation(tensor, lift_config, routing),
            })

        panel_summaries[panel_name] = {
            "context_id": context.context_id,
            "history_id": context.history_id,
            "group_rule_id": context.group_rule_id,
            "scenario": context.config.scenario,
            "metadata_policy": context.config.metadata_policy,
            "split": {"path": context.config.split_path,
                      "sha256": context.config.split_sha256},
            "groups": {"path": context.config.groups_path,
                       "sha256": context.config.groups_sha256},
            "n_queries": len(context.query_ids),
            "n_history": len(context.history_ids),
            "n_feedback_rows": int(len(bundle.rows)),
            "feedback_prior_global": bundle.prior_mean("global"),
            "feedback_audit": feedback_audit,
            "n_pool_candidates": int(pools["seq_id"].nunique()),
            "mean_pool_size": float(pools.groupby("query_id").size().mean()),
            "overlap": {row.family: float(row.fraction_queries)
                        for row in overlap.itertuples(index=False)},
        }
        print(f"[{panel_name}] queries={len(context.query_ids)} history={len(context.history_ids)} "
              f"feedback_rows={len(bundle.rows)} prior={bundle.prior_mean('global'):.3f} "
              f"shared_group={panel_summaries[panel_name]['overlap']['v2_group']:.3f}")

    run_id = make_run_id(f"retrieval_panels_{args.protocol}"
                         + ("" if args.split_seed == 42 else f"_seed{args.split_seed}"))
    out_dir = paths.results / "revision" / "retrieval_panels" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "panel_metrics.csv", index=False)
    pd.DataFrame(saturation_rows).to_csv(out_dir / "saturation.csv", index=False)
    pd.concat(overlap_rows, ignore_index=True).to_csv(out_dir / "pool_overlap_audit.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "scope_coverage.csv", index=False)
    if budget_rows:
        pd.DataFrame(budget_rows).to_csv(out_dir / "generation_budget.csv", index=False)

    summary = {
        "run_id": run_id,
        "protocol": args.protocol,
        "search_k": args.search_k,
        "top_k": args.top_k,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "retriever": "dense_minilm_history_only",
        "metric_note": (
            "Retrieval-level reply-similarity proxy (multi-qa-MiniLM-L6-cos-v1), "
            "not generated-answer quality."
        ),
        "panels": panel_summaries,
        "grid": [{"method": m, "config": c, "routing": r.name, "lift": l.name,
                  "scale_mode": l.scale_mode, "multiplier": l.multiplier, "cap": l.cap,
                  "prior_strength": l.prior_strength, "pool_lambda": l.pool_lambda}
                 for m, c, l, r in GRID],
        "inputs": {
            "dataset": {"path": str(paths.dataset_parquet), "sha256": file_sha256(paths.dataset_parquet)},
            "feedback": {"path": str(feedback_path), "sha256": file_sha256(feedback_path)},
            # The proxy matrix is cached and can be rebuilt; pin both files so a
            # rebuilt matrix is visible in provenance rather than silent.
            "reply_embeddings": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in {
                    "minilm_vectors": paths.root / "data" / "cache" / "reply_emb_minilm.npy",
                    "minilm_ids": paths.root / "data" / "cache" / "reply_emb_minilm_ids.json",
                }.items() if path.exists()
            },
        },
        "git": git_info(),
    }
    write_once_json(out_dir / "summary.json", summary)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
