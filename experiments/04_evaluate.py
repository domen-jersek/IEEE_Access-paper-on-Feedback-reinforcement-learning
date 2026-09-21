#!/usr/bin/env python3
"""
04_evaluate.py — Phase 3
==========================
Runs the LOO evaluation pipeline on dev queries (or train/eval queries).
Supports the SIKDD methods (baseline, M1-M4) plus the P3 granularity methods
(M5_backoff, M6_blend), alternative retrievers (P2), calibrated lifts (P1.5),
and a switchable generator model (economic-parity runs).

Default invocation reproduces the SIKDD replication runs exactly:
    python experiments/04_evaluate.py --method M2_team --split dev --seed 42

New flags (all optional; defaults keep folder names and numerics unchanged):
    --generator-model openai/gpt-4o-mini      generator (folder gets a _gen<name> suffix)
    --retriever hybrid_rrf                    candidate-pool retriever (see src/config.py RETRIEVERS)
    --lift laplace_eb --prior-strength 2      calibrated lift (P1.5)
    --scale-mode pool_std --pool-lambda 1.0   lift in pool-std units (needed for non-cosine retrievers)
    --blend-weights results/blend/learned_weights.json   weights for M6_blend
    --min-evidence 3                          backoff threshold for M5_backoff
    --search-k 100 --top-k 5                  pool / context sizes
    --limit 5                                 smoke test on the first N queries of the split
    --metric-set extended                     adds cosine_bge + bertscore_f1 (slow on CPU)
    --tag mytag                               free-form folder suffix
    --no-cache                                disable the generation cache
Every run writes manifest.json in its results folder and appends to results/registry.csv.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (  # noqa: E402  (sets USE_TF=0 before transformers import)
    setup_logging, load_dataset, load_split, split_path, get_arg_parser,
    set_seeds, write_run_manifest, append_registry, short_model_name, cache_dir, make_run_id,
)
from src.config import ProjectPaths, DEFAULT_METHODS, EvalConfig, LiftConfig, RoutingConfig, RETRIEVERS
from src.retrieval.encoder import TicketEncoder
from src.retrieval.index import FAISSIndex
from src.retrieval.retrievers import build_retriever
from src.retrieval.semantic import TicketTextSimilarity
from src.feedback.loader import load_feedback_bundle
from src.evaluation.runner import EvaluationRunner


def build_config(args, base_cfg: EvalConfig) -> tuple[EvalConfig, str]:
    """Return (config, results folder name). Defaults reproduce the legacy naming."""
    lift = base_cfg.lift
    routing = base_cfg.routing
    suffix_parts = []

    if args.method != "baseline":
        if args.lift != "laplace":
            if args.lift == "laplace_eb":
                lift = LiftConfig.laplace_eb(prior_strength=args.prior_strength)
            else:
                lift = replace(LiftConfig.laplace(), name=args.lift)
            suffix_parts.append(f"lift{args.lift}")
            if args.lift == "laplace_eb" and args.prior_strength != 2.0:
                suffix_parts.append(f"k{args.prior_strength:g}")
        if args.scale_mode != "absolute":
            lift = replace(lift, scale_mode=args.scale_mode, pool_lambda=args.pool_lambda)
            suffix_parts.append(f"{args.scale_mode}{args.pool_lambda:g}")
        if args.method == "M5_backoff" and args.min_evidence != 3.0:
            routing = replace(routing, min_evidence=args.min_evidence)
            suffix_parts.append(f"minev{args.min_evidence:g}")
        if args.method == "M6_blend" and args.blend_weights:
            w = json.loads(Path(args.blend_weights).read_text("utf-8"))
            w = w.get("weights", w)
            routing = RoutingConfig.blend(
                w_global=float(w.get("global", 0.0)), w_class=float(w.get("class", 0.0)),
                w_team=float(w.get("team", 0.0)), w_intersection=float(w.get("intersection", 0.0)),
            )
            suffix_parts.append("wlearned")
        if args.method in ("M7_semantic_intersection", "M8_semantic_backoff") and args.semantic_tau is not None:
            if args.method == "M7_semantic_intersection":
                routing = RoutingConfig.semantic_intersection(tau=args.semantic_tau)
            else:
                routing = RoutingConfig.semantic_backoff(tau=args.semantic_tau, min_evidence=args.min_evidence)
            suffix_parts.append(f"tau{args.semantic_tau:g}")

    if args.retriever != "dense_minilm":
        suffix_parts.append(args.retriever)
    gen_model = args.generator_model or base_cfg.generator_model
    if gen_model != base_cfg.generator_model:
        suffix_parts.append(f"gen{short_model_name(gen_model)}")
    if args.metric_set != "core":
        suffix_parts.append(args.metric_set)
    if args.regime != "random":
        suffix_parts.append(args.regime)
    if args.seed != 42:
        suffix_parts.append(f"seed{args.seed}")
    if args.tag:
        suffix_parts.append(args.tag)

    folder = f"{args.method}_{args.split}_{args.feedback_protocol}_{args.agg_mode}"
    if suffix_parts:
        folder += "_" + "_".join(suffix_parts)

    config = EvalConfig(
        experiment_id=folder,
        lift=lift,
        routing=routing,
        gating=base_cfg.gating,
        generator_model=gen_model,
        judge_model=base_cfg.judge_model,
        regime=args.regime,
        seed=args.seed,
        top_k=args.top_k,
        search_k=args.search_k,
        retriever=args.retriever,
        metric_set=args.metric_set,
    )
    return config, folder


async def main_async() -> None:
    parser = get_arg_parser("Evaluation runner")
    parser.add_argument("--method", type=str, required=True, choices=list(DEFAULT_METHODS.keys()))
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "eval"])
    parser.add_argument("--feedback-protocol", type=str, default="conditioned", choices=["conditioned", "blind"])
    parser.add_argument("--agg-mode", type=str, default="continuous", choices=["continuous", "binary"],
                        help="How to aggregate raw judge scores into pos/neg lifts")
    # --- new, all optional ---
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument("--retriever", type=str, default="dense_minilm", choices=list(RETRIEVERS))
    parser.add_argument("--lift", type=str, default="laplace", choices=["laplace", "laplace_eb", "tanh", "bayesian_lcb"])
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--scale-mode", type=str, default="absolute", choices=["absolute", "pool_std"])
    parser.add_argument("--pool-lambda", type=float, default=1.0)
    parser.add_argument("--min-evidence", type=float, default=3.0)
    parser.add_argument("--blend-weights", type=str, default=None)
    parser.add_argument("--semantic-tau", type=float, default=None,
                        help="Query<->candidate text cosine threshold for M7/M8 semantic routings")
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--metric-set", type=str, default="core", choices=["core", "extended"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    log = logging.getLogger("evaluate")
    set_seeds(args.seed)

    paths = ProjectPaths()
    log.info("=== Loading dataset + split ===")
    df = load_dataset()
    split = load_split(seed=args.seed, regime=args.regime)
    query_ids = split[args.split]
    if args.limit:
        query_ids = query_ids[: args.limit]
    log.info("Evaluating %s: %d queries (agg: %s)", args.split, len(query_ids), args.agg_mode)

    log.info("=== Loading FAISS index ===")
    index_dir = paths.data_processed / "faiss_index"
    faiss_idx = FAISSIndex(dim=384)
    faiss_idx.load(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")

    log.info("=== Loading encoder ===")
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    retrieval_source = faiss_idx  # legacy path == SIKDD numerics
    if args.retriever != "dense_minilm":
        log.info("=== Building retriever: %s ===", args.retriever)
        retrieval_source = build_retriever(args.retriever, paths, faiss_idx, encoder, ce_pool=args.search_k)

    log.info("=== Loading feedback scores ===")
    fb_db_path = paths.data_processed / f"feedback_{args.feedback_protocol}.db"
    priors = None
    if not fb_db_path.exists():
        log.warning("Feedback DB not found at %s, proceeding with zero feedback (baseline-only)", fb_db_path)
        feedback_scores = {}
    else:
        # Excludes the evaluated split's query IDs (leakage guard) — identical to the legacy loader.
        bundle = load_feedback_bundle(fb_db_path, exclude_query_ids=set(split[args.split]), mode=args.agg_mode)
        feedback_scores = bundle.scores
        priors = bundle
        log.info("Loaded feedback scores for %d candidates (global prior mean=%.3f)",
                 len(feedback_scores), bundle.prior_mean("global"))

    log.info("=== Configuring evaluation ===")
    base_cfg = DEFAULT_METHODS[args.method]
    config, folder = build_config(args, base_cfg)
    text_sim = TicketTextSimilarity(df, encoder=encoder) if config.routing.name.startswith("semantic") else None
    results_dir = paths.results / folder
    cache_path = None if args.no_cache else cache_dir() / "generation_cache.db"
    run_id = make_run_id(folder)

    manifest = write_run_manifest(
        results_dir, script="experiments/04_evaluate.py", args=args, config=config.to_dict(),
        inputs=[paths.dataset_parquet, split_path(args.seed, args.regime), fb_db_path,
                index_dir / "faiss.index", index_dir / "faiss_metadata.parquet"],
        extra={"n_queries": len(query_ids), "config_hash": config.config_hash,
               "generation_cache": str(cache_path) if cache_path else None,
               "feedback_priors_global": priors.prior_mean("global") if priors else None},
        run_id=run_id,
    )

    runner = EvaluationRunner(
        config=config,
        faiss_index=retrieval_source,
        dataset=df,
        encoder=encoder,
        feedback_scores=feedback_scores,
        results_dir=results_dir,
        concurrency=args.concurrency,
        interim_every=10,
        cache_path=cache_path,
        priors=priors,
        retriever_name=args.retriever,
        text_sim=text_sim,
    )

    log.info("=== Running evaluation (%s) ===", folder)
    summary = await runner.run(query_ids)
    log.info("Summary: mean_delta=%.4f, pct_improved=%.1f%%  (cache hits=%d misses=%d)",
             summary.get("mean_delta_cosine", 0), summary.get("pct_improved", 0) * 100,
             runner.llm_client.cache_hits, runner.llm_client.cache_misses)

    append_registry(
        run_id=run_id, phase="P4-generation" if not args.limit else "smoke", script="04_evaluate.py",
        out_dir=results_dir, manifest=manifest,
        headline_metric="mean_delta_cosine", headline_value=round(summary.get("mean_delta_cosine", 0.0), 5),
        notes=args.notes or (f"details={runner.last_paths['details'].name}" if runner.last_paths else ""),
    )
    log.info("=== DONE === results: %s", results_dir)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
