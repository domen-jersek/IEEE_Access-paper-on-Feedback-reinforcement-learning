#!/usr/bin/env python3
"""23_revision_evaluate.py — generated answers under the revision data protocol.

Thin entry point around the existing ``EvaluationRunner``. It differs from
``04_evaluate.py`` only in how the data are resolved:

* candidates come from a history-only index (train partition of a pinned regime);
* feedback rows use both endpoints in history, no self-pairs, priors from the
  same rows;
* the evidence source can be the LLM judge (conditioned/blind) or reply
  similarity (exact / BGE embedding), on exactly the same historical pairs.

The frozen finalist (selected on dev, see 22_revision_dev_lab.py) is
``M4_intersection`` with ``laplace_eb``, kappa=2, absolute scaling. Do not change
these defaults after the human-evaluation freeze.

Examples:
    python experiments/23_revision_evaluate.py --panel disjoint_prefix --source conditioned --limit 3
    python experiments/23_revision_evaluate.py --panel disjoint_prefix --source blind
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.utils import (  # noqa: E402
    append_registry, cache_dir, make_run_id, set_seeds, setup_logging, write_run_manifest,
)
from src.config import DEFAULT_METHODS, EvalConfig, GatingConfig, LiftConfig, ProjectPaths, RoutingConfig
from src.evaluation.revision_panels import load_feedback_rows, panel_specs, resolve_panel
from src.evaluation.runner import EvaluationRunner
from src.feedback.loader import FeedbackBundle
from src.feedback.reply_similarity import with_reply_similarity
from src.generation.regime import generation_regime
from src.retrieval.encoder import TicketEncoder
from src.retrieval.history import build_history_retriever

SOURCES = ("conditioned", "blind", "reply_exact", "reply_embed_bge")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--source", required=True, choices=list(SOURCES))
    parser.add_argument("--split", default="eval", choices=["dev", "eval"])
    parser.add_argument("--method", default="M4_intersection", choices=["M4_intersection", "baseline"])
    parser.add_argument("--search-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--notes", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(20 if args.verbose else 30)
    set_seeds(42)

    paths = ProjectPaths()
    import pandas as pd

    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)

    specs = panel_specs(paths, 42)
    if args.panel not in specs:
        raise SystemExit(f"Unknown panel {args.panel!r}; available: {sorted(specs)}")
    panel = specs[args.panel]
    context = resolve_panel(panel, paths, partition=args.split)

    retriever = build_history_retriever("dense_minilm", dataset, set(context.history_ids))
    encoder = TicketEncoder("all-MiniLM-L6-v2")

    if args.source in {"conditioned", "blind"}:
        rows = load_feedback_rows(paths.data_processed / f"feedback_{args.source}.db")
        allowed, feedback_audit = context.select_feedback(rows, protocol=args.source)
        bundle = FeedbackBundle(allowed)
    else:
        rows = load_feedback_rows(paths.data_processed / "feedback_conditioned.db")
        allowed, feedback_audit = context.select_feedback(rows, protocol="conditioned")
        bundle = FeedbackBundle(with_reply_similarity(
            allowed, dataset, args.source.removeprefix("reply_"), cache_dir=cache_dir()))
    feedback_scores = bundle.scores
    priors = bundle

    base = DEFAULT_METHODS[args.method]
    if args.method == "baseline":
        lift = LiftConfig(name="none", multiplier=0.0, cap=0.0)
        routing = RoutingConfig(name="none")
    else:
        lift = LiftConfig.laplace_eb(prior_strength=2.0)
        routing = RoutingConfig.intersection()
    folder = f"{args.method}_{args.split}_{args.source}_rev_{args.panel}"
    config = EvalConfig(
        experiment_id=folder, lift=lift, routing=routing, gating=GatingConfig.none(),
        generator_model=base.generator_model, judge_model=base.judge_model,
        top_k=args.top_k, search_k=args.search_k, regime="revision",
    )

    query_ids = list(context.query_ids)
    if args.limit:
        query_ids = query_ids[: args.limit]
    if not query_ids:
        raise SystemExit("Empty query partition")

    results_dir = paths.results / folder
    cache_path = None if args.no_cache else cache_dir() / "generation_cache.db"
    run_id = make_run_id(folder)
    regime = generation_regime(model=config.generator_model, cache_path=cache_path,
                               temperature=config.temperature)

    manifest = write_run_manifest(
        results_dir, script="experiments/23_revision_evaluate.py", args=args,
        config=config.to_dict(),
        inputs=[paths.dataset_parquet, paths.root / panel["split"], paths.root / panel["groups"],
                paths.data_processed / ("feedback_conditioned.db" if args.source.startswith("reply")
                                        else f"feedback_{args.source}.db")],
        extra={
            "context_id": context.context_id,
            "history_id": context.history_id,
            "panel": args.panel,
            "source": args.source,
            "n_queries": len(query_ids),
            "n_history": len(context.history_ids),
            "feedback_audit": feedback_audit,
            "feedback_prior_global": bundle.prior_mean("global"),
            "config_hash": config.config_hash,
            "generation_regime": regime,
        },
        run_id=run_id,
    )

    runner = EvaluationRunner(
        config=config, faiss_index=retriever, dataset=dataset, encoder=encoder,
        feedback_scores=feedback_scores, results_dir=results_dir,
        concurrency=args.concurrency, interim_every=10,
        cache_path=cache_path, priors=priors, retriever_name="dense_minilm_history_only",
    )
    summary = asyncio.run(runner.run(query_ids))
    append_registry(
        run_id=run_id, phase="revision-generation", script="23_revision_evaluate.py",
        out_dir=results_dir, manifest=manifest,
        headline_metric="mean_delta_cosine",
        headline_value=round(summary.get("mean_delta_cosine", 0.0), 5),
        notes=args.notes or f"panel={args.panel};source={args.source};details={runner.last_paths['details'].name}",
    )
    print(f"context={context.context_id[:12]} history={len(context.history_ids)} "
          f"queries={len(query_ids)} prior={bundle.prior_mean('global'):.3f}")
    print(f"cache hits={runner.llm_client.cache_hits} misses={runner.llm_client.cache_misses} "
          f"warm={summary.get('generation_regime', {}).get('warm')}")
    print(f"mean_delta_cosine={summary.get('mean_delta_cosine'):+.4f} results={results_dir}")


if __name__ == "__main__":
    main()
