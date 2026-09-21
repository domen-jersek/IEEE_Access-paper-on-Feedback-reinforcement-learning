"""
Per-ticket evaluation protocol.
Orchestrates baseline + feedback retrieval + generation + metric computation.

Record layout is a strict superset of the SIKDD replication records: every key
written by the original protocol is still written with the same semantics; the
P0 additions (query_description, per-scope evidence, pool coverage, prompt hashes,
pool_top20, generator_model, retriever) are extra keys only.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

import numpy as np

from ..config import EvalConfig
from ..retrieval.search import (
    retrieve_baseline,
    retrieve_feedback,
    compute_overlap,
    compute_retrieval_pool_features,
    scope_evidence,
    per_scope_lifts,
    pool_feedback_coverage,
)
from ..generation.prompts import build_generation_prompt, SYSTEM_PROMPT
from ..evaluation.metrics import compute_all_metrics, compute_delta

log = logging.getLogger(__name__)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def _generate_pair(llm_client, baseline_prompt: str, feedback_prompt: str) -> tuple[str, str]:
    """Generate once when retrieval produced identical generation context."""
    if baseline_prompt == feedback_prompt:
        answer = await llm_client.complete(baseline_prompt, system=SYSTEM_PROMPT)
        return answer, answer
    return await asyncio.gather(
        llm_client.complete(baseline_prompt, system=SYSTEM_PROMPT),
        llm_client.complete(feedback_prompt, system=SYSTEM_PROMPT),
    )


def _cand_record(r: dict, feedback_scores: dict, query_class: str, query_team: str,
                 config: EvalConfig, priors, side: str) -> dict:
    cid = str(r.get("seq_id", ""))
    rec = {
        "seq_id": cid,
        "faiss_score": float(r.get("faiss_score", 0.0)),
        "faiss_rank": int(r.get("faiss_rank", 0)),
    }
    if side == "feedback":
        rec.update({
            "enhanced_score": float(r.get("enhanced_score", 0.0)),
            "feedback_lift": float(r.get("feedback_lift", 0.0)),
            "feedback_rank": int(r.get("feedback_rank", 0)),
        })
    rec["title"] = str(r.get("Title_anon", ""))
    rec["reply"] = str(r.get("first_reply", ""))
    # --- P0 additions ---
    cdata = feedback_scores.get(cid, {})
    rec["evidence"] = scope_evidence(cdata, query_class, query_team)
    if config.lift.name != "none":
        rec["lift_by_scope"] = per_scope_lifts(cdata, query_class, query_team, config.lift, priors)
    if side == "feedback":
        rec["feedback_lift_raw"] = float(r.get("feedback_lift_raw", r.get("feedback_lift", 0.0)))
        rec["lift_scope"] = str(r.get("lift_scope", ""))
    return rec


async def evaluate_one_ticket(
    query_id: str,
    query_row,
    faiss_index,
    encoder,
    feedback_scores: dict,
    config: EvalConfig,
    llm_client,
    priors=None,
    retriever_name: str = "dense_minilm",
    text_sim=None,
) -> dict:
    query_title = str(query_row["Title_anon"])
    query_desc = str(query_row.get("Description_anon", "") or "")
    reference_reply = str(query_row.get("first_reply", "") or "")
    query_class = str(query_row.get("intent_class", "") or "other")
    query_team = str(query_row.get("Team->Name", "") or "Unknown")
    query_embedding = encoder.encode_ticket(query_title, query_desc)
    query_text = f"{query_title}\n{query_desc}" if query_desc else query_title

    # Self-exclusion (LOO). Works for FAISSIndex and for retrievers (they expose .index or .meta).
    id_lookup = faiss_index if hasattr(faiss_index, "id_to_index") else getattr(faiss_index, "index", None)
    if id_lookup is None:
        meta = faiss_index.meta
        matches = meta.index[meta["seq_id"] == query_id]
        self_idx = int(matches[0]) if len(matches) else None
    else:
        self_idx = id_lookup.id_to_index(query_id)
    exclude_idxs = {self_idx} if self_idx is not None else set()

    baseline_df = retrieve_baseline(
        faiss_index, query_embedding, config.top_k, exclude_idxs, query_text=query_text,
    )
    feedback_df, pool_df = retrieve_feedback(
        faiss_index, query_embedding, config.top_k, exclude_idxs,
        feedback_scores, config, query_class, query_team, config.search_k,
        query_text=query_text, priors=priors, return_pool=True,
        text_sim=text_sim, query_id=query_id,
    )

    bl_candidates = baseline_df.to_dict("records")
    fb_candidates = feedback_df.to_dict("records")

    bl_prompt = build_generation_prompt(query_title, query_desc, bl_candidates)

    is_baseline = (config.lift.name == "none")

    if is_baseline:
        fb_prompt = bl_prompt
    else:
        fb_prompt = build_generation_prompt(query_title, query_desc, fb_candidates)
    bl_answer, fb_answer = await _generate_pair(llm_client, bl_prompt, fb_prompt)

    bl_metrics = compute_all_metrics(bl_answer, reference_reply, metric_set=config.metric_set)
    fb_metrics = compute_all_metrics(fb_answer, reference_reply, metric_set=config.metric_set)
    deltas = compute_delta(bl_metrics, fb_metrics)

    pool_features = compute_retrieval_pool_features(baseline_df, feedback_df)

    pool_top20 = [
        {
            "seq_id": str(r["seq_id"]),
            "faiss_score": float(r["faiss_score"]),
            "faiss_rank": int(r.get("faiss_rank", 0)),
            "feedback_lift": float(r.get("feedback_lift", 0.0)),
            "enhanced_score": float(r.get("enhanced_score", 0.0)),
            "lift_scope": str(r.get("lift_scope", "")),
        }
        for r in pool_df.sort_values("enhanced_score", ascending=False, kind="stable").head(20).to_dict("records")
    ]

    return {
        "ticket_id": query_id,
        "query_title": query_title,
        "expected_team": query_team,
        "expected_class": query_class,
        "reference_reply": reference_reply,
        "baseline": {
            "answer": bl_answer,
            "metrics": bl_metrics,
            "retrieval": [
                _cand_record(r, feedback_scores, query_class, query_team, config, priors, "baseline")
                for r in bl_candidates
            ],
        },
        "feedback": {
            "answer": fb_answer,
            "metrics": fb_metrics,
            "retrieval": [
                _cand_record(r, feedback_scores, query_class, query_team, config, priors, "feedback")
                for r in fb_candidates
            ],
        },
        "deltas": deltas,
        "pool_features": pool_features,
        "retrieval_overlap": compute_overlap(baseline_df, feedback_df),
        "gate_active": bool(feedback_df["gate_active"].iloc[0]) if "gate_active" in feedback_df.columns else False,
        # --- P0 additions ---
        "query_description": query_desc,
        "generator_model": config.generator_model,
        "retriever": retriever_name,
        "prompt_sha256": {"baseline": _sha(bl_prompt), "feedback": _sha(fb_prompt), "system": _sha(SYSTEM_PROMPT)},
        "pool_size": int(len(pool_df)),
        "pool_feedback_coverage": pool_feedback_coverage(pool_df, feedback_scores, query_class, query_team),
        "lift_scale_factor": float(pool_df.attrs.get("lift_scale_factor", 1.0)),
        "pool_top20": pool_top20,
    }
