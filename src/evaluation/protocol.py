"""
Per-ticket evaluation protocol.
Orchestrates baseline + feedback retrieval + generation + metric computation.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

from ..config import EvalConfig
from ..retrieval.search import (
    retrieve_baseline,
    retrieve_feedback,
    compute_overlap,
    compute_retrieval_pool_features,
)
from ..generation.prompts import build_generation_prompt, SYSTEM_PROMPT
from ..evaluation.metrics import compute_all_metrics, compute_delta

log = logging.getLogger(__name__)


async def evaluate_one_ticket(
    query_id: str,
    query_row,
    faiss_index,
    encoder,
    feedback_scores: dict,
    config: EvalConfig,
    llm_client,
) -> dict:
    query_title = str(query_row["Title_anon"])
    query_desc = str(query_row.get("Description_anon", "") or "")
    reference_reply = str(query_row.get("first_reply", "") or "")
    query_class = str(query_row.get("intent_class", "") or "other")
    query_team = str(query_row.get("Team->Name", "") or "Unknown")
    query_embedding = encoder.encode_ticket(query_title, query_desc)

    exclude_idxs = {faiss_index.id_to_index(query_id)}
    if None in exclude_idxs:
        exclude_idxs.discard(None)

    baseline_df = retrieve_baseline(
        faiss_index, query_embedding, config.top_k, exclude_idxs
    )
    feedback_df = retrieve_feedback(
        faiss_index, query_embedding, config.top_k, exclude_idxs,
        feedback_scores, config, query_class, query_team, config.search_k,
    )

    bl_candidates = baseline_df.to_dict("records")
    fb_candidates = feedback_df.to_dict("records")

    bl_prompt = build_generation_prompt(query_title, query_desc, bl_candidates)
    fb_prompt = build_generation_prompt(query_title, query_desc, fb_candidates)

    bl_answer, fb_answer = await asyncio.gather(
        llm_client.complete(bl_prompt, system=SYSTEM_PROMPT),
        llm_client.complete(fb_prompt, system=SYSTEM_PROMPT),
    )

    bl_metrics = compute_all_metrics(bl_answer, reference_reply)
    fb_metrics = compute_all_metrics(fb_answer, reference_reply)
    deltas = compute_delta(bl_metrics, fb_metrics)

    pool_features = compute_retrieval_pool_features(baseline_df, feedback_df)

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
                {
                    "seq_id": str(r.get("seq_id", "")),
                    "faiss_score": float(r.get("faiss_score", 0.0)),
                    "faiss_rank": int(r.get("faiss_rank", 0)),
                    "title": str(r.get("Title_anon", "")),
                    "reply": str(r.get("first_reply", "")),
                }
                for r in bl_candidates
            ],
        },
        "feedback": {
            "answer": fb_answer,
            "metrics": fb_metrics,
            "retrieval": [
                {
                    "seq_id": str(r.get("seq_id", "")),
                    "faiss_score": float(r.get("faiss_score", 0.0)),
                    "faiss_rank": int(r.get("faiss_rank", 0)),
                    "enhanced_score": float(r.get("enhanced_score", 0.0)),
                    "feedback_lift": float(r.get("feedback_lift", 0.0)),
                    "feedback_rank": int(r.get("feedback_rank", 0)),
                    "title": str(r.get("Title_anon", "")),
                    "reply": str(r.get("first_reply", "")),
                }
                for r in fb_candidates
            ],
        },
        "deltas": deltas,
        "pool_features": pool_features,
        "retrieval_overlap": compute_overlap(baseline_df, feedback_df),
        "gate_active": bool(feedback_df["gate_active"].iloc[0]) if "gate_active" in feedback_df.columns else False,
    }