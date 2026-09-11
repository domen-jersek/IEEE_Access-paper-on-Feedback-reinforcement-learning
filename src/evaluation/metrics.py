"""
Response quality metrics: cosine similarity, ROUGE-L, BERTScore.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util as sent_util

log = logging.getLogger(__name__)

_COS_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
_cos_model: Optional[SentenceTransformer] = None


def _get_cos_model() -> SentenceTransformer:
    global _cos_model
    if _cos_model is None:
        log.info("Loading cosine-sim model: %s", _COS_MODEL_NAME)
        _cos_model = SentenceTransformer(_COS_MODEL_NAME)
    return _cos_model


def cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_cos_model()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(sent_util.cos_sim(emb[0:1], emb[1:2])[0][0])


def rouge_l_f1(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(text_a, text_b)
    return float(scores["rougeL"].fmeasure)


def compute_all_metrics(
    generated: str, reference: str
) -> dict[str, float]:
    cos = cosine_similarity(generated, reference)
    rouge = rouge_l_f1(generated, reference)
    return {
        "cosine": float(cos),
        "rouge_l": float(rouge),
        "length": float(len(generated)) if generated else 0.0,
    }


def compute_delta(baseline_metrics: dict, feedback_metrics: dict) -> dict:
    return {
        "delta_cosine": feedback_metrics.get("cosine", 0.0) - baseline_metrics.get("cosine", 0.0),
        "delta_rouge_l": feedback_metrics.get("rouge_l", 0.0) - baseline_metrics.get("rouge_l", 0.0),
    }