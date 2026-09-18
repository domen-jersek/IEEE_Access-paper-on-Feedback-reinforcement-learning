"""
Response quality metrics: cosine similarity, ROUGE-L, and (P1.2) independent-family
metrics: BERTScore F1 and cosine under a non-MiniLM embedder (BAAI/bge-base-en-v1.5).

metric_set="core"      -> {cosine, rouge_l, length}                 (SIKDD replication)
metric_set="extended"  -> core + {cosine_bge, bertscore_f1}
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util as sent_util

from ..config import ALT_METRIC_EMBEDDER

os.environ.setdefault("USE_TF", "0")

log = logging.getLogger(__name__)

_COS_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
_cos_model: Optional[SentenceTransformer] = None
_alt_model: Optional[SentenceTransformer] = None
_bertscore = None
BERTSCORE_MODEL = "roberta-large"  # bert_score default for lang="en"


def _get_cos_model() -> SentenceTransformer:
    global _cos_model
    if _cos_model is None:
        log.info("Loading cosine-sim model: %s", _COS_MODEL_NAME)
        _cos_model = SentenceTransformer(_COS_MODEL_NAME)
    return _cos_model


def _get_alt_model() -> SentenceTransformer:
    global _alt_model
    if _alt_model is None:
        log.info("Loading alternative cosine-sim model: %s", ALT_METRIC_EMBEDDER)
        _alt_model = SentenceTransformer(ALT_METRIC_EMBEDDER)
    return _alt_model


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        from bert_score import BERTScorer
        log.info("Loading BERTScore model: %s", BERTSCORE_MODEL)
        _bertscore = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", rescale_with_baseline=True, device="cpu")
    return _bertscore


def cosine_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_cos_model()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(sent_util.cos_sim(emb[0:1], emb[1:2])[0][0])


def cosine_similarity_alt(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_alt_model()
    emb = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(sent_util.cos_sim(emb[0:1], emb[1:2])[0][0])


def rouge_l_f1(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(text_a, text_b)
    return float(scores["rougeL"].fmeasure)


def bertscore_f1(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    _, _, f = _get_bertscore().score([text_a], [text_b])
    return float(f[0])


def compute_all_metrics(
    generated: str, reference: str, metric_set: str = "core"
) -> dict[str, float]:
    cos = cosine_similarity(generated, reference)
    rouge = rouge_l_f1(generated, reference)
    out = {
        "cosine": float(cos),
        "rouge_l": float(rouge),
        "length": float(len(generated)) if generated else 0.0,
    }
    if metric_set == "extended":
        out["cosine_bge"] = cosine_similarity_alt(generated, reference)
        out["bertscore_f1"] = bertscore_f1(generated, reference)
    return out


def compute_delta(baseline_metrics: dict, feedback_metrics: dict) -> dict:
    d = {
        "delta_cosine": feedback_metrics.get("cosine", 0.0) - baseline_metrics.get("cosine", 0.0),
        "delta_rouge_l": feedback_metrics.get("rouge_l", 0.0) - baseline_metrics.get("rouge_l", 0.0),
    }
    for k in ("cosine_bge", "bertscore_f1"):
        if k in feedback_metrics and k in baseline_metrics:
            d[f"delta_{k}"] = feedback_metrics[k] - baseline_metrics[k]
    return d


# ---------------------------------------------------------------------------
# Batched helpers for re-scoring stored answers (08_rescore.py)
# ---------------------------------------------------------------------------

def batch_cosine(gens: list[str], refs: list[str], model: str = "minilm", batch_size: int = 64) -> np.ndarray:
    m = _get_cos_model() if model == "minilm" else _get_alt_model()
    g = m.encode(gens, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    r = m.encode(refs, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    sims = (g * r).sum(axis=1)
    empty = np.array([not a.strip() or not b.strip() for a, b in zip(gens, refs)])
    sims[empty] = 0.0
    return sims.astype(float)


def batch_bertscore(gens: list[str], refs: list[str], batch_size: int = 16) -> np.ndarray:
    scorer = _get_bertscore()
    _, _, f = scorer.score(gens, refs, batch_size=batch_size)
    out = f.numpy().astype(float)
    empty = np.array([not a.strip() or not b.strip() for a, b in zip(gens, refs)])
    out[empty] = 0.0
    return out
