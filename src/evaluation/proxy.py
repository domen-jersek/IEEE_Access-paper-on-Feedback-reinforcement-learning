"""
Offline retrieval-level proxy metrics (P1.1).

The reference reply of query q is `first_reply[q]`, and every candidate c is a KB
ticket with `first_reply[c]`. So the "would this retrieved reply have been useful"
signal is the similarity between two rows of the SAME reply column — a 1595x1595
matrix that is computed once per embedding model and cached in data/cache.

Metrics per (query, ranked candidate list)
------------------------------------------
proxy_top1        sim(reply[c_1], reply[q])
proxy_best5       max_{i<=5} sim(reply[c_i], reply[q])
proxy_mean5       mean_{i<=5} sim(...)
proxy_rr          1/rank of the first candidate with sim >= tau (0 if none in top-k)
same_reply_hit1   reply_hash[c_1] == reply_hash[q]      (embedding-free)
same_reply_hit5   any_{i<=5} reply_hash[c_i] == reply_hash[q]
group_hit1/5      combined_group_hash equality          (embedding-free)

Two embedding families are supported: "minilm" (= the SIKDD answer metric model
multi-qa-MiniLM-L6-cos-v1) and "bge" (BAAI/bge-base-en-v1.5, independent family).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..config import ProjectPaths, ALT_METRIC_EMBEDDER

os.environ.setdefault("USE_TF", "0")
log = logging.getLogger(__name__)

PROXY_MODELS = {
    "minilm": "multi-qa-MiniLM-L6-cos-v1",
    "bge": ALT_METRIC_EMBEDDER,
}


class ReplySimilarity:
    """Cached reply-to-reply cosine matrix over the canonical dataset."""

    def __init__(self, dataset: pd.DataFrame, model_tag: str = "minilm", cache_dir: Optional[Path] = None):
        self.model_tag = model_tag
        self.model_name = PROXY_MODELS[model_tag]
        self.ids = dataset["seq_id"].astype(str).tolist()
        self.pos = {s: i for i, s in enumerate(self.ids)}
        self.replies = dataset["first_reply"].fillna("").astype(str).tolist()
        self.reply_hash = dict(zip(self.ids, dataset["reply_hash"].astype(str)))
        self.group_hash = dict(zip(self.ids, dataset["combined_group_hash"].astype(str)))
        cache_dir = cache_dir or (ProjectPaths().root / "data" / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._emb_path = cache_dir / f"reply_emb_{model_tag}.npy"
        self._ids_path = cache_dir / f"reply_emb_{model_tag}_ids.json"
        self.emb = self._load_or_build()
        self.sim = self.emb @ self.emb.T

    def _load_or_build(self) -> np.ndarray:
        if self._emb_path.exists() and self._ids_path.exists():
            ids = json.loads(self._ids_path.read_text("utf-8"))
            if ids == self.ids:
                return np.load(self._emb_path)
            log.warning("Reply-embedding cache id mismatch; rebuilding %s", self._emb_path)
        from sentence_transformers import SentenceTransformer
        log.info("Encoding %d replies with %s (one-off, cached)", len(self.replies), self.model_name)
        model = SentenceTransformer(self.model_name, device="cpu")
        emb = model.encode(self.replies, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        emb = emb.astype(np.float32)
        np.save(self._emb_path, emb)
        self._ids_path.write_text(json.dumps(self.ids), encoding="utf-8")
        return emb

    def s(self, query_id: str, cand_id: str) -> float:
        return float(self.sim[self.pos[query_id], self.pos[cand_id]])

    def metrics(self, query_id: str, ranked_ids: Sequence[str], k: int = 5, tau: float = 0.8) -> dict:
        ranked = [c for c in ranked_ids[:k] if c in self.pos]
        if not ranked:
            return {"proxy_top1": 0.0, "proxy_best5": 0.0, "proxy_mean5": 0.0, "proxy_rr": 0.0,
                    "same_reply_hit1": 0.0, "same_reply_hit5": 0.0, "group_hit1": 0.0, "group_hit5": 0.0}
        sims = np.array([self.s(query_id, c) for c in ranked])
        rr = 0.0
        hits = np.where(sims >= tau)[0]
        if len(hits):
            rr = 1.0 / (hits[0] + 1)
        qh, qg = self.reply_hash[query_id], self.group_hash[query_id]
        same = [self.reply_hash[c] == qh for c in ranked]
        grp = [self.group_hash[c] == qg for c in ranked]
        return {
            "proxy_top1": float(sims[0]),
            "proxy_best5": float(sims.max()),
            "proxy_mean5": float(sims.mean()),
            "proxy_rr": float(rr),
            "same_reply_hit1": float(same[0]),
            "same_reply_hit5": float(any(same)),
            "group_hit1": float(grp[0]),
            "group_hit5": float(any(grp)),
        }

    def delta(self, query_id: str, baseline_ids: Sequence[str], feedback_ids: Sequence[str], k: int = 5) -> dict:
        b = self.metrics(query_id, baseline_ids, k)
        f = self.metrics(query_id, feedback_ids, k)
        out = {f"base_{k_}": v for k_, v in b.items()}
        out.update({f"fb_{k_}": v for k_, v in f.items()})
        out.update({f"d_{k_}": f[k_] - b[k_] for k_ in b})
        return out


def proxy_frame_from_details(recs: list[dict], sims: dict[str, ReplySimilarity], k: int = 5) -> pd.DataFrame:
    """One row per ticket: generated deltas + proxy deltas under each similarity model."""
    rows = []
    for r in recs:
        qid = r["ticket_id"]
        bl = [c["seq_id"] for c in r["baseline"]["retrieval"]]
        fb = [c["seq_id"] for c in r["feedback"]["retrieval"]]
        row = {
            "ticket_id": qid,
            "expected_team": r.get("expected_team"),
            "expected_class": r.get("expected_class"),
            "delta_cosine": r["deltas"]["delta_cosine"],
            "delta_rouge_l": r["deltas"].get("delta_rouge_l"),
            "baseline_cosine": r["baseline"]["metrics"]["cosine"],
            "top1_changed": float(bl[0] != fb[0]) if bl and fb else 0.0,
            "retrieval_overlap": r.get("retrieval_overlap"),
            "top1_faiss": r.get("pool_features", {}).get("top1_faiss"),
        }
        for k_ in ("delta_cosine_bge", "delta_bertscore_f1"):
            if k_ in r["deltas"]:
                row[k_] = r["deltas"][k_]
        for tag, sim in sims.items():
            d = sim.delta(qid, bl, fb, k)
            row.update({f"{tag}_{kk}": v for kk, v in d.items()})
        rows.append(row)
    return pd.DataFrame(rows)
