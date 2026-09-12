"""
FAISS index builder and search.
Supports inner-product (cosine) search over normalized embeddings.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class FAISSIndex:
    def __init__(
        self,
        dim: int,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.dim = dim
        self._index: Optional[faiss.IndexIDMap] = None
        self._meta: Optional[pd.DataFrame] = None

        if index_path and index_path.exists():
            self.load(index_path, metadata_path)

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        log.info("Building FAISS index: %d vectors, dim=%d", len(embeddings), self.dim)
        base_index = faiss.IndexFlatIP(self.dim)
        self._index = faiss.IndexIDMap(base_index)
        ids = np.arange(len(embeddings), dtype=np.int64)
        self._index.add_with_ids(embeddings.astype(np.float32), ids)
        self._meta = metadata.reset_index(drop=True)
        log.info("Index built: %d vectors", self._index.ntotal)

    def search(self, query: np.ndarray, k: int, exclude_idxs: Optional[set[int]] = None) -> tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)

        search_k = k + (len(exclude_idxs) if exclude_idxs else 0) + 1
        search_k = min(search_k, self._index.ntotal)
        scores, indices = self._index.search(query, search_k)

        if exclude_idxs:
            filtered_scores, filtered_indices = [], []
            for i in range(len(scores)):
                row_scores, row_indices = [], []
                for s, idx in zip(scores[i], indices[i]):
                    if idx not in exclude_idxs and idx != -1:
                        row_scores.append(s)
                        row_indices.append(idx)
                        if len(row_indices) >= k:
                            break
                filtered_scores.append(np.array(row_scores[:k], dtype=np.float32))
                filtered_indices.append(np.array(row_indices[:k], dtype=np.int64))
            return np.array(filtered_scores), np.array(filtered_indices)

        return scores[:, :k], indices[:, :k]

    def get_metadata(self, indices: np.ndarray) -> pd.DataFrame:
        flat_idx = indices.flatten()
        valid = flat_idx[flat_idx >= 0]
        rows = self._meta.iloc[valid].copy()
        rows["_faiss_idx"] = valid
        return rows

    def get_metadata_by_id(self, seq_ids: list[str]) -> pd.DataFrame:
        return self._meta[self._meta["seq_id"].isin(seq_ids)].copy()

    def id_to_index(self, seq_id: str) -> Optional[int]:
        matches = self._meta[self._meta["seq_id"] == seq_id]
        if matches.empty:
            return None
        return int(matches.index[0])

    def index_to_id(self, idx: int) -> Optional[str]:
        try:
            return str(self._meta.iloc[idx]["seq_id"])
        except (IndexError, KeyError):
            return None

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index else 0

    def save(self, index_path: Path, metadata_path: Path):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        self._meta.to_parquet(metadata_path, compression="zstd")
        log.info("Index saved to %s (%d vectors)", index_path, self._index.ntotal)

    def load(self, index_path: Path, metadata_path: Optional[Path] = None):
        log.info("Loading FAISS index from %s", index_path)
        self._index = faiss.read_index(str(index_path))
        if metadata_path and metadata_path.exists():
            self._meta = pd.read_parquet(metadata_path)
            log.info("Loaded metadata: %d rows", len(self._meta))

    @staticmethod
    def compute_ticket_similarity(encoder, df: pd.DataFrame, k: int = 1) -> pd.Series:
        from .encoder import TicketEncoder
        embeddings = encoder.encode_batch(
            df["Title_anon"].tolist(),
            df["Description_anon"].fillna("").tolist(),
        )
        temp = faiss.IndexFlatIP(encoder.dim)
        temp_index = faiss.IndexIDMap(temp)
        ids = np.arange(len(df), dtype=np.int64)
        temp_index.add_with_ids(embeddings.astype(np.float32), ids)
        max_sims = []
        for i in range(len(df)):
            q = embeddings[i].reshape(1, -1).astype(np.float32)
            s, idx = temp_index.search(q, k + 1)
            valid = [(score, j) for score, j in zip(s[0], idx[0]) if j != i and j != -1]
            max_sims.append(valid[0][0] if valid else 0.0)
        return pd.Series(max_sims, index=df.index)


def compute_recommended_judging_depth(
    faiss_index,
    encoder,
    query_ids: list[str],
    lift_cap: float = 0.20,
    sample_depth: int = 500,
) -> dict:
    """
    Compute the minimum number of candidates that must be judged per query so
    that feedback can potentially promote candidates into the RAG top-5.

    For each query:
      1. Retrieve top `sample_depth` FAISS candidates (self-excluded).
      2. The RAG generator sees the top 5; the lift cap is ±LIFT_CAP.
         A candidate at rank R can enter the top-5 iff:
             FAISS[R] + LIFT_CAP >= FAISS[4]
         i.e., FAISS[R] >= FAISS[4] - LIFT_CAP
      3. `max_promotable_rank` = largest R where the above holds, capped at
         `sample_depth` (report if capped — it means the KB is flat enough
         that even the deepest-sampled candidates are theoretically promotable).

    NOTE: This metric is an *upper bound*. In practice, a candidate at rank R
    must compete with all candidates at ranks 6..R-1 that also receive lifts,
    so the practical promotion depth is substantially lower. The 95th-percentile
    is reported for reference but may be capped by sample_depth.

    Returns a dict with percentile distribution and the 95th-percentile value
    (may be capped — check whether p95 == sample_depth).
    """

    ranks = []
    capped_count = 0
    for qid in query_ids:
        try:
            row = faiss_index._meta[faiss_index._meta["seq_id"] == qid].iloc[0]
        except (KeyError, IndexError):
            continue
        title = str(row["Title_anon"])
        desc = str(row.get("Description_anon", "") or "")
        emb = encoder.encode_ticket(title, desc)

        exclude = {faiss_index.id_to_index(qid)}
        if None in exclude:
            exclude.discard(None)
        scores, _indices = faiss_index.search(emb, sample_depth, exclude)
        arr = scores[0]

        if len(arr) < 5:
            ranks.append(len(arr))
            continue

        threshold = float(arr[4] - lift_cap)
        promotable = np.where(arr >= threshold)[0]
        max_rank = int(promotable[-1]) + 1 if len(promotable) > 0 else 5
        if max_rank >= sample_depth:
            max_rank = sample_depth
            capped_count += 1
        ranks.append(max_rank)

    ranks_arr = np.array(ranks)
    percentiles = [50, 75, 90, 95, 99]
    result = {
        "lift_cap": lift_cap,
        "sample_depth": sample_depth,
        "n_queries_sampled": int(len(ranks_arr)),
        "n_queries_capped_at_max": int(capped_count),
        "pct_capped": float(capped_count / max(len(ranks_arr), 1)),
        "min_rank": int(np.min(ranks_arr)),
        "max_rank": int(np.max(ranks_arr)),
        "mean_rank": float(np.mean(ranks_arr)),
        "median_rank": float(np.median(ranks_arr)),
    }
    for p in percentiles:
        result[f"p{p}"] = int(np.percentile(ranks_arr, p))

    p95_raw = result["p95"]
    if p95_raw >= sample_depth:
        result["note"] = (
            f"95th percentile ({p95_raw}) is at or above sample_depth ({sample_depth}). "
            "FAISS scores are flat across the KB — even deep candidates are "
            "theoretically promotable. Choose a practical budget (e.g., 200) "
            "rather than a theoretical lower bound."
        )
        result["recommended_judging_depth"] = 200
    else:
        result["recommended_judging_depth"] = p95_raw

    result["ranks"] = ranks_arr.tolist()
    return result