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