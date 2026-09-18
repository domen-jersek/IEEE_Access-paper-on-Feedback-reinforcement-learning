"""
Retriever abstraction (P2 — retriever ladder).

Every retriever exposes

    search_pool(query_text, query_emb, k, exclude_idxs) -> DataFrame

returning the FAISS-metadata rows of the top-k candidates with the columns
    faiss_score   : retrieval score of THIS retriever (name kept for record
                    compatibility; it is a cosine for dense, BM25 for bm25,
                    RRF for hybrids, CE logit for cross-encoder re-ranking)
    faiss_rank    : 1-based rank under this retriever
    _faiss_idx    : row position in the shared metadata frame (== FAISS id)

All retrievers share the row ordering of `faiss_metadata.parquet`, so the LOO
self-exclusion by index position works identically for every retriever.

`DenseRetriever` over the all-MiniLM-L6-v2 index reproduces the SIKDD retrieval
path exactly (same FAISSIndex.search call, same metadata join).
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import ProjectPaths, ALT_EMBEDDERS, CROSS_ENCODER_MODEL
from .index import FAISSIndex
from .encoder import TicketEncoder

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def doc_text(title: str, description: str) -> str:
    description = description if isinstance(description, str) else ""
    return f"{title}\n{description}" if description else str(title)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


class BaseRetriever:
    name: str = "base"

    def __init__(self, meta: pd.DataFrame):
        self.meta = meta.reset_index(drop=True)

    def search_pool(self, query_text: str, query_emb: Optional[np.ndarray], k: int,
                    exclude_idxs: Optional[set[int]] = None) -> pd.DataFrame:
        raise NotImplementedError

    def _rows(self, idxs: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        rows = self.meta.iloc[idxs].copy()
        rows["_faiss_idx"] = idxs
        rows["faiss_score"] = scores.astype(np.float32)
        rows["faiss_rank"] = np.arange(1, len(rows) + 1)
        return rows

    @staticmethod
    def _top_k_excluding(all_scores: np.ndarray, k: int, exclude_idxs: Optional[set[int]]) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(-all_scores, kind="stable")
        if exclude_idxs:
            order = np.array([i for i in order if i not in exclude_idxs], dtype=np.int64)
        order = order[:k]
        return order, all_scores[order]


class DenseRetriever(BaseRetriever):
    """FAISS inner-product search. With `encoder=None` the caller must pass query_emb."""

    def __init__(self, faiss_index: FAISSIndex, encoder: Optional[TicketEncoder] = None, name: str = "dense_minilm"):
        super().__init__(faiss_index._meta)
        self.index = faiss_index
        self.encoder = encoder
        self.name = name

    def embed(self, query_text: str) -> np.ndarray:
        if self.encoder is None:
            raise RuntimeError(f"{self.name}: no encoder attached; pass query_emb instead")
        title, _, desc = query_text.partition("\n")
        return self.encoder.encode_ticket(title, desc)

    def search_pool(self, query_text, query_emb, k, exclude_idxs=None):
        if query_emb is None or (self.encoder is not None and self.name != "dense_minilm"):
            query_emb = self.embed(query_text)
        scores, indices = self.index.search(query_emb, k, exclude_idxs)
        rows = self.index.get_metadata(indices[0])
        rows["faiss_score"] = scores[0]
        rows["faiss_rank"] = range(1, len(rows) + 1)
        return rows


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, meta: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        super().__init__(meta)
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise ImportError("rank_bm25 is required for the BM25 retriever: pip install rank-bm25") from e
        self.k1, self.b = k1, b
        docs = [tokenize(doc_text(t, d)) for t, d in zip(self.meta["Title_anon"], self.meta["Description_anon"].fillna(""))]
        self.bm25 = BM25Okapi(docs, k1=k1, b=b)
        self.tokenizer_desc = "lowercase, [a-z0-9]+ tokens, no stemming, no stopwords"

    def all_scores(self, query_text: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query_text)), dtype=np.float32)

    def search_pool(self, query_text, query_emb, k, exclude_idxs=None):
        idxs, sc = self._top_k_excluding(self.all_scores(query_text), k, exclude_idxs)
        return self._rows(idxs, sc)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"k1": self.k1, "b": self.b, "bm25": self.bm25, "n_docs": len(self.meta)}, f)


class HybridRRFRetriever(BaseRetriever):
    """Reciprocal-rank fusion of several retrievers: score = sum_r 1/(k_rrf + rank_r)."""

    def __init__(self, retrievers: list[BaseRetriever], k_rrf: int = 60, pool_each: int = 200, name: str = "hybrid_rrf"):
        super().__init__(retrievers[0].meta)
        self.retrievers = retrievers
        self.k_rrf = k_rrf
        self.pool_each = pool_each
        self.name = name

    def search_pool(self, query_text, query_emb, k, exclude_idxs=None):
        fused: dict[int, float] = {}
        for r in self.retrievers:
            df = r.search_pool(query_text, query_emb, max(self.pool_each, k), exclude_idxs)
            for idx, rank in zip(df["_faiss_idx"].to_numpy(), df["faiss_rank"].to_numpy()):
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (self.k_rrf + float(rank))
        items = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        idxs = np.array([i for i, _ in items], dtype=np.int64)
        sc = np.array([s for _, s in items], dtype=np.float32)
        return self._rows(idxs, sc)


class CrossEncoderReranker(BaseRetriever):
    """Re-ranks the top-`pool` of a base retriever with a cross-encoder (scores cached in SQLite)."""

    def __init__(self, base: BaseRetriever, model_name: str = CROSS_ENCODER_MODEL, pool: int = 100,
                 cache_path: Optional[Path] = None, name: str = "ce_rerank", device: str = "cpu"):
        super().__init__(base.meta)
        from sentence_transformers import CrossEncoder
        self.base = base
        self.model_name = model_name
        self.pool = pool
        self.name = name
        self.model = CrossEncoder(model_name, device=device)
        self.cache_path = cache_path
        self._conn = None
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(cache_path)
            self._conn.execute("CREATE TABLE IF NOT EXISTS ce (key TEXT PRIMARY KEY, score REAL)")
            self._conn.commit()
        self._docs = [doc_text(t, d) for t, d in zip(self.meta["Title_anon"], self.meta["Description_anon"].fillna(""))]

    def _key(self, query_text: str, idx: int) -> str:
        return hashlib.sha256(f"{self.model_name}|{query_text}|{idx}".encode()).hexdigest()

    def _scores(self, query_text: str, idxs: np.ndarray) -> np.ndarray:
        out = np.zeros(len(idxs), dtype=np.float32)
        todo = []
        keys = [self._key(query_text, int(i)) for i in idxs]
        if self._conn is not None:
            for j, key in enumerate(keys):
                row = self._conn.execute("SELECT score FROM ce WHERE key=?", (key,)).fetchone()
                if row is None:
                    todo.append(j)
                else:
                    out[j] = row[0]
        else:
            todo = list(range(len(idxs)))
        if todo:
            pairs = [(query_text, self._docs[int(idxs[j])]) for j in todo]
            preds = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
            for j, p in zip(todo, preds):
                out[j] = float(p)
            if self._conn is not None:
                self._conn.executemany("INSERT OR REPLACE INTO ce (key, score) VALUES (?, ?)",
                                       [(keys[j], float(out[j])) for j in todo])
                self._conn.commit()
        return out

    def search_pool(self, query_text, query_emb, k, exclude_idxs=None):
        base_df = self.base.search_pool(query_text, query_emb, max(self.pool, k), exclude_idxs)
        idxs = base_df["_faiss_idx"].to_numpy().astype(np.int64)
        ce = self._scores(query_text, idxs)
        order = np.argsort(-ce, kind="stable")[:k]
        rows = self._rows(idxs[order], ce[order])
        rows["base_score"] = base_df["faiss_score"].to_numpy()[order]
        rows["base_rank"] = base_df["faiss_rank"].to_numpy()[order]
        return rows


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_dense_index(paths: ProjectPaths, tag: str) -> tuple[FAISSIndex, TicketEncoder]:
    model_name = ALT_EMBEDDERS[tag]
    index_dir = paths.data_processed / ("faiss_index" if tag == "minilm" else f"faiss_index_{tag}")
    enc = TicketEncoder(model_name)
    idx = FAISSIndex(dim=enc.dim)
    idx.load(index_dir / "faiss.index", index_dir / "faiss_metadata.parquet")
    return idx, enc


def build_retriever(
    name: str,
    paths: ProjectPaths,
    faiss_minilm: FAISSIndex,
    encoder_minilm: TicketEncoder,
    ce_pool: int = 100,
) -> BaseRetriever:
    cache = paths.root / "data" / "cache"
    dense = DenseRetriever(faiss_minilm, encoder_minilm, name="dense_minilm")
    if name == "dense_minilm":
        return dense
    if name == "bm25":
        return BM25Retriever(faiss_minilm._meta)
    if name == "hybrid_rrf":
        return HybridRRFRetriever([dense, BM25Retriever(faiss_minilm._meta)], name=name)
    if name == "dense_bge":
        idx, enc = load_dense_index(paths, "bge")
        return DenseRetriever(idx, enc, name=name)
    if name == "hybrid_bge_rrf":
        idx, enc = load_dense_index(paths, "bge")
        return HybridRRFRetriever([DenseRetriever(idx, enc, name="dense_bge"), BM25Retriever(faiss_minilm._meta)], name=name)
    if name == "ce_rerank":
        return CrossEncoderReranker(dense, pool=ce_pool, cache_path=cache / "ce_scores.db", name=name)
    if name == "ce_hybrid_rerank":
        hyb = HybridRRFRetriever([dense, BM25Retriever(faiss_minilm._meta)], name="hybrid_rrf")
        return CrossEncoderReranker(hyb, pool=ce_pool, cache_path=cache / "ce_scores.db", name=name)
    raise ValueError(f"Unknown retriever: {name}")
