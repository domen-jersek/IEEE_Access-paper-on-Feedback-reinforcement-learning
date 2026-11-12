"""History-restricted retrievers: one allowed candidate universe for every branch.

The allowed history is a pinned set of ticket IDs (normally the train partition).
Dense indices are rebuilt from the frozen embeddings of the completed run, so no
ticket is re-encoded and no eval ticket can enter the candidate pool. BM25 is
fitted on history documents only, so its IDF cannot use eval text. Hybrid fusion
and cross-encoder re-ranking operate on the same restricted pool.

Rebuilding is deterministic from (frozen index, history IDs). No artifact is
written unless the caller asks for it.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from ..config import ProjectPaths
from .index import FAISSIndex
from .retrievers import (
    BM25Retriever,
    BaseRetriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRRFRetriever,
)


def reconstruct_vectors(index_path: Path) -> np.ndarray:
    """Recover normalized embeddings from a flat inner-product index."""
    index = faiss.read_index(str(index_path))
    base = faiss.downcast_index(index.index)
    return base.reconstruct_n(0, base.ntotal).astype(np.float32)


def _subset(meta: pd.DataFrame, vectors: np.ndarray, history_ids: set[str],
            *, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    if len(meta) != len(vectors):
        raise ValueError(f"{source}: metadata/vector length mismatch")
    if meta["seq_id"].duplicated().any():
        raise ValueError(f"{source}: duplicate seq_id in metadata")
    unknown = set(history_ids) - set(meta["seq_id"])
    if unknown:
        raise ValueError(f"{source}: history IDs missing from metadata ({len(unknown)} unknown)")
    position = {seq_id: i for i, seq_id in enumerate(meta["seq_id"])}
    ordered = sorted(history_ids)
    rows = np.array([position[seq_id] for seq_id in ordered], dtype=np.int64)
    sub_meta = meta.iloc[rows].reset_index(drop=True)
    if sub_meta["seq_id"].tolist() != ordered:
        raise RuntimeError(f"{source}: history subset is not aligned")
    return sub_meta, vectors[rows]


def history_dense_index(dataset: pd.DataFrame, history_ids: set[str],
                        index_dir: Path | None = None) -> FAISSIndex:
    """Dense index whose universe is exactly ``history_ids``."""
    paths = ProjectPaths()
    index_dir = Path(index_dir) if index_dir is not None else paths.data_processed / "faiss_index"
    meta = pd.read_parquet(index_dir / "faiss_metadata.parquet")
    vectors = reconstruct_vectors(index_dir / "faiss.index")
    if meta["seq_id"].tolist() != dataset["seq_id"].astype(str).tolist():
        raise ValueError("Index metadata order does not match the canonical dataset")
    sub_meta, sub_vectors = _subset(meta, vectors, history_ids, source=str(index_dir))
    index = FAISSIndex(dim=int(sub_vectors.shape[1]))
    index.build(sub_vectors, sub_meta)
    return index


def history_meta(dataset: pd.DataFrame, history_ids: set[str]) -> pd.DataFrame:
    sub = dataset.loc[dataset["seq_id"].astype(str).isin(history_ids)].copy()
    if len(sub) != len(history_ids):
        raise ValueError("History IDs missing from dataset")
    return sub.sort_values("seq_id").reset_index(drop=True)


def build_history_retriever(
    name: str,
    dataset: pd.DataFrame,
    history_ids: set[str],
    *,
    ce_pool: int = 100,
    ce_cache_path: Path | None = None,
    index_dir: Path | None = None,
) -> BaseRetriever:
    """All seven retrievers, restricted to the same allowed history.

    The dense index is built lazily, so lexical-only branches never load vectors.
    """
    paths = ProjectPaths()
    hist_meta = history_meta(dataset, history_ids)

    def dense_minilm() -> DenseRetriever:
        return DenseRetriever(history_dense_index(dataset, history_ids, index_dir=index_dir),
                              encoder=None, name="dense_minilm")

    if name == "dense_minilm":
        return dense_minilm()
    if name == "bm25":
        return BM25Retriever(hist_meta)
    if name == "hybrid_rrf":
        return HybridRRFRetriever([dense_minilm(), BM25Retriever(hist_meta)], name=name)
    if name in ("dense_bge", "hybrid_bge_rrf"):
        bge_dir = paths.data_processed / "faiss_index_bge"
        dense_bge = DenseRetriever(history_dense_index(dataset, history_ids, index_dir=bge_dir),
                                   encoder=None, name="dense_bge")
        if name == "dense_bge":
            return dense_bge
        return HybridRRFRetriever([dense_bge, BM25Retriever(hist_meta)], name=name)
    if name in ("ce_rerank", "ce_hybrid_rerank"):
        cache = ce_cache_path or (paths.root / "data" / "cache" / "revision" / "ce_scores.db")
        base: BaseRetriever = dense_minilm()
        if name == "ce_hybrid_rerank":
            base = HybridRRFRetriever([dense_minilm(), BM25Retriever(hist_meta)], name="hybrid_rrf")
        return CrossEncoderReranker(base, pool=ce_pool, cache_path=cache, name=name)
    raise ValueError(f"Unknown retriever: {name}")
