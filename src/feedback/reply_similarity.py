"""Feedback scores derived from recorded historical replies, without an LLM judge.

Two sources, applied to exactly the same allowed (query, candidate) pairs as the
judge:

``exact``      1.0 when the two normalized full replies are identical, else 0.0.
``embed_bge``  cosine similarity of the two recorded replies under an embedding
               family that is independent of the MiniLM answer metric.
``embed_minilm`` the MiniLM reply embedding, for a same-embedder control.

These scores replace the judge score and use the same aggregation and lift, which
is the baseline requested in R1-M5. No API call is made.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..data.groups_v2 import normalize_signature

ReplySource = Literal["exact", "embed_bge", "embed_minilm"]
_EMBEDDING_TAG = {"embed_bge": "bge", "embed_minilm": "minilm"}


def _reply_embeddings(dataset: pd.DataFrame, tag: str, cache_dir: Path) -> tuple[list[str], np.ndarray]:
    ids_path = cache_dir / f"reply_emb_{tag}_ids.json"
    vector_path = cache_dir / f"reply_emb_{tag}.npy"
    if not ids_path.exists() or not vector_path.exists():
        raise FileNotFoundError(
            f"Missing cached reply embeddings for {tag}; run the proxy metric once to build them"
        )
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if ids != dataset["seq_id"].astype(str).tolist():
        raise ValueError(f"Reply embedding cache for {tag} is not aligned with the dataset")
    matrix = np.load(vector_path)
    if matrix.shape[0] != len(dataset):
        raise ValueError(f"Reply embedding matrix for {tag} has the wrong shape")
    return ids, matrix.astype(np.float32)


def with_reply_similarity(
    rows: pd.DataFrame,
    dataset: pd.DataFrame,
    source: ReplySource,
    *,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a copy of ``rows`` with ``score`` from reply similarity and a new protocol label."""
    required = {"query_id", "candidate_id", "score"}
    if missing := required - set(rows.columns):
        raise ValueError(f"Missing feedback columns: {sorted(missing)}")
    if source not in {"exact", "embed_bge", "embed_minilm"}:
        raise ValueError(f"Unsupported reply-similarity source: {source}")
    dataset = dataset.reset_index(drop=True)
    seq_ids = dataset["seq_id"].astype(str).tolist()
    known = set(seq_ids)
    unknown = (set(rows["query_id"]) | set(rows["candidate_id"])) - known
    if unknown:
        raise ValueError(f"Unknown ticket IDs in feedback rows: {len(unknown)}")

    if source == "exact":
        keys = dict(zip(seq_ids, dataset["first_reply"].map(normalize_signature)))
        scores = np.array([
            1.0 if keys[q] and keys[q] == keys[c] else 0.0
            for q, c in zip(rows["query_id"], rows["candidate_id"])
        ], dtype=float)
    else:
        tag = _EMBEDDING_TAG[source]
        cache = Path(cache_dir) if cache_dir is not None else ProjectPaths().root / "data" / "cache"
        ids, matrix = _reply_embeddings(dataset, tag, cache)
        position = {seq_id: i for i, seq_id in enumerate(ids)}
        q_idx = np.array([position[q] for q in rows["query_id"]], dtype=np.int64)
        c_idx = np.array([position[c] for c in rows["candidate_id"]], dtype=np.int64)
        scores = np.clip((matrix[q_idx] * matrix[c_idx]).sum(axis=1), 0.0, 1.0).astype(float)

    out = rows.copy()
    out["score"] = scores
    out["protocol"] = f"reply_{source}"
    return out
