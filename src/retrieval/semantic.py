"""
Ticket-text semantic similarity for the semantic relevance filter (P5).

The semantic relevance filter ignores a candidate's feedback lift when the
query<->candidate *ticket text* cosine falls below a threshold. Unlike the
retriever score (BM25 units for lexical retrievers, cosine for dense ones),
this similarity is computed from a single fixed encoder so the threshold means
the same thing across retrievers.

Cached like `ReplySimilarity`: the embedding matrix is built once per model tag
and stored under `data/cache/`.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("USE_TF", "0")
log = logging.getLogger(__name__)

TEXT_MODELS = {"minilm": "all-MiniLM-L6-v2"}


class TicketTextSimilarity:
    """Cached query<->candidate ticket-text cosine matrix over the canonical dataset."""

    def __init__(self, dataset: pd.DataFrame, encoder=None, model_tag: str = "minilm",
                 cache_dir: Optional[Path] = None):
        from ..config import ProjectPaths

        self.model_tag = model_tag
        self.ids = dataset["seq_id"].astype(str).tolist()
        self.pos = {s: i for i, s in enumerate(self.ids)}
        self.titles = dataset["Title_anon"].fillna("").astype(str).tolist()
        self.descriptions = dataset.get("Description_anon", pd.Series([""] * len(dataset))).fillna("").astype(str).tolist()
        cache_dir = cache_dir or (ProjectPaths().root / "data" / "cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._emb_path = cache_dir / f"ticket_text_emb_{model_tag}.npy"
        self._ids_path = cache_dir / f"ticket_text_emb_{model_tag}_ids.json"
        self.emb = self._load_or_build(encoder)
        self.sim = self.emb @ self.emb.T

    def _load_or_build(self, encoder) -> np.ndarray:
        if self._emb_path.exists() and self._ids_path.exists():
            cached_ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            if cached_ids == self.ids:
                return np.load(self._emb_path)
        if encoder is None:
            from .encoder import TicketEncoder
            encoder = TicketEncoder(TEXT_MODELS[self.model_tag])
        emb = encoder.encode_batch(self.titles, self.descriptions)
        np.save(self._emb_path, emb)
        self._ids_path.write_text(json.dumps(self.ids), encoding="utf-8")
        return emb

    def s(self, query_id: str, candidate_id: str) -> float:
        qi = self.pos.get(str(query_id))
        ci = self.pos.get(str(candidate_id))
        if qi is None or ci is None:
            return 0.0
        return float(self.sim[qi, ci])