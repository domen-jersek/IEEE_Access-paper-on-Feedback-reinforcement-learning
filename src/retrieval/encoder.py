"""
SentenceTransformer encoder wrapper.
Handles text formatting and embedding for tickets.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class TicketEncoder:
    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str = "cpu"):
        self.model_name = model_name
        log.info("Loading SentenceTransformer: %s", model_name)
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def encode_ticket(self, title: str, description: str = "") -> np.ndarray:
        text = f"{title}\n{description}" if description else title
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(self, titles: list[str], descriptions: Optional[list[str]] = None) -> np.ndarray:
        if descriptions:
            texts = [f"{t}\n{d}" if isinstance(d, str) and d else t for t, d in zip(titles, descriptions)]
        else:
            texts = titles
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)