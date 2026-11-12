"""Reply-similarity feedback sources (no LLM judge, no API)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.feedback.reply_similarity import with_reply_similarity


def _dataset(replies):
    n = len(replies)
    return pd.DataFrame({
        "seq_id": [f"R-{i+1}" for i in range(n)],
        "Title_anon": [f"title {i}" for i in range(n)],
        "Description_anon": [""] * n,
        "first_reply": replies,
    })


def _rows(pairs):
    return pd.DataFrame(pairs, columns=["query_id", "candidate_id", "score"])


def test_exact_match_scores_identical_replies():
    dataset = _dataset(["same reply", "same reply  ", "other"])
    rows = _rows([("R-1", "R-2", 0.0), ("R-1", "R-3", 0.0), ("R-1", "R-1", 0.0)])
    scored = with_reply_similarity(rows, dataset, "exact")
    assert scored["score"].tolist() == [1.0, 0.0, 1.0]
    assert scored["protocol"].tolist() == ["reply_exact"] * 3
    # Source rows are not mutated.
    assert rows["score"].tolist() == [0.0, 0.0, 0.0]


def test_embedding_scores_use_cached_vectors(tmp_path):
    dataset = _dataset(["a", "b"])
    ids = dataset["seq_id"].tolist()
    vectors = np.array([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    np.save(tmp_path / "reply_emb_bge.npy", vectors)
    (tmp_path / "reply_emb_bge_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    rows = _rows([("R-1", "R-2", 0.0), ("R-1", "R-1", 0.0)])
    scored = with_reply_similarity(rows, dataset, "embed_bge", cache_dir=tmp_path)
    assert scored["score"].iloc[0] == pytest.approx(0.6, abs=1e-6)
    assert scored["score"].iloc[1] == pytest.approx(1.0, abs=1e-6)


def test_embedding_cache_must_align_with_dataset(tmp_path):
    dataset = _dataset(["a", "b"])
    np.save(tmp_path / "reply_emb_bge.npy", np.eye(2, dtype=np.float32))
    (tmp_path / "reply_emb_bge_ids.json").write_text(json.dumps(["X", "Y"]), encoding="utf-8")
    with pytest.raises(ValueError, match="not aligned"):
        with_reply_similarity(_rows([("R-1", "R-2", 0.0)]), dataset, "embed_bge", cache_dir=tmp_path)


def test_unknown_ids_and_missing_columns_are_rejected(tmp_path):
    dataset = _dataset(["a", "b"])
    with pytest.raises(ValueError, match="Unknown ticket"):
        with_reply_similarity(_rows([("R-1", "R-9", 0.0)]), dataset, "exact")
    with pytest.raises(ValueError, match="Missing feedback columns"):
        with_reply_similarity(pd.DataFrame({"query_id": ["R-1"]}), dataset, "exact")
    with pytest.raises(ValueError, match="Unsupported"):
        with_reply_similarity(_rows([("R-1", "R-2", 0.0)]), dataset, "magic")
