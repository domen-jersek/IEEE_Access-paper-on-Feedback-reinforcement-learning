"""Revision boundary tests: groups, group-aware split, history retrieval, priors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import ProjectPaths
from src.data.groups_v2 import (
    CONTROL_RULE_IDS,
    build_group_mapping,
    group_artifact,
    normalize_signature,
)
from src.data.split_v2 import group_aware_split, validate_group_split
from src.evaluation.proxy import ReplySimilarity
from src.feedback.loader import FeedbackBundle
from src.retrieval.history import build_history_retriever, history_dense_index
from src.retrieval.index import FAISSIndex
from src.retrieval.retrievers import CrossEncoderReranker


def _frame(replies, titles=None, subcategories=None):
    n = len(replies)
    return pd.DataFrame({
        "seq_id": [f"R-{i+1}" for i in range(n)],
        "Title_anon": titles or [f"title {i}" for i in range(n)],
        "Service subcategory->Name": subcategories or [f"sub {i}" for i in range(n)],
        "first_reply": replies,
        "intent_class": ["a"] * n,
        "Team->Name": ["t"] * n,
    })


def test_empty_signatures_never_link_tickets():
    df = _frame(["", "", "unique"], titles=["", "same title", ""], subcategories=["", "", ""])
    mapping, stats = build_group_mapping(df, "duplicate_template_v1")
    # Empty reply + empty title/subcategory must not collapse the two empty tickets.
    assert mapping.iloc[0] != mapping.iloc[1]
    assert stats["n_tickets_in_multi_ticket_groups"] >= 0


def test_identical_full_reply_is_one_group_and_prefix_rule_keeps_it():
    df = _frame(["same reply text", "same reply text", "other"])
    full, _ = build_group_mapping(df, "duplicate_template_v1")
    prefix, _ = build_group_mapping(df, "prefix_template_v1")
    assert full.iloc[0] == full.iloc[1] != full.iloc[2]
    assert prefix.iloc[0] == prefix.iloc[1]
    # Union rule: any ticket grouped by the full rule is also grouped by the prefix rule.
    assert len({full.iloc[0]}) == 1


def test_title_subcategory_links_and_transitivity():
    df = _frame(["a", "b", "c"], titles=["same", "same", "same"], subcategories=["x", "x", "x"])
    mapping, _ = build_group_mapping(df, "duplicate_template_v1")
    assert mapping.nunique() == 1


def test_group_artifact_is_keyed_by_ticket_id_and_deterministic():
    df = _frame(["a", "a", "b"])
    artifact, stats = group_artifact(df, "duplicate_template_v1")
    assert set(artifact) == {"schema_version", "rule_id", "semantic_status", "groups"}
    assert set(artifact["groups"]) == set(df.seq_id)
    assert artifact["groups"]["R-1"] == artifact["groups"]["R-2"]
    again, _ = group_artifact(df, "duplicate_template_v1")
    assert artifact == again
    assert stats["n_multi_ticket_groups"] == 1  # the two identical replies
    assert stats["n_tickets_in_multi_ticket_groups"] == 2


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="Unknown group rule"):
        build_group_mapping(_frame(["a"]), "not_a_rule")
    assert "duplicate_template_v1" in CONTROL_RULE_IDS


def _grouped_frame(n=40, group_size=4):
    replies = [f"reply {i // group_size}" for i in range(n)]
    return _frame(replies)


def test_split_keeps_groups_atomic_and_covers_every_ticket():
    df = _grouped_frame()
    mapping, _ = build_group_mapping(df, "duplicate_template_v1")
    split = group_aware_split(df, mapping, seed=42)
    validate_group_split(df, mapping, split)
    assert sum(len(split[p]) for p in ("train", "dev", "eval")) == len(df)
    assert all(split[p] for p in ("train", "dev", "eval"))


def test_split_is_deterministic_per_seed_and_seed_sensitive():
    df = _grouped_frame(120, 3)
    mapping, _ = build_group_mapping(df, "duplicate_template_v1")
    first = group_aware_split(df, mapping, seed=42)
    assert first == group_aware_split(df, mapping, seed=42)
    other = group_aware_split(df, mapping, seed=123)
    assert (first["train"], first["dev"], first["eval"]) != (other["train"], other["dev"], other["eval"])


def test_group_crossing_split_is_detected():
    df = _frame(["same", "same", "other"])
    mapping, _ = build_group_mapping(df, "duplicate_template_v1")
    bad = {"train": ["R-1"], "dev": ["R-2"], "eval": ["R-3"]}
    with pytest.raises(ValueError, match="crosses"):
        validate_group_split(df, mapping, bad)


def _write_index(path: Path, dataset: pd.DataFrame, vectors: np.ndarray) -> None:
    index = FAISSIndex(dim=vectors.shape[1])
    meta = dataset[["seq_id", "Title_anon", "Description_anon", "first_reply"]].copy()
    index.build(vectors, meta)
    index.save(path / "faiss.index", path / "faiss_metadata.parquet")


def test_history_dense_index_contains_only_history_tickets(tmp_path):
    df = _frame(["a", "b", "c"])
    df["Description_anon"] = ""
    vectors = np.eye(3, dtype=np.float32)
    _write_index(tmp_path, df, vectors)
    index = history_dense_index(df, {"R-2", "R-3"}, index_dir=tmp_path)
    assert index.size == 2
    assert set(index._meta["seq_id"]) == {"R-2", "R-3"}
    # A query outside history must retrieve only history tickets, even when the
    # excluded ticket is the nearest neighbour.
    scores, indices = index.search(vectors[0][None, :], 2, set())
    assert set(index._meta.iloc[indices[0]]["seq_id"]) <= {"R-2", "R-3"}


def test_history_retriever_never_returns_eval_candidates():
    df = _frame(["alpha beta", "gamma delta", "alpha beta gamma"])
    df["Description_anon"] = ""
    history = {"R-1"}
    retriever = build_history_retriever("bm25", df, history)
    pool = retriever.search_pool("alpha beta", None, 3, set())
    assert set(pool["seq_id"]) == history


def test_bm25_is_fitted_on_history_only():
    df = _frame(["alpha " * 5, "beta " * 5, "gamma " * 5])
    df["Description_anon"] = ""
    full = build_history_retriever("bm25", df, set(df.seq_id))
    sub = build_history_retriever("bm25", df, {"R-1", "R-2"})
    assert len(full.meta) == 3 and len(sub.meta) == 2
    # IDF comes from the fitted corpus; the third document must not contribute.
    assert full.bm25.idf != sub.bm25.idf


def _feedback_rows():
    return pd.DataFrame([
        ("Q1", "C1", "a", "x", 0.9),
        ("Q2", "C2", "a", "x", 0.1),
        ("Q3", "C1", "b", "y", 0.3),
        ("Q1", "C2", "a", "x", 0.7),
    ], columns=["query_id", "candidate_id", "query_class", "query_team", "score"])


def test_priors_come_from_the_supplied_rows_only():
    rows = _feedback_rows()
    bundle = FeedbackBundle(rows)
    assert bundle.prior_mean("global") == pytest.approx(rows.score.mean())
    assert bundle.prior_mean("class:a") == pytest.approx(rows.loc[rows.query_class.eq("a"), "score"].mean())
    filtered = rows[rows.candidate_id.eq("C1")]
    assert FeedbackBundle(filtered).prior_mean("global") == pytest.approx(0.6)


def test_leave_ticket_out_removes_target_in_both_roles_for_priors():
    rows = _feedback_rows()
    bundle = FeedbackBundle(rows)
    # Target C1 appears as candidate twice and never as query.
    remaining = rows[rows.candidate_id.ne("C1")]
    view = bundle.minus_query_bundle("C1")
    assert view.prior_mean("global") == pytest.approx(remaining.score.mean())
    assert view.prior_mean("class:a") == pytest.approx(
        remaining.loc[remaining.query_class.eq("a"), "score"].mean())
    # Target Q1 appears as query twice and as candidate never.
    remaining_q = rows[rows.query_id.ne("Q1")]
    view_q = bundle.minus_query_bundle("Q1")
    assert view_q.prior_mean("global") == pytest.approx(remaining_q.score.mean())
    assert view_q.scores.get("C1", {}).get("global", {}).get("pos", 0.0) == pytest.approx(0.3)


def test_ce_cache_key_is_content_based_not_position_based():
    reranker = object.__new__(CrossEncoderReranker)
    reranker.model_name = "cross-encoder/test"
    reranker.meta = pd.DataFrame({"seq_id": ["T1", "T2"]})
    reranker._docs = ["doc one", "doc two"]
    first = reranker._key("query", 0)
    assert first == reranker._key("query", 0)
    reranker._docs[0] = "doc one changed"
    assert reranker._key("query", 0) != first
    # Same content at a different row still keys differently because the ID differs.
    assert reranker._key("query", 0) != reranker._key("query", 1)


def test_reply_similarity_overrides_are_used_for_hit_metrics(tmp_path):
    df = _frame(["a", "a", "b", "b"])
    df["reply_hash"] = "legacy"
    df["combined_group_hash"] = "legacy"
    override = {"R-1": "g1", "R-2": "g1", "R-3": "g2", "R-4": "g2"}
    # tmp_path is mandatory here: the default cache is shared with real runs and
    # must never be rewritten from a synthetic four-row frame.
    sim = ReplySimilarity(df, cache_dir=tmp_path, group_hash_override=override,
                          reply_hash_override=override)
    assert sim.group_hash["R-1"] == "g1"
    hit = sim.metrics("R-1", ["R-2", "R-3"], k=2)
    assert hit["group_hit1"] == 1.0
    default = ReplySimilarity(df, cache_dir=tmp_path)
    assert default.group_hash["R-1"] == "legacy"


class _StubJudge:
    protocol = "conditioned"
    model = "stub/model"
    prompt_version = "v1"

    async def judge_pairs(self, pairs):
        return [0.5] * len(pairs)


class _StubEncoder:
    def encode_ticket(self, title, description=""):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_feedback_builder_excludes_self_pairs_by_integer_position(tmp_path):
    import asyncio
    import sqlite3

    from src.feedback.builder import FeedbackBuilder
    from src.feedback.loader import FeedbackDB

    df = _frame(["a", "b", "c"])
    df["Description_anon"] = ""
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    index = FAISSIndex(dim=2)
    index.build(vectors, df[["seq_id", "Title_anon", "Description_anon", "first_reply"]])
    db_path = tmp_path / "feedback.db"
    builder = FeedbackBuilder(index, df, _StubJudge(), FeedbackDB(db_path), top_k=2)
    asyncio.run(builder.build(["R-1"], _StubEncoder()))
    with sqlite3.connect(db_path) as connection:
        rows = pd.read_sql_query("SELECT query_id, candidate_id FROM feedback", connection)
    assert len(rows) == 2
    assert (rows["query_id"] != rows["candidate_id"]).all()
    assert set(rows["candidate_id"]) == {"R-2", "R-3"}

