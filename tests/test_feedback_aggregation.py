"""Tests for feedback aggregation modes (continuous vs binary)."""
import pytest
import pandas as pd
from src.feedback.loader import aggregate_feedback_scores


def _make_df(scores: list[float]) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(scores):
        rows.append({
            "query_id": f"Q-{i}",
            "candidate_id": "C-1",
            "query_class": "admin_rights",
            "query_team": "(GI-UX) Group",
            "score": s,
        })
    return pd.DataFrame(rows)


def test_continuous_all_high():
    df = _make_df([0.95, 0.88, 0.92, 0.85, 0.90])
    scores = aggregate_feedback_scores(df, mode="continuous")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == pytest.approx(4.50, abs=0.01)
    assert entry["neg"] == pytest.approx(0.50, abs=0.01)


def test_continuous_all_low():
    df = _make_df([0.10, 0.15, 0.20, 0.05])
    scores = aggregate_feedback_scores(df, mode="continuous")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == pytest.approx(0.50, abs=0.01)
    assert entry["neg"] == pytest.approx(3.50, abs=0.01)


def test_continuous_all_middling():
    df = _make_df([0.50, 0.50, 0.50, 0.50])
    scores = aggregate_feedback_scores(df, mode="continuous")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == pytest.approx(2.00, abs=0.01)
    assert entry["neg"] == pytest.approx(2.00, abs=0.01)


def test_continuous_mixed():
    df = _make_df([0.95, 0.10, 0.80, 0.20, 0.50])
    scores = aggregate_feedback_scores(df, mode="continuous")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == pytest.approx(2.55, abs=0.01)
    assert entry["neg"] == pytest.approx(2.45, abs=0.01)


def test_binary_all_high():
    df = _make_df([0.95, 0.88, 0.92, 0.85, 0.90])
    scores = aggregate_feedback_scores(df, mode="binary")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == 5.0
    assert entry["neg"] == 0.0


def test_binary_all_low():
    df = _make_df([0.10, 0.15, 0.20, 0.05])
    scores = aggregate_feedback_scores(df, mode="binary")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == 0.0
    assert entry["neg"] == 4.0


def test_binary_all_middling():
    df = _make_df([0.50, 0.60, 0.70, 0.79])
    scores = aggregate_feedback_scores(df, mode="binary")
    assert "C-1" not in scores


def test_binary_cliff():
    df = _make_df([0.79, 0.80, 0.79, 0.80])
    scores = aggregate_feedback_scores(df, mode="binary")
    entry = scores["C-1"]["global"]
    assert entry["pos"] == 2.0
    assert entry["neg"] == 0.0


def test_continuous_scopes():
    df = _make_df([0.90, 0.10])
    scores = aggregate_feedback_scores(df, mode="continuous")
    entry = scores["C-1"]
    assert entry["global"]["pos"] == pytest.approx(1.00, abs=0.01)
    assert entry["class:admin_rights"]["pos"] == pytest.approx(1.00, abs=0.01)
    assert entry["team:(GI-UX) Group"]["pos"] == pytest.approx(1.00, abs=0.01)
    assert "intersection:admin_rights:(GI-UX) Group" in entry