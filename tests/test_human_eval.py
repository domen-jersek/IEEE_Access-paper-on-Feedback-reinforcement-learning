"""Human-evaluation analysis: orientation, paired effects, agreement."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.human_eval import (
    Study,
    agreement_tables,
    cohen_kappa,
    combined_effect,
    gwet_ac1,
    load_ratings,
    mcnemar_exact,
    mde_proportion,
    orient,
    pabak,
    rater_tables,
    weighted_kappa,
)


def _study() -> Study:
    items = {f"i{k}": {"item_id": f"i{k}", "A": "a" * (100 + k * 50), "B": "b" * (100 + k * 50)}
             for k in range(1, 5)}
    manifest = {"sample": {"p/s": {"changed_share": 0.5}}, "runs": {}}
    automatic = {("p", "s", f"T{k}"): [-0.1, 0.2, 0.0, 0.05][k - 1] for k in range(1, 5)}
    return Study(Path("."), items, {}, manifest, automatic, "pid")


def _ratings(rater="r1", second=False) -> pd.DataFrame:
    rows = []
    specs = [
        ("i1", "baseline", "correct", "incorrect", "A"),
        ("i2", "feedback", "correct", "incorrect", "A"),
        ("i3", "baseline", "correct", "correct", "equal"),
        ("i4", "baseline", "incorrect", "incorrect", "both_incorrect"),
    ]
    for item, a_side, a_correct, b_correct, preference in specs:
        if second and item == "i1":
            preference = "B"
        b_side = "feedback" if a_side == "baseline" else "baseline"
        if a_side == "feedback":
            a_correct, b_correct = a_correct, b_correct
        rows.append({
            "rater": rater, "item_id": item, "panel": "p", "ticket_id": "T" + item[-1],
            "source": "s", "block": "answer", "a_correct": a_correct, "b_correct": b_correct,
            "preference": preference, "a_side": a_side, "b_side": b_side,
            "a_unsafe": 0, "b_unsafe": 0, "a_missing": 0, "b_missing": 0,
            "package_id": "pid",
        })
    return pd.DataFrame(rows)


def test_orientation_maps_sides_and_scores():
    study = _study()
    df = orient(_ratings(), study)
    row = df[df.item_id == "i1"].iloc[0]
    assert row["baseline_correct"] == "correct"
    assert row["feedback_correct"] == "incorrect"
    assert row["baseline_strict"] == 1 and row["feedback_strict"] == 0
    assert row["preferred"] == "baseline"
    assert row["auto_delta"] == pytest.approx(-0.1)
    assert row["baseline_len"] == 150 and row["feedback_len"] == 150


def test_rater_tables_effects_and_preferences():
    study = _study()
    df = orient(_ratings(), study)
    tables = rater_tables(df, study)
    summary = tables["summary"][tables["summary"].metric.eq("strict")].iloc[0]
    assert summary["n"] == 4
    assert summary["baseline_rate"] == pytest.approx(0.5)
    assert summary["feedback_rate"] == pytest.approx(0.5)
    assert summary["delta"] == pytest.approx(0.0)
    assert summary["baseline_only"] == 1 and summary["feedback_only"] == 1
    prefs = tables["preferences"].iloc[0]
    assert prefs["equal"] == 1 and prefs["both_incorrect"] == 1
    assert prefs["net_feedback_minus_baseline"] == 0


def test_agreement_and_combined_effect():
    study = _study()
    first = orient(_ratings("r1"), study)
    second = orient(_ratings("r2", second=True), study)
    tables = agreement_tables(first, second)
    row = tables["agreement"].iloc[0]
    assert row["n_items"] == 4
    assert row["n_common"] == 4 and row["n_rater_1"] == 4 and row["n_rater_2"] == 4
    assert row["preference_agreement"] == pytest.approx(0.75)
    assert 0 <= row["preference_kappa"] < 1
    combined = combined_effect(first, second).iloc[0]
    assert combined["combined_delta"] == pytest.approx(0.0)


def test_agreement_supports_partial_overlap():
    study = _study()
    first = orient(_ratings("r1"), study)
    second = orient(_ratings("r2", second=True), study).iloc[:-1]
    tables = agreement_tables(first, second)
    row = tables["agreement"].iloc[0]
    assert row["n_common"] == 3
    assert row["n_rater_1"] == 4 and row["n_rater_2"] == 3
    combined = combined_effect(first, second).iloc[0]
    assert combined["n_common"] == 3 and combined["n_items"] == 3


def test_perfect_agreement_kappa_is_one():
    labels = ["A", "B", "equal", "both_incorrect"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert pabak(labels, labels) == pytest.approx(1.0)
    assert gwet_ac1(labels, labels) == pytest.approx(1.0)


def test_weighted_kappa_prefers_near_misses():
    a = ["correct", "partially_correct", "incorrect", "correct"]
    near = ["partially_correct", "incorrect", "incorrect", "correct"]
    far = ["incorrect", "incorrect", "incorrect", "incorrect"]
    assert weighted_kappa(a, near) > weighted_kappa(a, far)


def test_mcnemar_and_mde():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    result = mde_proportion(35, 0.3)
    assert 0.5 < result["detectable_advantage"] <= 1.0
    assert result["mde_delta"] > 0


def test_load_ratings_rejects_wrong_package_and_duplicates():
    study = _study()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.csv"
        _ratings().to_csv(path, index=False)
        load_ratings(path, study)
        bad = _ratings()
        bad["package_id"] = "other"
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError, match="package_id"):
            load_ratings(path, study)
        dup = pd.concat([_ratings(), _ratings().iloc[:1]])
        dup.to_csv(path, index=False)
        with pytest.raises(ValueError, match="Duplicate"):
            load_ratings(path, study)
