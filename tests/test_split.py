"""Tests for split validation."""
import json
import pytest
from pathlib import Path


def test_disjoint_splits():
    splits_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
    if not splits_dir.exists():
        pytest.skip("Split files not yet generated")

    for split_file in sorted(splits_dir.glob("split_seed*.json")):
        data = json.loads(split_file.read_text("utf-8"))
        train = set(data["train"])
        dev = set(data["dev"])
        eval_ = set(data["eval"])

        assert len(train & dev) == 0, f"{split_file.name}: train ∩ dev non-empty"
        assert len(train & eval_) == 0, f"{split_file.name}: train ∩ eval non-empty"
        assert len(dev & eval_) == 0, f"{split_file.name}: dev ∩ eval non-empty"

        total = len(train | dev | eval_)
        assert total == len(train) + len(dev) + len(eval_), f"{split_file.name}: union size mismatch"


def test_disjoint_has_no_overlap_with_train():
    splits_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
    if not splits_dir.exists():
        pytest.skip("Split files not yet generated")

    random_42 = json.loads(Path(splits_dir, "split_seed42.json").read_text("utf-8"))
    disjoint_42 = json.loads(Path(splits_dir, "split_seed42_disjoint.json").read_text("utf-8"))

    dj_train = set(disjoint_42["train"])
    dj_eval = set(disjoint_42["eval"])
    assert len(dj_train & dj_eval) == 0, "Disjoint: train ∩ eval"


def test_split_sizes():
    splits_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "splits"
    if not splits_dir.exists():
        pytest.skip("Split files not yet generated")

    for split_file in sorted(splits_dir.glob("split_seed*.json")):
        data = json.loads(split_file.read_text("utf-8"))
        tc = data.get("ticket_count", {})
        if tc:
            assert tc.get("train", 0) > 0
            assert tc.get("dev", 0) > 0
            assert tc.get("eval", 0) > 0