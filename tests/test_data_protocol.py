"""Revision contract tests: small synthetic artifacts, no models or APIs."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import DataProtocolConfig, DEFAULT_METHODS
from src.data.artifacts import content_id, file_sha256, write_once_json
from src.data.protocol import load_data_protocol, resolve_data_context


@pytest.fixture
def inputs(tmp_path):
    dataset = pd.DataFrame({
        "seq_id": ["T1", "T2", "T3", "D1", "E1", "E2"],
        "Title_anon": ["request"] * 6,
        "Description_anon": [None, "description", "", "dev", "eval", "eval two"],
        "first_reply": ["private reference"] * 6,
        "intent_class": ["reply-derived-class"] * 6,
        "Team->Name": ["resolving-team"] * 6,
    })
    dataset.to_parquet(tmp_path / "dataset.parquet", index=False)
    split = {"train": ["T1", "T2", "T3"], "dev": ["D1"], "eval": ["E1", "E2"]}
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return tmp_path, dataset, split


def config_for(root, **kwargs):
    return DataProtocolConfig(
        dataset_path="dataset.parquet", dataset_sha256=file_sha256(root / "dataset.parquet"),
        split_path="split.json", split_sha256=file_sha256(root / "split.json"), **kwargs,
    )


def context_for(inputs, **kwargs):
    return resolve_data_context(config_for(inputs[0], **kwargs), root=inputs[0])


def feedback():
    return pd.DataFrame([
        ("T1", "T2", .9, "conditioned"),
        ("T2", "T3", .1, "conditioned"),
        ("T3", "T1", .8, "conditioned"),
        ("D1", "T2", 1., "conditioned"),
        ("T2", "E1", 1., "conditioned"),
        ("T1", "T1", 1., "conditioned"),
        ("T1", "T2", .4, "blind"),
    ], columns=["query_id", "candidate_id", "score", "protocol"])


def test_text_only_context_has_no_reference_or_posthoc_labels(inputs):
    ctx = context_for(inputs)
    assert ctx.query_ids == ("D1",)
    assert ctx.history_ids == {"T1", "T2", "T3"}
    query = asdict(ctx.queries["D1"])
    assert query["routing_class"] is None and query["routing_team"] is None
    assert "first_reply" not in query and "private reference" not in json.dumps(query)
    assert "private reference" not in json.dumps(ctx.to_manifest())
    assert "resolving-team" not in json.dumps(ctx.to_manifest())
    with pytest.raises(TypeError):
        ctx.queries["bad"] = ctx.queries["D1"]


def test_posthoc_is_explicit_and_changes_context_not_history(inputs):
    plain = context_for(inputs)
    ctx = context_for(inputs, metadata_policy="posthoc_diagnostic")
    assert ctx.queries["D1"].routing_team == "resolving-team"
    assert ctx.queries["D1"].metadata_policy == "posthoc_diagnostic"
    assert ctx.context_id != plain.context_id
    assert ctx.history_id == plain.history_id


def test_missing_posthoc_metadata_fails_but_text_only_does_not_need_it(inputs):
    root, dataset, _ = inputs
    dataset.loc[dataset.seq_id.eq("D1"), "Team->Name"] = None
    dataset.to_parquet(root / "dataset.parquet", index=False)
    context_for(inputs)
    with pytest.raises(ValueError, match="Missing posthoc"):
        context_for(inputs, metadata_policy="posthoc_diagnostic")


def test_duplicate_dataset_ids_fail(inputs):
    root, dataset, _ = inputs
    dataset.loc[0, "seq_id"] = "T2"
    dataset.to_parquet(root / "dataset.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate IDs"):
        context_for(inputs)


def test_identity_is_stable_and_location_independent(inputs, tmp_path):
    root = inputs[0]
    ctx = context_for(inputs)
    assert context_for(inputs).context_id == ctx.context_id
    other = root / "moved"
    other.mkdir()
    for name in ("dataset.parquet", "split.json"):
        (other / name).write_bytes((root / name).read_bytes())
    cfg = replace(config_for(root), dataset_path=str(other / "dataset.parquet"),
                  split_path=str(other / "split.json"), revision_id="other-label")
    moved = resolve_data_context(cfg, root=root)
    assert moved.context_id == ctx.context_id
    assert moved.history_id == ctx.history_id
    assert context_for(inputs, query_partition="eval").context_id != ctx.context_id


def test_changed_input_requires_repinning(inputs):
    root, dataset, _ = inputs
    cfg = config_for(root)
    original = resolve_data_context(cfg, root=root)
    dataset.loc[0, "Title_anon"] = "changed"
    dataset.to_parquet(root / "dataset.parquet", index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_data_context(cfg, root=root)
    changed = context_for(inputs)
    assert changed.context_id != original.context_id
    assert changed.history_id != resolve_history_id(cfg, ["T1", "T2", "T3"])


def resolve_history_id(cfg, ids):
    return content_id({"schema": "history_v1", "dataset": cfg.dataset_sha256, "ids": sorted(ids)})


@pytest.mark.parametrize("setting,value", [
    ("schema_version", True), ("schema_version", 2), ("dataset_sha256", "bad"),
    ("query_partition", "test"), ("history_partition", "all"),
    ("metadata_policy", "predicted"), ("self_pairs", "include"),
    ("evidence_policy", "history_top200"), ("scenario", "procedure_disjoint"),
    ("revision_id", ""), ("groups_path", "groups.json"),
])
def test_config_rejects_unimplemented_or_incomplete_policies(inputs, setting, value):
    values = config_for(inputs[0]).to_dict()
    values[setting] = value
    with pytest.raises(ValueError):
        DataProtocolConfig.from_dict(values)


def test_config_rejects_unknown_missing_and_duplicate_keys(inputs):
    root = inputs[0]
    values = config_for(root).to_dict()
    with pytest.raises(ValueError, match="Unknown"):
        DataProtocolConfig.from_dict({**values, "search_kk": 100})
    with pytest.raises(ValueError, match="Missing"):
        DataProtocolConfig.from_dict({})
    for ext, text in [("yaml", "scenario: a\nscenario: b\n"),
                      ("json", '{"scenario":"a", "scenario":"b"}')]:
        path = root / f"duplicate.{ext}"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate"):
            load_data_protocol(path)


@pytest.mark.parametrize("ext", ["json", "yaml"])
def test_config_roundtrip(inputs, ext):
    cfg = config_for(inputs[0])
    # JSON is also valid YAML.
    path = inputs[0] / f"config.{ext}"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert load_data_protocol(path) == cfg


@pytest.mark.parametrize("change,error", [
    ("duplicate", "duplicate"), ("overlap", "overlap"),
    ("missing", "cover"), ("unknown", "cover"), ("empty", "nonempty"),
])
def test_split_boundaries(inputs, change, error):
    root, _, split = inputs
    if change == "duplicate":
        split["train"].append("T1")
    elif change == "overlap":
        split["dev"] = ["T1"]
    elif change == "missing":
        split["eval"] = ["E1"]
    elif change == "unknown":
        split["eval"] = ["E1", "unknown"]
    else:
        split["dev"] = []
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        context_for(inputs)


def groups_config(inputs, groups=None, **kwargs):
    root, dataset, _ = inputs
    artifact = {"schema_version": 1, "rule_id": "synthetic_full_reply_or_title_v1",
                "semantic_status": "structural_proxy",
                "groups": groups if groups is not None else {q: q for q in dataset.seq_id}}
    (root / "groups.json").write_text(json.dumps(artifact), encoding="utf-8")
    return config_for(root, scenario="held_out_groups", groups_path="groups.json",
                      groups_sha256=file_sha256(root / "groups.json"), **kwargs)


def test_held_out_requires_groups(inputs):
    with pytest.raises(ValueError, match="requires"):
        config_for(inputs[0], scenario="held_out_groups")
    ctx = resolve_data_context(groups_config(inputs), root=inputs[0])
    assert ctx.group_semantic_status == "structural_proxy"
    assert ctx.group_rule_id == "synthetic_full_reply_or_title_v1"


@pytest.mark.parametrize("pair", [("T1", "D1"), ("T1", "E1"), ("D1", "E1")])
def test_all_partition_group_overlaps_are_forbidden(inputs, pair):
    groups = {q: q for q in inputs[1].seq_id}
    for q in pair:
        groups[q] = "same-group"
    with pytest.raises(ValueError, match="Group overlap"):
        resolve_data_context(groups_config(inputs, groups), root=inputs[0])


def test_groups_must_cover_all_tickets(inputs):
    with pytest.raises(ValueError, match="cover"):
        resolve_data_context(groups_config(inputs, {"T1": "g"}), root=inputs[0])


def test_changed_group_artifact_requires_repinning(inputs):
    root = inputs[0]
    cfg = groups_config(inputs)
    path = root / "groups.json"
    artifact = json.loads(path.read_text("utf-8"))
    artifact["rule_id"] = "different_rule"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_data_context(cfg, root=root)


def test_evidence_filters_both_endpoints_self_and_protocol(inputs):
    ctx = context_for(inputs)
    rows = feedback()
    original = rows.copy(deep=True)
    selected, audit = ctx.select_feedback(rows, protocol="conditioned")
    assert list(zip(selected.query_id, selected.candidate_id)) == [
        ("T1", "T2"), ("T2", "T3"), ("T3", "T1")]
    assert audit["excluded_outside_history"] == 2
    assert audit["excluded_self_pairs"] == 1
    assert audit["selected_rows"] == 3
    assert selected.score.mean() == pytest.approx(.6)
    pd.testing.assert_frame_equal(rows, original)
    blind, _ = ctx.select_feedback(rows, protocol="blind")
    assert blind.score.tolist() == [.4]


def test_train_loo_removes_target_in_both_roles(inputs):
    ctx = context_for(inputs, query_partition="train")
    assert ctx.history_for("T1") == {"T2", "T3"}
    with pytest.raises(ValueError, match="explicit query_id"):
        ctx.select_feedback(feedback(), protocol="conditioned")
    selected, audit = ctx.select_feedback(feedback(), protocol="conditioned", query_id="T1")
    assert list(zip(selected.query_id, selected.candidate_id)) == [("T2", "T3")]
    assert selected.score.mean() == .1  # Prior must be computed from these rows, not the source.
    assert audit["excluded_target_pairs"] == 2
    with pytest.raises(ValueError, match="outside"):
        ctx.history_for("E1")


def test_evidence_empty_and_nonunique_dataframe_index(inputs):
    ctx = context_for(inputs)
    empty, audit = ctx.select_feedback(feedback().iloc[:0], protocol="blind")
    assert empty.empty and audit["selected_rows"] == 0
    rows = feedback()
    rows.index = [0] * len(rows)
    selected, _ = ctx.select_feedback(rows, protocol="conditioned")
    assert selected.score.tolist() == [.9, .1, .8]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_scores_fail(inputs, value):
    rows = feedback()
    rows.loc[0, "score"] = value
    with pytest.raises(ValueError, match="finite"):
        context_for(inputs).select_feedback(rows, protocol="conditioned")


def test_unknown_ids_and_duplicate_allowed_pairs_fail(inputs):
    ctx = context_for(inputs)
    rows = feedback()
    rows.loc[0, "candidate_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        ctx.select_feedback(rows, protocol="conditioned")
    with pytest.raises(ValueError, match="Duplicate"):
        ctx.select_feedback(pd.concat([feedback(), feedback().iloc[:1]]), protocol="conditioned")


def test_missing_or_invalid_feedback_protocol_fails(inputs):
    ctx = context_for(inputs)
    with pytest.raises(ValueError, match="Missing feedback"):
        ctx.select_feedback(feedback().drop(columns="protocol"), protocol="conditioned")
    rows = feedback()
    rows.loc[0, "protocol"] = None
    with pytest.raises(ValueError, match="unsupported protocols"):
        ctx.select_feedback(rows, protocol="conditioned")


def test_nullable_missing_scores_are_not_silently_accepted(inputs):
    rows = feedback()
    rows["score"] = rows.score.astype("Float64")
    rows.loc[0, "score"] = pd.NA
    with pytest.raises(ValueError, match="finite"):
        context_for(inputs).select_feedback(rows, protocol="conditioned")


def test_reference_change_cannot_change_resolved_query_input(inputs):
    root, dataset, _ = inputs
    old = context_for(inputs).queries["D1"]
    dataset.loc[dataset.seq_id.eq("D1"), ["first_reply", "intent_class", "Team->Name"]] = "changed"
    dataset.to_parquet(root / "dataset.parquet", index=False)
    assert context_for(inputs).queries["D1"] == old


def test_write_once_and_no_nonfinite_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    write_once_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_once_json(path, {"a": 2})
    assert json.loads(path.read_text("utf-8")) == {"a": 1}
    with pytest.raises(ValueError):
        write_once_json(tmp_path / "invalid.json", {"a": float("nan")})
    assert not (tmp_path / "invalid.json").exists()


def test_legacy_serialization_is_not_extended_with_revision_defaults():
    for cfg in DEFAULT_METHODS.values():
        assert "data_protocol" not in cfg.to_dict()
        assert "schema_version" not in cfg.to_dict()
