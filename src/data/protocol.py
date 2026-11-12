"""Revision data boundaries, resolved once and shared by pipeline consumers.

This module does not build models, aggregate feedback or generate answers.
References are deliberately absent from QueryInput and ResolvedDataContext.
Group artifacts specify structural guarantees, not unverified semantic claims.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from ..config import DataProtocolConfig
from .artifacts import ArtifactIdentity, content_id


def _unique_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration/artifact key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_json_keys)


def load_data_protocol(path: Path) -> DataProtocolConfig:
    """Load JSON/YAML with unknown/duplicate key rejection; no implicit overrides."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        values = read_json(path)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node):
            loader.flatten_mapping(node)
            return _unique_json_keys(loader.construct_pairs(node, deep=True))

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        values = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    else:
        raise ValueError("Data protocol configuration must be .json, .yaml or .yml")
    return DataProtocolConfig.from_dict(values)


def _ids(values, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list of IDs")
    if any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{label} contains a missing/non-string ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate IDs")
    return tuple(values)


def _text(value) -> str:
    return "" if pd.isna(value) else str(value)


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    title: str
    description: str
    routing_class: str | None = None
    routing_team: str | None = None
    metadata_policy: str = "text_only"


@dataclass(frozen=True)
class ResolvedDataContext:
    config: DataProtocolConfig
    artifacts: Mapping[str, ArtifactIdentity]
    partitions: Mapping[str, tuple[str, ...]]
    dataset_ids: frozenset[str]
    history_ids: frozenset[str]
    queries: Mapping[str, QueryInput]
    group_ids: Mapping[str, str]
    group_rule_id: str | None
    group_semantic_status: str | None
    context_id: str
    history_id: str

    @property
    def query_ids(self) -> tuple[str, ...]:
        return self.partitions[self.config.query_partition]

    def _check_query(self, query_id: str) -> None:
        if query_id not in self.queries:
            raise ValueError(f"Query is outside the requested partition: {query_id}")

    def history_for(self, query_id: str) -> frozenset[str]:
        self._check_query(query_id)
        return self.history_ids - {query_id}

    def select_feedback(self, rows: pd.DataFrame, *, protocol: str,
                        query_id: str | None = None) -> tuple[pd.DataFrame, dict]:
        """Select evidence before aggregation AND priors; never mutate source rows.

        The caller supplies one declared scoring regime. This selector does not
        establish judge-model/prompt provenance or claim complete top-k coverage.
        Train LOO removes the target in both roles, including prior evidence.
        """
        if protocol not in {"conditioned", "blind"}:
            raise ValueError(f"Unsupported feedback protocol: {protocol}")
        rows = rows.reset_index(drop=True)
        required = {"query_id", "candidate_id", "score", "protocol"}
        if missing := required - set(rows.columns):
            raise ValueError(f"Missing feedback columns: {sorted(missing)}")
        for name in ("query_id", "candidate_id"):
            if not rows[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"Invalid feedback {name}")
            if set(rows[name]) - self.dataset_ids:
                raise ValueError(f"Feedback {name} contains unknown dataset IDs")
        if not rows["protocol"].isin(["conditioned", "blind"]).all():
            raise ValueError("Feedback contains missing/unsupported protocols")
        if query_id is not None:
            self._check_query(query_id)
        elif self.config.query_partition == "train":
            raise ValueError("Train evidence requires an explicit query_id for leave-ticket-out")
        scores = pd.to_numeric(rows["score"], errors="raise")
        if scores.isna().any():
            raise ValueError("Feedback scores must be finite and within [0, 1]")
        scores = scores.astype(float)
        if not (np.isfinite(scores) & scores.between(0, 1)).all():
            raise ValueError("Feedback scores must be finite and within [0, 1]")

        selected = rows.loc[rows["protocol"].eq(protocol)].copy()
        audit = {"source_rows": len(rows), "protocol_rows": len(selected)}
        allowed = selected.query_id.isin(self.history_ids) & selected.candidate_id.isin(self.history_ids)
        audit["excluded_outside_history"] = int((~allowed).sum())
        selected = selected.loc[allowed]
        self_pair = selected.query_id.eq(selected.candidate_id)
        audit["excluded_self_pairs"] = int(self_pair.sum())
        selected = selected.loc[~self_pair]
        audit["excluded_target_pairs"] = 0
        if query_id is not None:
            target = selected.query_id.eq(query_id) | selected.candidate_id.eq(query_id)
            audit["excluded_target_pairs"] = int(target.sum())
            selected = selected.loc[~target]
        if selected.duplicated(["query_id", "candidate_id"]).any():
            raise ValueError("Duplicate allowed feedback pairs; declare one scoring regime before selection")
        selected = selected.copy()
        selected["score"] = scores.loc[selected.index].to_numpy()
        audit["selected_rows"] = len(selected)
        audit["covered_queries"] = selected.query_id.nunique()
        audit["covered_candidates"] = selected.candidate_id.nunique()
        return selected, audit

    def to_manifest(self) -> dict:
        return {
            "schema_version": 1,
            "revision_id": self.config.revision_id,
            "context_id": self.context_id,
            "history_id": self.history_id,
            "config": self.config.to_dict(),
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "partition_counts": {k: len(v) for k, v in self.partitions.items()},
            "query_ids": list(self.query_ids),
            "history_ids": sorted(self.history_ids),
            "group_rule_id": self.group_rule_id,
            "group_semantic_status": self.group_semantic_status,
            "status": "data_boundaries_validated_not_experiment_ready",
        }


def resolve_data_context(config: DataProtocolConfig, *, root: Path) -> ResolvedDataContext:
    """Read and validate pinned local inputs, without modifying any artifact."""
    root = Path(root).resolve()
    artifacts = {
        "dataset": ArtifactIdentity.verify(root / config.dataset_path, config.dataset_sha256),
        "split": ArtifactIdentity.verify(root / config.split_path, config.split_sha256),
    }
    columns = ["seq_id", "Title_anon", "Description_anon"]
    if config.metadata_policy == "posthoc_diagnostic":
        columns += ["intent_class", "Team->Name"]
    dataset = pd.read_parquet(artifacts["dataset"].path, columns=columns)
    dataset_ids = frozenset(_ids(dataset.seq_id.tolist(), "dataset.seq_id"))
    if not dataset.Title_anon.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError("Dataset contains missing/empty query titles")
    split = read_json(Path(artifacts["split"].path))
    if not isinstance(split, dict):
        raise ValueError("Split must be a mapping")
    partitions = {k: _ids(split.get(k), f"split.{k}") for k in ("train", "dev", "eval")}
    for a, b in combinations(partitions, 2):
        if set(partitions[a]) & set(partitions[b]):
            raise ValueError(f"Split ID overlap: {a}/{b}")
    if set().union(*partitions.values()) != dataset_ids:
        raise ValueError("Split must cover the dataset exactly, with no unknown IDs")

    groups = {}
    rule_id = semantic_status = None
    if config.groups_path is not None:
        artifacts["groups"] = ArtifactIdentity.verify(root / config.groups_path, config.groups_sha256)
        group_artifact = read_json(Path(artifacts["groups"].path))
        if not isinstance(group_artifact, dict) or set(group_artifact) != {
            "schema_version", "rule_id", "semantic_status", "groups"
        }:
            raise ValueError("Groups require schema_version, rule_id, semantic_status and groups")
        if type(group_artifact["schema_version"]) is not int or group_artifact["schema_version"] != 1:
            raise ValueError("Unsupported groups schema_version")
        rule_id = group_artifact["rule_id"]
        semantic_status = group_artifact["semantic_status"]
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError("Group rule_id must be nonempty")
        if not isinstance(semantic_status, str) or semantic_status not in {"structural_proxy", "domain_validated"}:
            raise ValueError("Unsupported group semantic_status")
        groups = group_artifact["groups"]
        if not isinstance(groups, dict) or set(groups) != dataset_ids:
            raise ValueError("Group mapping must cover the dataset exactly")
        if any(not isinstance(v, str) or not v.strip() for v in groups.values()):
            raise ValueError("Group IDs must be nonempty strings")
        if config.scenario == "held_out_groups":
            for a, b in combinations(partitions, 2):
                if {groups[q] for q in partitions[a]} & {groups[q] for q in partitions[b]}:
                    raise ValueError(f"Group overlap: {a}/{b}")

    queries = {}
    indexed = dataset.set_index("seq_id")
    for qid in partitions[config.query_partition]:
        row = indexed.loc[qid]
        metadata = {}
        if config.metadata_policy == "posthoc_diagnostic":
            for name in ("intent_class", "Team->Name"):
                if not _text(row[name]).strip():
                    raise ValueError(f"Missing posthoc routing metadata: {name}")
            metadata = {"routing_class": str(row["intent_class"]), "routing_team": str(row["Team->Name"])}
        queries[qid] = QueryInput(qid, str(row.Title_anon), _text(row.Description_anon),
                                  metadata_policy=config.metadata_policy, **metadata)
    # Physical locations and revision labels do not alter scientific identity.
    identity_config = {k: v for k, v in config.to_dict().items()
                       if not k.endswith("_path") and k != "revision_id"}
    history_ids = frozenset(partitions["train"])
    return ResolvedDataContext(
        config=config, artifacts=MappingProxyType(artifacts),
        partitions=MappingProxyType(partitions), dataset_ids=dataset_ids,
        history_ids=history_ids, queries=MappingProxyType(queries),
        group_ids=MappingProxyType(groups), group_rule_id=rule_id,
        group_semantic_status=semantic_status,
        context_id=content_id({"schema": "data_context_v1", "config": identity_config}),
        history_id=content_id({"schema": "history_v1", "dataset": config.dataset_sha256,
                               "ids": sorted(history_ids)}),
    )
