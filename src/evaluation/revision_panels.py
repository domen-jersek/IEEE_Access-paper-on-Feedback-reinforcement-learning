"""Shared construction of revision panels (history-only, pinned inputs).

Used by the retrieval-panel script and the dev lab so that every offline analysis
resolves the same data context, the same pools and the same metrics. No model or
API call is made here.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import DataProtocolConfig, ProjectPaths
from ..data.artifacts import file_sha256
from ..data.groups_v2 import normalize_signature
from ..data.protocol import resolve_data_context
from ..feedback.loader import FeedbackBundle

SCOPES = ("global", "class", "team", "intersection")


def panel_specs(paths: ProjectPaths, split_seed: int = 42) -> dict[str, dict]:
    """Named data regimes; paths are resolved relative to the project root."""
    revision = paths.data_processed / "revision"
    suffix = "" if split_seed == 42 else f"_seed{split_seed}"
    return {
        "legacy_random": {
            "split": paths.splits_dir / "split_seed42.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "scenario": "familiar_procedure_possible",
        },
        "legacy_disjoint": {
            "split": paths.splits_dir / "split_seed42_disjoint.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "scenario": "familiar_procedure_possible",
        },
        f"disjoint_duplicate{suffix}": {
            "split": revision / "splits" / f"split_duplicate_template_v1_seed{split_seed}.json",
            "groups": revision / "groups" / "duplicate_template_v1.json",
            "scenario": "held_out_groups",
        },
        f"disjoint_prefix{suffix}": {
            "split": revision / "splits" / f"split_prefix_template_v1_seed{split_seed}.json",
            "groups": revision / "groups" / "prefix_template_v1.json",
            "scenario": "held_out_groups",
        },
    }


def resolve_panel(panel: dict, paths: ProjectPaths, partition: str = "eval"):
    """Resolve a pinned data context for one regime and one query partition."""
    if not 0 < len(partition) and partition not in {"train", "dev", "eval"}:
        raise ValueError(f"Unsupported partition: {partition}")
    config = DataProtocolConfig(
        dataset_path=str(paths.dataset_parquet.relative_to(paths.root)).replace("\\", "/"),
        dataset_sha256=file_sha256(paths.dataset_parquet),
        split_path=str(panel["split"].relative_to(paths.root)).replace("\\", "/"),
        split_sha256=file_sha256(panel["split"]),
        groups_path=str(panel["groups"].relative_to(paths.root)).replace("\\", "/"),
        groups_sha256=file_sha256(panel["groups"]),
        scenario=panel["scenario"],
        metadata_policy="posthoc_diagnostic",
        query_partition=partition,
    )
    return resolve_data_context(config, root=paths.root)


def ticket_embeddings(dataset: pd.DataFrame, paths: ProjectPaths) -> np.ndarray:
    """Frozen query embeddings, verified against the canonical ticket order."""
    cache = paths.root / "data" / "cache"
    ids = json.loads((cache / "ticket_text_emb_minilm_ids.json").read_text("utf-8"))
    if ids != dataset["seq_id"].astype(str).tolist():
        raise ValueError("Cached ticket embeddings are not aligned with the canonical dataset")
    matrix = np.load(cache / "ticket_text_emb_minilm.npy")
    if matrix.shape[0] != len(dataset):
        raise ValueError("Cached ticket embedding matrix has the wrong shape")
    return matrix.astype(np.float32)


def collect_pools(context, dataset, embeddings, retriever, search_k: int) -> pd.DataFrame:
    """Top-k history-only candidates for every evaluation query of the context."""
    position = {seq_id: i for i, seq_id in enumerate(dataset["seq_id"].astype(str))}
    rows = []
    for qid in context.query_ids:
        if qid in context.history_ids:
            raise AssertionError(f"Query {qid} is inside the allowed history")
        row = dataset.iloc[position[qid]]
        description = str(row["Description_anon"] or "")
        text = f"{row['Title_anon']}\n{description}" if description else str(row["Title_anon"])
        pool = retriever.search_pool(text, embeddings[position[qid]][None, :], search_k, set())
        for record in pool.itertuples(index=False):
            rows.append((qid, str(record.seq_id), float(record.faiss_score), int(record.faiss_rank)))
    pools = pd.DataFrame(rows, columns=["query_id", "seq_id", "faiss_score", "faiss_rank"])
    forbidden = set(pools["seq_id"]) - set(context.history_ids)
    if forbidden:
        raise AssertionError(f"Pool contains {len(forbidden)} candidates outside history")
    return pools


def load_feedback_rows(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        return pd.read_sql_query(
            "SELECT query_id, candidate_id, query_class, query_team, score, protocol FROM feedback",
            connection,
        )


def judge_bundle(context, rows: pd.DataFrame, protocol: str) -> tuple[FeedbackBundle, dict]:
    allowed, audit = context.select_feedback(rows, protocol=protocol)
    return FeedbackBundle(allowed), audit


def bootstrap(delta: np.ndarray, n_resamples: int = 5000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(n_resamples, len(delta)))
    sampled = delta[draws]
    means = sampled.mean(axis=1)
    medians = np.median(sampled, axis=1)
    return {
        "mean": float(delta.mean()),
        "mean_ci_lower": float(np.percentile(means, 2.5)),
        "mean_ci_upper": float(np.percentile(means, 97.5)),
        "median": float(np.median(delta)),
        "median_ci_lower": float(np.percentile(medians, 2.5)),
        "median_ci_upper": float(np.percentile(medians, 97.5)),
    }


def p_value(delta: np.ndarray) -> float:
    if not np.any(delta != 0):
        return float("nan")
    return float(stats.wilcoxon(delta).pvalue)


def pool_overlap(context, dataset, pools: pd.DataFrame, group_map: dict[str, str]) -> pd.DataFrame:
    frame = dataset.set_index("seq_id")
    title = frame["Title_anon"].map(normalize_signature)
    subcategory = frame["Service subcategory->Name"].map(normalize_signature)
    families = {
        "v2_group": group_map,
        "full_reply": dict(zip(frame.index, frame["first_reply"].map(normalize_signature))),
        "reply_prefix300": dict(zip(frame.index, frame["first_reply"].fillna("")
                                    .astype(str).str[:300].map(normalize_signature))),
        "title_subcategory": dict(zip(frame.index, (title + " | " + subcategory).where(
            ~((title == "") & (subcategory == "")), ""))),
    }
    rows = []
    grouped = {qid: g for qid, g in pools.groupby("query_id")}
    for family, mapping in families.items():
        affected = 0
        excess = 0
        for qid in context.query_ids:
            own = mapping.get(qid)
            matches = int(sum(1 for cid in grouped[qid]["seq_id"] if mapping.get(cid) == own))
            affected += matches > 0
            excess += matches
        rows.append({
            "family": family,
            "queries_with_shared_candidate": affected,
            "fraction_queries": affected / len(context.query_ids),
            "shared_candidates_total": excess,
            "pool_slots": int(len(pools)),
        })
    return pd.DataFrame(rows)


def saturation(tensor, lift_config, routing) -> dict:
    lift = tensor.routed_lift(lift_config, routing)
    factor = tensor.scale_factor(lift_config)
    mask = tensor.mask
    effective = np.abs(lift * factor[:, None])
    cap = lift_config.cap
    at_cap = np.isclose(np.abs(lift), cap) & mask
    scores = np.where(mask, tensor.score, np.nan)
    score_std = np.nanstd(scores, axis=1, ddof=1)
    lift_std = np.nanstd(np.where(mask, lift * factor[:, None], np.nan), axis=1, ddof=1)
    ratio = lift_std / np.where(score_std > 0, score_std, np.nan)
    return {
        "pct_candidates_at_cap": float(at_cap.sum() / max(mask.sum(), 1)),
        "pct_candidates_zero": float((np.isclose(lift, 0.0) & mask).sum() / max(mask.sum(), 1)),
        "mean_abs_lift": float(np.nanmean(np.where(mask, effective, np.nan))),
        "mean_lift_std_over_score_std": float(np.nanmean(ratio)),
    }
