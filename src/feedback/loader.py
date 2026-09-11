"""
Feedback database loader and aggregation.

Converts raw (query_id, candidate_id, score) rows into a nested
`feedback_scores` dict:

    feedback_scores[candidate_id]["global"] = {"pos": float, "neg": float}
    feedback_scores[candidate_id]["class:admin_rights"] = {...}
    feedback_scores[candidate_id]["team:(GI-UX) Group"] = {...}
    feedback_scores[candidate_id]["intersection:cls:team"] = {...}

Two aggregation modes:

  "continuous" (default, recommended):
    Every query's score contributes directly. pos = sum(scores), neg = n - sum(scores).
    A candidate rated 0.75 by 100 queries gets pos=75, neg=25 (strong boost).
    A candidate rated 0.50 by 100 queries gets pos=50, neg=50 (near-zero lift).
    No information is discarded or thresholded away.

  "binary":
    score >= 0.80 -> positive (+1), score <= 0.40 -> negative (+1), else ignored.
    This keeps the neutral band defined in the original SIKDD methodology.
    Candidates with middling scores (0.79 by 100 queries) get zero lift.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

log = logging.getLogger(__name__)

POS_THRESHOLD = 0.80
NEG_THRESHOLD = 0.40


class FeedbackDB:
    SCHEMA = """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            query_class TEXT,
            query_team TEXT,
            score REAL NOT NULL,
            protocol TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            ts TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_candidate ON feedback(candidate_id);
        CREATE INDEX IF NOT EXISTS idx_query ON feedback(query_id);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.executescript(self.SCHEMA)
        conn.commit()
        conn.close()

    def insert_rows(self, rows: list[dict]) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            "INSERT INTO feedback (query_id, candidate_id, query_class, query_team, score, protocol, judge_model, prompt_version, ts) "
            "VALUES (:query_id, :candidate_id, :query_class, :query_team, :score, :protocol, :judge_model, :prompt_version, :ts)",
            rows,
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        return int(count)

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        return int(n)

    def distinct_queries(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(DISTINCT query_id) FROM feedback").fetchone()[0]
        conn.close()
        return int(n)

    def distinct_candidates(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(DISTINCT candidate_id) FROM feedback").fetchone()[0]
        conn.close()
        return int(n)


def aggregate_feedback_scores(
    scores_df: pd.DataFrame,
    mode: Literal["continuous", "binary"] = "continuous",
    pos_threshold: float = POS_THRESHOLD,
    neg_threshold: float = NEG_THRESHOLD,
) -> dict:
    """
    Aggregate raw scored rows into per-candidate, per-scope pos/neg dict.

    Parameters
    ----------
    scores_df : DataFrame
        Required columns: candidate_id, query_id, query_class, query_team, score.
    mode : str
        "continuous" (default): pos += score, neg += (1 - score) per row.
        "binary": pos += 1 if score >= pos_threshold; neg += 1 if score <= neg_threshold.

    `pos` and `neg` are floats regardless of mode; the downstream lift functions
    (`src/feedback/lift.py`) operate on them identically.
    """
    feedback_scores: dict[str, dict[str, dict[str, float]]] = {}

    def contrib(score: float) -> tuple[float, float]:
        if mode == "continuous":
            return score, 1.0 - score
        if score >= pos_threshold:
            return 1.0, 0.0
        if score <= neg_threshold:
            return 0.0, 1.0
        return 0.0, 0.0

    for _, row in scores_df.iterrows():
        cid = str(row["candidate_id"])
        q_class = str(row.get("query_class", "") or "")
        q_team = str(row.get("query_team", "") or "")
        p, n = contrib(float(row["score"]))
        if p == 0.0 and n == 0.0:
            continue

        entry = feedback_scores.setdefault(cid, {})

        g = entry.setdefault("global", {"pos": 0.0, "neg": 0.0})
        g["pos"] += p
        g["neg"] += n

        if q_class:
            c = entry.setdefault(f"class:{q_class}", {"pos": 0.0, "neg": 0.0})
            c["pos"] += p
            c["neg"] += n

        if q_team:
            t = entry.setdefault(f"team:{q_team}", {"pos": 0.0, "neg": 0.0})
            t["pos"] += p
            t["neg"] += n

        if q_class and q_team:
            isec_key = f"intersection:{q_class}:{q_team}"
            i = entry.setdefault(isec_key, {"pos": 0.0, "neg": 0.0})
            i["pos"] += p
            i["neg"] += n

    return feedback_scores


def load_feedback_as_scores(
    db_path: Path,
    exclude_query_ids: Optional[set[str]] = None,
    protocol: Optional[str] = None,
    mode: Literal["continuous", "binary"] = "continuous",
) -> dict:
    conn = sqlite3.connect(db_path)
    params: list = []
    sql = "SELECT query_id, candidate_id, query_class, query_team, score, protocol FROM feedback"
    conds = []
    if exclude_query_ids:
        placeholders = ",".join("?" for _ in exclude_query_ids)
        conds.append(f"query_id NOT IN ({placeholders})")
        params.extend(exclude_query_ids)
    if protocol:
        conds.append("protocol = ?")
        params.append(protocol)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return aggregate_feedback_scores(df, mode=mode)


def export_scores(scores: dict, path: Path) -> None:
    path.write_text(json.dumps(scores), encoding="utf-8")