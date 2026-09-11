"""
Feedback database loader and aggregation.

Converts raw (query_id, candidate_id, score) rows into a nested
`feedback_scores` dict:

    feedback_scores[candidate_id]["global"] = {"pos": float, "neg": float}
    feedback_scores[candidate_id]["class:admin_rights"] = {...}
    feedback_scores[candidate_id]["team:(GI-UX) Group"] = {...}
    feedback_scores[candidate_id]["intersection:cls:team"] = {...}

Binarization: score >= 0.80 -> positive, score <= 0.40 -> negative, else neutral.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

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
    scores_df,
    pos_threshold: float = POS_THRESHOLD,
    neg_threshold: float = NEG_THRESHOLD,
) -> dict:
    """
    Aggregate raw scored rows into per-candidate, per-scope pos/neg dict.
    scores_df must have columns: query_id, candidate_id, query_class, query_team, score.
    """
    feedback_scores: dict[str, dict[str, dict[str, float]]] = {}

    def bucket(score: float):
        if score >= pos_threshold:
            return 1
        if score <= neg_threshold:
            return -1
        return 0

    for _, row in scores_df.iterrows():
        cid = str(row["candidate_id"])
        q_class = str(row.get("query_class", "") or "")
        q_team = str(row.get("query_team", "") or "")
        b = bucket(float(row["score"]))
        if b == 0:
            continue

        entry = feedback_scores.setdefault(cid, {})

        g = entry.setdefault("global", {"pos": 0.0, "neg": 0.0})
        if b > 0:
            g["pos"] += 1.0
        else:
            g["neg"] += 1.0

        if q_class:
            c = entry.setdefault(f"class:{q_class}", {"pos": 0.0, "neg": 0.0})
            if b > 0:
                c["pos"] += 1.0
            else:
                c["neg"] += 1.0

        if q_team:
            t = entry.setdefault(f"team:{q_team}", {"pos": 0.0, "neg": 0.0})
            if b > 0:
                t["pos"] += 1.0
            else:
                t["neg"] += 1.0

        if q_class and q_team:
            isec_key = f"intersection:{q_class}:{q_team}"
            i = entry.setdefault(isec_key, {"pos": 0.0, "neg": 0.0})
            if b > 0:
                i["pos"] += 1.0
            else:
                i["neg"] += 1.0

    return feedback_scores


def load_feedback_as_scores(
    db_path: Path,
    exclude_query_ids: Optional[set[str]] = None,
) -> dict:
    conn = sqlite3.connect(db_path)
    import pandas as pd
    if exclude_query_ids:
        placeholders = ",".join("?" for _ in exclude_query_ids)
        query = f"SELECT query_id, candidate_id, query_class, query_team, score FROM feedback WHERE query_id NOT IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=list(exclude_query_ids))
    else:
        df = pd.read_sql_query(
            "SELECT query_id, candidate_id, query_class, query_team, score FROM feedback", conn
        )
    conn.close()
    return aggregate_feedback_scores(df)


def export_scores(scores: dict, path: Path) -> None:
    path.write_text(json.dumps(scores), encoding="utf-8")