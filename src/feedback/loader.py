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
    required = {"candidate_id", "score"}
    missing = required.difference(scores_df.columns)
    if missing:
        raise ValueError(f"Feedback rows are missing required columns: {sorted(missing)}")
    if mode not in {"continuous", "binary"}:
        raise ValueError(f"Unsupported feedback aggregation mode: {mode}")
    if scores_df.empty:
        return {}

    df = scores_df[["candidate_id", "score"]].copy()
    df["candidate_id"] = df["candidate_id"].astype(str)
    for column in ("query_class", "query_team"):
        if column in scores_df:
            df[column] = scores_df[column].fillna("").astype(str)
        else:
            df[column] = ""

    score = pd.to_numeric(df["score"], errors="raise").astype(float)
    if mode == "continuous":
        df["pos"] = score
        df["neg"] = 1.0 - score
    else:
        df["pos"] = (score >= pos_threshold).astype(float)
        df["neg"] = (score <= neg_threshold).astype(float)
        df = df[(df["pos"] != 0.0) | (df["neg"] != 0.0)]
    if df.empty:
        return {}

    frames = [df.assign(scope="global")]
    class_rows = df[df["query_class"] != ""]
    team_rows = df[df["query_team"] != ""]
    both = df[(df["query_class"] != "") & (df["query_team"] != "")]
    if not class_rows.empty:
        frames.append(class_rows.assign(scope="class:" + class_rows["query_class"]))
    if not team_rows.empty:
        frames.append(team_rows.assign(scope="team:" + team_rows["query_team"]))
    if not both.empty:
        frames.append(both.assign(scope="intersection:" + both["query_class"] + ":" + both["query_team"]))

    grouped = (
        pd.concat(frames, ignore_index=True)
        .groupby(["candidate_id", "scope"], sort=False, observed=True)[["pos", "neg"]]
        .sum()
    )
    feedback_scores: dict[str, dict[str, dict[str, float]]] = {}
    for (candidate_id, scope), values in grouped.iterrows():
        feedback_scores.setdefault(str(candidate_id), {})[str(scope)] = {
            "pos": float(values["pos"]),
            "neg": float(values["neg"]),
        }
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


# ---------------------------------------------------------------------------
# P1.5 / P3 additions: scope priors, bundle object, per-query LOO subtraction.
# `load_feedback_as_scores` above is unchanged (used by the SIKDD replication runs).
# ---------------------------------------------------------------------------

def _scope_keys(q_class: str, q_team: str) -> list[str]:
    keys = ["global"]
    if q_class:
        keys.append(f"class:{q_class}")
    if q_team:
        keys.append(f"team:{q_team}")
    if q_class and q_team:
        keys.append(f"intersection:{q_class}:{q_team}")
    return keys


def compute_scope_priors(scores_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Empirical mean judge score per scope key (global, class:X, team:Y,
    intersection:X:Y) over ALL (query, candidate) rows in the frame.
    Returned as {scope_key: {"mean": p_bar, "n": rows}}.
    Used as the centring point of `laplace_eb`.
    """
    df = scores_df.copy()
    df["query_class"] = df["query_class"].fillna("").astype(str)
    df["query_team"] = df["query_team"].fillna("").astype(str)
    out: dict[str, dict[str, float]] = {}
    out["global"] = {"mean": float(df["score"].mean()), "n": float(len(df))}
    for col, prefix in (("query_class", "class"), ("query_team", "team")):
        g = df[df[col] != ""].groupby(col)["score"].agg(["mean", "count"])
        for k, row in g.iterrows():
            out[f"{prefix}:{k}"] = {"mean": float(row["mean"]), "n": float(row["count"])}
    both = df[(df["query_class"] != "") & (df["query_team"] != "")]
    g = both.groupby(["query_class", "query_team"])["score"].agg(["mean", "count"])
    for (c, t), row in g.iterrows():
        out[f"intersection:{c}:{t}"] = {"mean": float(row["mean"]), "n": float(row["count"])}
    return out


def load_feedback_rows(
    db_path: Path,
    exclude_query_ids: Optional[set[str]] = None,
    protocol: Optional[str] = None,
) -> pd.DataFrame:
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
    return df


class FeedbackBundle:
    """
    feedback_scores dict + per-scope priors + the raw rows (for per-query LOO).

    `scores`  : same structure as load_feedback_as_scores() output.
    `priors`  : compute_scope_priors() output.
    `prior_mean(scope_key)` : p_bar for a scope, falling back to the global mean.
    `minus_query(qid)`      : scores with that query's own contributions removed
                              (true leave-one-out when evaluating TRAIN queries whose
                              judgements are in the DB).
    """

    def __init__(self, rows: pd.DataFrame, mode: str = "continuous"):
        self.mode = mode
        self.rows = rows
        self.scores = aggregate_feedback_scores(rows, mode=mode)
        self.priors = compute_scope_priors(rows)
        self._by_query: Optional[dict] = None

    def prior_mean(self, scope_key: str) -> float:
        p = self.priors.get(scope_key)
        if p is None:
            p = self.priors.get("global", {"mean": 0.5})
        return float(p["mean"])

    def _contrib(self, score: float) -> tuple[float, float]:
        if self.mode == "continuous":
            return score, 1.0 - score
        if score >= POS_THRESHOLD:
            return 1.0, 0.0
        if score <= NEG_THRESHOLD:
            return 0.0, 1.0
        return 0.0, 0.0

    def minus_query(self, query_id: str) -> dict:
        """
        Return a *view-like copy* of `scores` in which the rows contributed by
        `query_id` (as a query) are subtracted. Only candidates touched by that
        query are copied; the rest reference the shared dicts (read-only use).
        """
        if self._by_query is None:
            self._by_query = {k: v for k, v in self.rows.groupby("query_id")}
        sub = self._by_query.get(query_id)
        if sub is None or len(sub) == 0:
            return self.scores
        out = dict(self.scores)  # shallow copy of the candidate map
        for _, row in sub.iterrows():
            cid = str(row["candidate_id"])
            entry = self.scores.get(cid)
            if entry is None:
                continue
            p, n = self._contrib(float(row["score"]))
            if p == 0.0 and n == 0.0:
                continue
            new_entry = {k: dict(v) for k, v in entry.items()}
            for key in _scope_keys(str(row.get("query_class", "") or ""), str(row.get("query_team", "") or "")):
                if key in new_entry:
                    new_entry[key]["pos"] = max(0.0, new_entry[key]["pos"] - p)
                    new_entry[key]["neg"] = max(0.0, new_entry[key]["neg"] - n)
            out[cid] = new_entry
        return out


def load_feedback_bundle(
    db_path: Path,
    exclude_query_ids: Optional[set[str]] = None,
    protocol: Optional[str] = None,
    mode: Literal["continuous", "binary"] = "continuous",
) -> FeedbackBundle:
    rows = load_feedback_rows(db_path, exclude_query_ids=exclude_query_ids, protocol=protocol)
    return FeedbackBundle(rows, mode=mode)
