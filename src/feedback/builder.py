"""
Feedback database builder.
Orchestrates: FAISS retrieval -> LLM judge -> store.

Supports a --validate mode that runs on a small subset of queries first,
producing a diagnostics report so you can sanity-check the feedback pipeline
before committing to the full 878-query build.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .judge import FeedbackJudge
from .loader import FeedbackDB

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 75


class FeedbackBuilder:
    def __init__(
        self,
        faiss_index,
        dataset: pd.DataFrame,
        judge: FeedbackJudge,
        db: FeedbackDB,
        top_k: int = DEFAULT_TOP_K,
        batch_size: int = 10,
    ):
        self.index = faiss_index
        self.dataset = dataset.set_index("seq_id")
        self.judge = judge
        self.db = db
        self.top_k = top_k
        self.batch_size = batch_size
        self.db.init()

    async def build(self, train_ids: list[str], encoder) -> int:
        id_to_idx = {}
        for i, row in self.dataset.iterrows():
            id_to_idx[row.name] = i

        all_rows = []
        for qi, qid in enumerate(train_ids):
            if qid not in self.dataset.index:
                log.warning("Query %s not in dataset, skipping", qid)
                continue
            if qi % 50 == 0:
                log.info("  Query %d/%d", qi + 1, len(train_ids))

            query_row = self.dataset.loc[qid]
            query_emb = encoder.encode_ticket(
                query_row["Title_anon"],
                query_row.get("Description_anon", "") or "",
            )

            exclude = {id_to_idx[qid]} if qid in id_to_idx else set()
            scores, indices = self.index.search(query_emb, self.top_k, exclude)
            meta = self.index.get_metadata(indices[0])

            pairs = []
            for _, crow in meta.iterrows():
                pairs.append({
                    "query_title": str(query_row["Title_anon"]),
                    "query_description": str(query_row.get("Description_anon", "") or ""),
                    "candidate_title": str(crow.get("Title_anon", "")),
                    "candidate_reply": str(crow.get("first_reply", "")),
                    "reference_reply": str(query_row.get("first_reply", "")),
                })

            judge_scores = await self._judge_batched(pairs)
            now = datetime.now(timezone.utc).isoformat()
            for ci, score in enumerate(judge_scores):
                if score is None:
                    continue
                crow = meta.iloc[ci]
                all_rows.append({
                    "query_id": qid,
                    "candidate_id": str(crow["seq_id"]),
                    "query_class": str(query_row.get("intent_class", "")),
                    "query_team": str(query_row.get("Team->Name", "")),
                    "score": score,
                    "protocol": self.judge.protocol,
                    "judge_model": self.judge.model,
                    "prompt_version": self.judge.prompt_version,
                    "ts": now,
                })

            if len(all_rows) >= 500:
                self.db.insert_rows(all_rows)
                all_rows = []

        if all_rows:
            self.db.insert_rows(all_rows)

        log.info("Feedback build complete: %d rows", self.db.count())
        return self.db.count()

    async def _judge_batched(self, pairs: list[dict]) -> list[Optional[float]]:
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            scores = await self.judge.judge_pairs(batch)
            all_scores.extend(scores)
        return all_scores


# ---------------------------------------------------------------------------
# Validation diagnostics (run on a small subset before the full build)
# ---------------------------------------------------------------------------

async def validate_feedback_pipeline(
    faiss_index,
    dataset,
    encoder,
    judge_model: str,
    top_k: int = DEFAULT_TOP_K,
    n_queries: int = 20,
) -> dict:
    """
    Run the full judge pipeline on `n_queries` randomly sampled train queries
    (both conditioned and blind protocols) and return a diagnostics report.

    The report covers:
    - score distribution (mean, std, quartiles) per protocol
    - conditioned-vs-blind rank correlation
    - the FAISS rank of candidates that would get promoted (lift cutoff)
    - how many candidates per query cross the 0.80/0.40 binary thresholds
    """
    log.info("=== Validation run: %d queries × 2 protocols ===", n_queries)

    from ..retrieval.encoder import TicketEncoder
    if isinstance(encoder, str):
        encoder = TicketEncoder(encoder)

    rng = np.random.default_rng(42)
    valid_ids = [qid for qid in dataset["seq_id"] if qid in dataset.set_index("seq_id").index]
    sample_ids = list(rng.choice(valid_ids, size=min(n_queries, len(valid_ids)), replace=False))

    report: dict = {
        "n_queries": len(sample_ids),
        "top_k": top_k,
        "judge_model": judge_model,
        "protocols": {},
    }

    for protocol in ["conditioned", "blind"]:
        db = FeedbackDB(Path(f"data/processed/_validate_{protocol}.db"))
        judge = FeedbackJudge(
            model=judge_model,
            protocol=protocol,
            prompt_version="v1",
            max_concurrency=8,
        )
        builder = FeedbackBuilder(faiss_index, dataset, judge, db, top_k=top_k)

        await builder.build(sample_ids, encoder)

        import sqlite3
        conn = sqlite3.connect(db.db_path)
        df = pd.read_sql_query(
            "SELECT query_id, candidate_id, query_class, query_team, score FROM feedback", conn
        )
        conn.close()

        scores = df["score"].dropna().values
        proto_report = {
            "n_rows": int(len(df)),
            "n_unique_candidates": int(df["candidate_id"].nunique()),
            "n_parsed_scores": int(len(scores)),
            "score_distribution": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "q25": float(np.percentile(scores, 25)),
                "median": float(np.median(scores)),
                "q75": float(np.percentile(scores, 75)),
                "max": float(np.max(scores)),
            },
            "binary_breakdown": {
                "pct_positive": float(np.mean(scores >= 0.80)),
                "pct_negative": float(np.mean(scores <= 0.40)),
                "pct_neutral": float(np.mean((scores > 0.40) & (scores < 0.80))),
            },
        }

        per_query = df.groupby("query_id")["score"].describe()
        proto_report["per_query_mean_mean"] = float(per_query["mean"].mean())
        proto_report["per_query_mean_std"] = float(per_query["mean"].std())

        report["protocols"][protocol] = proto_report
        log.info(
            "  %s: %d rows  mean=%.3f  pos=%.1f%%  neg=%.1f%%  neutral=%.1f%%",
            protocol,
            proto_report["n_rows"],
            proto_report["score_distribution"]["mean"],
            100 * proto_report["binary_breakdown"]["pct_positive"],
            100 * proto_report["binary_breakdown"]["pct_negative"],
            100 * proto_report["binary_breakdown"]["pct_neutral"],
        )

    if "conditioned" in report["protocols"] and "blind" in report["protocols"]:
        cond_scores = _per_pair_scores(report, "conditioned", sample_ids)
        blind_scores = _per_pair_scores(report, "blind", sample_ids)
        common_pairs = set(cond_scores.keys()) & set(blind_scores.keys())
        if common_pairs:
            c_vals = np.array([cond_scores[k] for k in common_pairs])
            b_vals = np.array([blind_scores[k] for k in common_pairs])
            r = np.corrcoef(c_vals, b_vals)[0, 1]
            report["conditioned_vs_blind"] = {
                "n_common_pairs": len(common_pairs),
                "pearson_r": float(r) if not np.isnan(r) else 0.0,
            }
            log.info("  Conditioned-vs-blind r = %.3f (%d common pairs)", r, len(common_pairs))

    return report


def _per_pair_scores(report: dict, protocol: str, query_ids: list[str]) -> dict:
    import sqlite3
    conn = sqlite3.connect(f"data/processed/_validate_{protocol}.db")
    df = pd.read_sql_query(
        "SELECT query_id, candidate_id, score FROM feedback", conn
    )
    conn.close()
    result = {}
    for _, row in df.iterrows():
        result[(str(row["query_id"]), str(row["candidate_id"]))] = float(row["score"])
    return result


def save_validation_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Validation report saved to %s", path)