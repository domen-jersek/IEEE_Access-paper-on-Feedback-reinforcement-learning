"""
Feedback database builder.
Orchestrates: FAISS retrieval -> LLM judge -> store.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from .judge import FeedbackJudge
from .loader import FeedbackDB

log = logging.getLogger(__name__)


class FeedbackBuilder:
    def __init__(
        self,
        faiss_index,
        dataset: pd.DataFrame,
        judge: FeedbackJudge,
        db: FeedbackDB,
        top_k: int = 100,
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
        for qid in train_ids:
            if qid not in self.dataset.index:
                log.warning("Query %s not in dataset, skipping", qid)
                continue
            query_row = self.dataset.loc[qid]
            query_text = f"{query_row['Title_anon']}\n{query_row.get('Description_anon', '') or ''}"
            query_emb = encoder.encode_ticket(query_row["Title_anon"], query_row.get("Description_anon", "") or "")

            exclude = {id_to_idx[qid]} if qid in id_to_idx else set()
            scores, indices = self.index.search(query_emb, self.top_k, exclude)
            meta = self.index.get_metadata(indices[0])

            pairs = []
            for ci, (_, crow) in enumerate(meta.iterrows()):
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