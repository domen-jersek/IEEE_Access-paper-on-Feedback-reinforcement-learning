"""
Bulk evaluation runner with async concurrency, interim saves, and semaphore.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import EvalConfig
from ..generation.client import LLMClient
from ..evaluation.protocol import evaluate_one_ticket

log = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        config: EvalConfig,
        faiss_index,
        dataset: pd.DataFrame,
        encoder,
        feedback_scores: dict,
        results_dir: Path,
        concurrency: int = 4,
        interim_every: int = 10,
    ):
        self.config = config
        self.dataset = dataset.set_index("seq_id")
        self.faiss_index = faiss_index
        self.encoder = encoder
        self.feedback_scores = feedback_scores
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.concurrency = concurrency
        self.interim_every = interim_every
        self.llm_client = LLMClient(
            model=config.generator_model,
            temperature=config.temperature,
            max_concurrency=concurrency,
        )

    async def run(self, query_ids: list[str]) -> dict:
        sem = asyncio.Semaphore(self.concurrency)
        all_results = []
        results_lock = asyncio.Lock()

        async def process_one(qid: str):
            async with sem:
                try:
                    if qid not in self.dataset.index:
                        log.warning("Query %s not in dataset", qid)
                        return None
                    row = self.dataset.loc[qid]
                    result = await evaluate_one_ticket(
                        qid, row,
                        self.faiss_index, self.encoder,
                        self.feedback_scores, self.config, self.llm_client,
                    )
                    result["_config_hash"] = self.config.config_hash
                    async with results_lock:
                        all_results.append(result)
                        if len(all_results) % self.interim_every == 0:
                            self._save_interim(all_results)
                    return result
                except Exception:
                    log.exception("Error processing %s", qid)
                    return None

        tasks = [process_one(qid) for qid in query_ids]
        await asyncio.gather(*tasks)

        self._save_final(all_results)
        return self._compute_summary(all_results, query_ids)

    def _save_interim(self, results: list[dict]) -> None:
        path = self.results_dir / "_interim.json"
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_final(self, results: list[dict]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        detail_path = self.results_dir / f"{self.config.experiment_id}_{ts}_details.json"
        result_clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        detail_path.write_text(json.dumps(result_clean, indent=2, ensure_ascii=False), encoding="utf-8")

        valid = [r for r in results if r is not None]
        summary = {
            "experiment_id": self.config.experiment_id,
            "config": self.config.to_dict(),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(valid),
            "total_valid": len(valid),
            "total_skipped": 0,
            "metrics": self._compute_summary(valid, [r["ticket_id"] for r in valid]),
        }
        summary_path = self.results_dir / f"{self.config.experiment_id}_{ts}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _compute_summary(results: list[dict], query_ids: list[str]) -> dict:
        import numpy as np
        valid = [r for r in results if r is not None]
        if not valid:
            return {"error": "no valid results"}
        deltas = np.array([r["deltas"]["delta_cosine"] for r in valid])
        return {
            "n_tickets": len(valid),
            "mean_delta_cosine": float(np.mean(deltas)),
            "median_delta_cosine": float(np.median(deltas)),
            "std_delta_cosine": float(np.std(deltas)),
            "pct_improved": float(np.mean(deltas > 0)),
            "pct_worsened": float(np.mean(deltas < 0)),
            "pct_unchanged": float(np.mean(deltas == 0)),
        }