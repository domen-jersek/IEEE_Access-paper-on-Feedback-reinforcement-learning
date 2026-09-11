"""
LLM-as-judge feedback generation protocol.

Two variants:
- "conditioned": judge sees query + ground-truth reply + candidate reply.
- "blind":       judge sees only query title/description + candidate reply
                 (simulated real user/expert feedback).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..generation.client import LLMClient
from ..generation.prompts import build_judge_prompt

log = logging.getLogger(__name__)


class FeedbackJudge:
    def __init__(
        self,
        model: str,
        protocol: str,
        prompt_version: str = "v1",
        cache_path: Optional[Path] = None,
        max_concurrency: int = 8,
    ):
        self.model = model
        self.protocol = protocol
        self.prompt_version = prompt_version
        self.client = LLMClient(
            model=model,
            temperature=0.0,
            cache_path=cache_path,
            max_concurrency=max_concurrency,
        )

    async def judge_pairs(self, pairs: list[dict]) -> list[float]:
        prompts = []
        for p in pairs:
            prompts.append(
                build_judge_prompt(
                    protocol=self.protocol,
                    query_title=p["query_title"],
                    query_description=p.get("query_description", ""),
                    candidate_title=p.get("candidate_title", ""),
                    candidate_reply=p.get("candidate_reply", ""),
                    reference_reply=p.get("reference_reply", ""),
                )
            )

        responses = await self.client.complete_batch(prompts)
        scores = []
        for text in responses:
            score = self.client.parse_score(text)
            if score is None:
                log.warning("Could not parse judge score from: %r", text[:60])
                score = None
            scores.append(score)
        return scores