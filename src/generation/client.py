"""
Async OpenRouter / OpenAI client with temperature=0, retries, and optional cache.
No hardcoded keys: reads OPENROUTER_API_KEY / OPENAI_API_KEY from the environment.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _get_api_key(required: bool = True) -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key and required:
        raise RuntimeError(
            "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment."
        )
    return key


def _get_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)


class LLMClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        cache_path: Optional[Path] = None,
        max_concurrency: int = 8,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.cache_hits = 0
        self.cache_misses = 0
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # The API client is built lazily so a run whose prompts are all cached does
        # not require an API key (useful for reproducing gated/ablated runs offline).
        self._api_key = _get_api_key(required=False)
        self._client = None
        self._cache_path = cache_path
        self.cache_rows_before: Optional[int] = None
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_cache()
            self.cache_rows_before = self.cache_row_count()

    def cache_row_count(self) -> Optional[int]:
        """Current number of cached responses (read-only), or None when disabled."""
        if not self._cache_path:
            return None
        conn = sqlite3.connect(f"file:{Path(self._cache_path).as_posix()}?mode=ro", uri=True)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "No API key found and the requested prompt is not in the cache. "
                    "Set OPENROUTER_API_KEY or OPENAI_API_KEY to make new calls."
                )
            self._client = AsyncOpenAI(api_key=self._api_key, base_url=_get_base_url())
        return self._client

    def _init_cache(self) -> None:
        conn = sqlite3.connect(self._cache_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, model TEXT, response TEXT, ts TEXT)"
        )
        conn.commit()
        conn.close()

    def _cache_key(self, prompt: str, system: str = "") -> str:
        # NOTE: the system prompt is included only when non-empty, so keys for the
        # judge cache (which never passes a system prompt) are unchanged.
        payload = {"model": self.model, "temperature": self.temperature, "prompt": prompt}
        if system:
            payload["system"] = system
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self._cache_path:
            return None
        conn = sqlite3.connect(self._cache_path)
        row = conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        conn.close()
        return row[0] if row else None

    def _cache_put(self, key: str, response: str) -> None:
        if not self._cache_path:
            return
        conn = sqlite3.connect(self._cache_path)
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, model, response, ts) VALUES (?, ?, ?, datetime('now'))",
            (key, self.model, response),
        )
        conn.commit()
        conn.close()

    async def complete(self, prompt: str, system: str = "") -> str:
        key = self._cache_key(prompt, system)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        async with self._semaphore:
            last_err = None
            for attempt in range(self.max_retries):
                try:
                    messages = []
                    if system:
                        messages.append({"role": "system", "content": system})
                    messages.append({"role": "user", "content": prompt})
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    text = response.choices[0].message.content or ""
                    self._cache_put(key, text)
                    return text
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(2 ** attempt)
            raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    async def complete_batch(self, prompts: list[str], system: str = "") -> list[str]:
        tasks = [self.complete(p, system=system) for p in prompts]
        return await asyncio.gather(*tasks)

    @staticmethod
    def parse_score(text: str) -> Optional[float]:
        """Extract a 0-1 float from a judge response."""
        import re
        if not text:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if not match:
            return None
        value = float(match.group(1))
        if value > 1.0:
            value = value / 100.0 if value <= 100 else 1.0
        return max(0.0, min(1.0, value))