"""Cache-only operation: a fully-cached run must not require an API key."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.client import LLMClient


def _clear_keys(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_cached_response_without_api_key(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    client = LLMClient("m", cache_path=tmp_path / "c.db")
    client._cache_put(client._cache_key("hello", ""), "world")
    assert asyncio.run(client.complete("hello", "")) == "world"
    assert client.cache_hits == 1


def test_cache_miss_without_api_key_raises(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    client = LLMClient("m", cache_path=tmp_path / "c.db", max_retries=1)
    with pytest.raises(RuntimeError):
        asyncio.run(client.complete("not cached", ""))