"""Generation-regime provenance: stable IDs, cache accounting, warm flag."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.client import LLMClient
from src.generation.regime import cache_row_count, generation_regime, regime_id, system_prompt_sha256


def test_regime_id_stable_and_sensitive():
    base = regime_id("openai/gpt-5.6-luna", 0.0)
    assert base == regime_id("openai/gpt-5.6-luna", 0.0)
    assert base != regime_id("openai/gpt-5.6-luna", 1.0)
    assert base != regime_id("openai/other-model", 0.0)
    assert base != regime_id("openai/gpt-5.6-luna", 0.0, system_prompt="different system prompt")
    assert len(base) == 16


def test_system_prompt_sha_is_recorded():
    block = generation_regime("m", cache_path=None, temperature=0.0)
    assert block["system_prompt_sha256"] == system_prompt_sha256()
    assert block["cache_enabled"] is False
    assert block["warm"] is None


def test_cache_row_count_and_warm_flag(tmp_path):
    cache = tmp_path / "c.db"
    client = LLMClient("m", cache_path=cache)
    assert client.cache_rows_before == 0
    client._cache_put(client._cache_key("p1", ""), "a")
    client._cache_put(client._cache_key("p2", ""), "b")
    assert client.cache_row_count() == 2
    assert cache_row_count(cache) == 2

    warm = generation_regime("m", cache_path=cache, temperature=0.0,
                             cache_rows_before=0, cache_rows_after=2, hits=2, misses=0)
    assert warm["warm"] is True
    cold = generation_regime("m", cache_path=cache, temperature=0.0,
                             cache_rows_before=0, cache_rows_after=2, hits=1, misses=1)
    assert cold["warm"] is False


def test_missing_cache_returns_zero(tmp_path):
    assert cache_row_count(tmp_path / "does_not_exist.db") == 0
