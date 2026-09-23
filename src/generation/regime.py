"""
Generation-regime provenance.

A *generation regime* is the tuple (generator model, temperature, system prompt).
Runs that share a regime and whose generation cache had zero misses reuse the
same cached answers and are therefore mutually comparable. Runs from different
regimes must never be compared as if they had been generated under identical
conditions (the SIKDD replication runs before 2026-09-17 were uncached and are
therefore not reproducible bit-for-bit).

Every evaluation summary and manifest records:

    regime_id             stable hash of (model, temperature, system prompt)
    system_prompt_sha256  so a prompt edit changes the regime
    cache_path            the SQLite cache used for this run (or None)
    cache_rows_before/after  size of the shared cache at run start/end
    cache_hits/misses     how many prompts were served from / written to the cache
    warm                  True iff the run had no cache misses (fully anchored)

A run with ``warm=False`` may mix answers generated at different times and must
be regenerated before it is used as evidence.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

from .prompts import SYSTEM_PROMPT


def system_prompt_sha256(system_prompt: str = SYSTEM_PROMPT, length: int = 16) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:length]


def cache_row_count(cache_path: Optional[Path]) -> Optional[int]:
    """Number of cached responses, or None when the cache is disabled/missing."""
    if cache_path is None:
        return None
    path = Path(cache_path)
    if not path.exists():
        return 0
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def regime_id(model: str, temperature: float, system_prompt: str = SYSTEM_PROMPT) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def generation_regime(
    model: str,
    cache_path: Optional[Path] = None,
    temperature: float = 0.0,
    system_prompt: str = SYSTEM_PROMPT,
    cache_rows_before: Optional[int] = None,
    cache_rows_after: Optional[int] = None,
    hits: Optional[int] = None,
    misses: Optional[int] = None,
) -> dict:
    """Provenance block for a run's summary/manifest."""
    cache_enabled = cache_path is not None
    warm: Optional[bool] = None
    if cache_enabled and misses is not None:
        warm = misses == 0
    return {
        "regime_id": regime_id(model, temperature, system_prompt),
        "model": model,
        "temperature": temperature,
        "system_prompt_sha256": system_prompt_sha256(system_prompt),
        "cache_path": str(cache_path) if cache_enabled else None,
        "cache_enabled": cache_enabled,
        "cache_rows_before": cache_rows_before,
        "cache_rows_after": cache_rows_after,
        "cache_hits": hits,
        "cache_misses": misses,
        "warm": warm,
    }
