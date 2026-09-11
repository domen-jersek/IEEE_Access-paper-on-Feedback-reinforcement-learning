"""
Shared CLI utilities for experiment scripts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def load_dataset() -> pd.DataFrame:
    paths = ProjectPaths()
    return pd.read_parquet(paths.dataset_parquet)


def load_split(seed: int = 42, regime: str = "random") -> dict:
    paths = ProjectPaths()
    suffix = f"_{regime}" if regime != "random" else ""
    path = paths.splits_dir / f"split_seed{seed}{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return json.loads(path.read_text("utf-8"))


def get_arg_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--regime", type=str, default="random", choices=["random", "disjoint"])
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser