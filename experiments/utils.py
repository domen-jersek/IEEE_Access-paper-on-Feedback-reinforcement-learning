"""
Shared CLI utilities for experiment scripts.

Reproducibility helpers (P0):
  - set_seeds(seed)              : seed random / numpy / torch
  - file_sha256(path)            : content hash of an input artifact
  - git_info()                   : commit SHA + dirty flag of the working tree
  - write_run_manifest(...)      : manifest.json in the run's output folder
  - append_registry(...)         : one row per run in results/registry.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Avoid importing TensorFlow through `transformers` (slow, noisy, unused).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths

REGISTRY_COLUMNS = [
    "run_id", "timestamp_utc", "phase", "script", "argv", "out_dir",
    "manifest", "git_sha", "git_dirty", "headline_metric", "headline_value", "notes",
]


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


def split_path(seed: int = 42, regime: str = "random") -> Path:
    paths = ProjectPaths()
    suffix = f"_{regime}" if regime != "random" else ""
    return paths.splits_dir / f"split_seed{seed}{suffix}.json"


def get_arg_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--regime", type=str, default="random", choices=["random", "disjoint"])
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover
        pass


def file_sha256(path: Path, chunk: int = 1 << 20) -> Optional[str]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info() -> dict:
    root = ProjectPaths().root
    out = {"sha": None, "dirty": None, "branch": None}
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, timeout=10)
        out["sha"] = sha.decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, timeout=10)
        out["dirty"] = bool(status.decode().strip())
        br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, timeout=10)
        out["branch"] = br.decode().strip()
    except Exception:
        pass
    return out


_PKG_SNAPSHOT: Optional[dict] = None


def package_versions() -> dict:
    """Versions of the packages that influence results (cheap; no pip freeze subprocess)."""
    global _PKG_SNAPSHOT
    if _PKG_SNAPSHOT is not None:
        return _PKG_SNAPSHOT
    from importlib import metadata as md
    names = [
        "numpy", "pandas", "scipy", "scikit-learn", "sentence-transformers", "transformers",
        "torch", "faiss-cpu", "openai", "rouge-score", "bert-score", "rank-bm25", "xgboost",
    ]
    vers = {}
    for n in names:
        try:
            vers[n] = md.version(n)
        except Exception:
            vers[n] = None
    _PKG_SNAPSHOT = vers
    return vers


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if hasattr(x, "to_dict") and callable(x.to_dict):
        return x.to_dict()
    if hasattr(x, "item") and callable(x.item):  # numpy scalar
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def make_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def write_run_manifest(
    out_dir: Path,
    *,
    script: str,
    args: Any = None,
    config: Any = None,
    inputs: Iterable[Path] = (),
    extra: Optional[dict] = None,
    run_id: Optional[str] = None,
    filename: str = "manifest.json",
) -> Path:
    """Write a manifest describing exactly how a run was produced."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or make_run_id(Path(script).stem)
    manifest = {
        "run_id": run_id,
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": _jsonable(vars(args)) if args is not None and hasattr(args, "__dict__") else _jsonable(args),
        "config": _jsonable(config),
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": package_versions(),
        "inputs": {str(p): file_sha256(Path(p)) for p in inputs},
        "extra": _jsonable(extra or {}),
    }
    path = out_dir / filename
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_registry(
    *,
    run_id: str,
    phase: str,
    script: str,
    out_dir: Path,
    manifest: Optional[Path] = None,
    headline_metric: str = "",
    headline_value: Any = "",
    notes: str = "",
) -> Path:
    """Append one row to results/registry.csv (created if missing)."""
    paths = ProjectPaths()
    reg = paths.results / "registry.csv"
    reg.parent.mkdir(parents=True, exist_ok=True)
    g = git_info()
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "script": script,
        "argv": " ".join(sys.argv),
        "out_dir": str(Path(out_dir).relative_to(paths.root)) if str(out_dir).startswith(str(paths.root)) else str(out_dir),
        "manifest": str(manifest) if manifest else "",
        "git_sha": g.get("sha") or "",
        "git_dirty": g.get("dirty"),
        "headline_metric": headline_metric,
        "headline_value": headline_value,
        "notes": notes,
    }
    new = not reg.exists()
    with reg.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return reg


def short_model_name(model: str) -> str:
    """'openai/gpt-5.6-luna' -> 'gpt56luna' (for folder names)."""
    tail = model.split("/")[-1]
    return "".join(ch for ch in tail if ch.isalnum()).lower()


def cache_dir() -> Path:
    p = ProjectPaths().root / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p
