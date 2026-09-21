"""Compile or execute the generated IEEE notebooks without requiring Jupyter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
NOTEBOOKS = sorted(HERE.glob("0[2-6]_*.ipynb"))


def validate(path: Path, execute: bool) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compiled = compile(source, f"{path.name}:cell-{index}", "exec")
        if execute:
            exec(compiled, namespace)
    print(f"{'executed' if execute else 'compiled'} {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run cells after compiling them")
    args = parser.parse_args()
    if not NOTEBOOKS:
        raise SystemExit("No IEEE notebooks found; run build_ieee_notebooks.py first")
    os.chdir(HERE)
    for notebook in NOTEBOOKS:
        validate(notebook, args.execute)
