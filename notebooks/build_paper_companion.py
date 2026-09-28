"""
Build the IEEE Access paper companion notebook (08_paper_companion.ipynb).

The companion is the paper's complete evidence base: every figure, table and number in the
manuscript appears here in the same order, with the code that produced it and the reasoning that
explains it. Sections are defined in companion_*.py modules and concatenated in paper order.

    python notebooks/build_paper_companion.py
    python notebooks/validate_ieee_notebooks.py --execute
"""
from __future__ import annotations

import json
from pathlib import Path

import companion_analysis
import companion_appendix
import companion_foundations
import companion_foundations2
import companion_front
import companion_mechanism
import companion_modeling
import companion_synthesis
import companion_validation

HERE = Path(__file__).resolve().parent

SETUP = {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": """\
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

import paper_lib as L

sys.path.insert(0, str(L.ROOT))  # so notebook cells can import from src/
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_colwidth", 160)
pd.set_option("display.width", 240)
np.random.seed(42)
print("Project root:", L.ROOT)
""".splitlines(keepends=True),
}


def build() -> dict:
    cells = [SETUP]
    for module in (
        companion_front,
        companion_foundations,
        companion_foundations2,
        companion_analysis,
        companion_mechanism,
        companion_modeling,
        companion_validation,
        companion_synthesis,
        companion_appendix,
    ):
        cells.extend(module.cells())
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    notebook = build()
    out = HERE / "08_paper_companion.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    n_md = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"wrote {out}  ({n_md} markdown cells, {n_code} code cells)")
