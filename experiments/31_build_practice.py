#!/usr/bin/env python3
"""31_build_practice.py — two private practice items for rubric calibration.

Writes ``practice.md`` into a study package directory. The items are drawn from
the training partition (never the evaluation sample), paired with a deliberately
unrelated or partially relevant reply, and include a suggested key and a short
explanation. They are for the introductory call only and are not part of the
study data.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ProjectPaths


def _pick(df: pd.DataFrame, seed: str, predicate, label: str) -> pd.Series:
    candidates = df[predicate]
    if candidates.empty:
        raise SystemExit(f"No candidate ticket for practice item: {label}")
    ordered = sorted(candidates.index,
                     key=lambda i: hashlib.sha256(f"{seed}|{df.at[i, 'seq_id']}".encode()).hexdigest())
    return df.loc[ordered[0]]


def _pick_first(df: pd.DataFrame, seed: str, predicates, label: str) -> pd.Series:
    last = None
    for index, predicate in enumerate(predicates):
        try:
            return _pick(df, seed, predicate, label)
        except SystemExit as error:
            last = error
            continue
    raise last


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--seed", default="practice-20261120")
    args = parser.parse_args()

    paths = ProjectPaths()
    dataset = pd.read_parquet(paths.dataset_parquet)
    dataset["seq_id"] = dataset["seq_id"].astype(str)
    split_path = paths.data_processed / "revision" / "splits" / "split_prefix_template_v1_seed42.json"
    train = set(json.loads(split_path.read_text(encoding="utf-8"))["train"])
    pool = dataset[dataset.seq_id.isin(train)].copy()
    pool["reply_len"] = pool["first_reply"].str.len()
    usable = pool[(pool.reply_len.between(180, 900)) & (pool.Description_anon.fillna("").str.len() > 40)]

    first = _pick_first(usable, args.seed + "|first", [
        lambda df: df["Team->Name"].eq(df["Team->Name"].value_counts().index[0]),
    ], "first")
    second = _pick_first(usable, args.seed + "|second", [
        lambda df: df["Team->Name"].ne(first["Team->Name"]),
    ], "second")
    unrelated = _pick_first(usable, args.seed + "|unrelated", [
        lambda df: df["Team->Name"].ne(first["Team->Name"])
        & df["intent_class"].ne(first["intent_class"]),
        lambda df: df["Team->Name"].ne(first["Team->Name"]),
        lambda df: df["intent_class"].ne(first["intent_class"]),
    ], "unrelated")
    partial = _pick_first(usable, args.seed + "|partial", [
        lambda df: df["Team->Name"].eq(second["Team->Name"])
        & df["intent_class"].ne(second["intent_class"]),
        lambda df: df["Team->Name"].eq(second["Team->Name"])
        & df["Service subcategory->Name"].ne(second["Service subcategory->Name"]),
        lambda df: df["intent_class"].eq(second["intent_class"])
        & df["Team->Name"].ne(second["Team->Name"]),
    ], "partial")

    def block(ticket: pd.Series) -> str:
        description = str(ticket["Description_anon"])
        return f"**Title:** {ticket['Title_anon']}\n\n**Description:** {description}"

    text = f"""# Practice items (not part of the study)

Two examples for the introductory call. Read them with the evaluator and agree on
the rubric before the real sample. The suggested key is ours; the point is to
discuss it, not to test the evaluator.

## Practice 1 — clearly wrong alternative

{block(first)}

**Reply A**

{first['first_reply']}

**Reply B**

{unrelated['first_reply']}

Suggested key: **A correct, B incorrect.** A is the recorded reply for this ticket.
B comes from a different team and a different request type, so following it would
send the user to the wrong procedure. Note that B may still look professional and
well written; the rating is about the procedure, not the style.

## Practice 2 — partially relevant alternative

{block(second)}

**Reply A**

{partial['first_reply']}

**Reply B**

{second['first_reply']}

Suggested key: **A partially correct, B correct.** A comes from the same team but a
different request type, so parts of the process may be useful while the specific
steps are wrong. This is the case where a simple A/B preference is not enough and
the per-reply correctness rating matters.
"""
    out = args.package / "practice.md"
    if out.exists():
        raise SystemExit(f"{out} already exists")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
