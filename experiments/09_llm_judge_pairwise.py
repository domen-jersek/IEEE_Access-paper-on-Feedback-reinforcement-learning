#!/usr/bin/env python3
"""
09_llm_judge_pairwise.py — P1.3
================================
Pairwise LLM-as-judge on the STORED answers of completed runs: for a fixed,
stratified subsample of dev tickets, a judge model (which must differ from the
generator) sees the reference reply and the two generated first replies
(baseline vs feedback) in BOTH orders and picks the one that better matches
the reference procedure. A win is counted only when both orders agree.

    python experiments/09_llm_judge_pairwise.py --judge-model <openrouter-model-id> \
        --runs M2_team_dev_conditioned_continuous M4_intersection_dev_conditioned_continuous

Outputs: results/answer_judge/{subsample_ids.json, scores.csv, summary.json, manifest.json}
Judge responses are cached in data/cache/judge_pairwise_cache.db (re-runs are free).
"""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from experiments.utils import (setup_logging, get_arg_parser, write_run_manifest, append_registry, make_run_id, cache_dir)
from src.config import ProjectPaths
from src.generation.client import LLMClient

PAIRWISE_TEMPLATE = """\
You are an expert IT support quality reviewer.

TICKET
Title: {title}
Description: {description}

REFERENCE FIRST REPLY (ground truth written by the support team)
{reference}

CANDIDATE REPLY A
{reply_a}

CANDIDATE REPLY B
{reply_b}

Which candidate reply matches the REFERENCE more closely in procedural content
(same resolution steps, same form / redirect / action requested)? Ignore tone and length.
Answer with exactly one token: A, B, or TIE."""
PROMPT_VERSION = "pairwise_v1"


def parse_choice(text: str) -> str:
    t = (text or "").strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", t)
    return m.group(1) if m else "TIE"


def stratified_subsample(recs: list[dict], n: int, seed: int) -> list[str]:
    df = pd.DataFrame({"ticket_id": [r["ticket_id"] for r in recs], "cls": [r["expected_class"] for r in recs]})
    rng = np.random.default_rng(seed)
    df = df.iloc[rng.permutation(len(df))]
    # proportional allocation with at least 1 per class where possible
    counts = (df["cls"].value_counts(normalize=True) * n).round().astype(int)
    picked = []
    for cls, k in counts.items():
        picked += df[df["cls"] == cls]["ticket_id"].head(max(int(k), 1)).tolist()
    picked = picked[:n]
    if len(picked) < n:
        rest = [t for t in df["ticket_id"] if t not in set(picked)]
        picked += rest[: n - len(picked)]
    return sorted(picked)


async def judge_run(details: Path, ids: set[str], client: LLMClient, log) -> pd.DataFrame:
    recs = {r["ticket_id"]: r for r in json.loads(details.read_text("utf-8"))}
    rows = []
    tasks = []
    for tid in sorted(ids):
        r = recs.get(tid)
        if r is None:
            continue
        a, b = r["baseline"]["answer"], r["feedback"]["answer"]
        if a.strip() == b.strip():
            rows.append({"run": details.parent.name, "ticket_id": tid, "identical": True, "verdict": 0, "order1": "TIE", "order2": "TIE"})
            continue
        base = dict(title=r["query_title"], description=r.get("query_description", ""), reference=r["reference_reply"])
        p1 = PAIRWISE_TEMPLATE.format(reply_a=a, reply_b=b, **base)   # A=baseline, B=feedback
        p2 = PAIRWISE_TEMPLATE.format(reply_a=b, reply_b=a, **base)   # A=feedback, B=baseline
        tasks.append((tid, r, asyncio.gather(client.complete(p1), client.complete(p2))))
    for tid, r, fut in tasks:
        o1, o2 = await fut
        c1, c2 = parse_choice(o1), parse_choice(o2)
        fb_wins = (c1 == "B") and (c2 == "A")
        bl_wins = (c1 == "A") and (c2 == "B")
        verdict = 1 if fb_wins else (-1 if bl_wins else 0)
        rows.append({"run": details.parent.name, "ticket_id": tid, "identical": False, "verdict": verdict,
                     "order1": c1, "order2": c2, "position_consistent": bool(fb_wins or bl_wins or (c1 == "TIE" and c2 == "TIE")),
                     "delta_cosine": r["deltas"]["delta_cosine"], "expected_class": r["expected_class"],
                     "expected_team": r["expected_team"]})
    return pd.DataFrame(rows)


async def main_async() -> None:
    parser = get_arg_parser("Pairwise LLM judge on stored answers")
    parser.add_argument("--judge-model", required=True, help="OpenRouter model id; must differ from the generator")
    parser.add_argument("--runs", nargs="+", required=True, help="result folder names under results/")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("judge_pairwise")
    paths = ProjectPaths()
    out_dir = paths.results / "answer_judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("answer_judge")

    details_paths = []
    for name in args.runs:
        files = sorted(glob.glob(str(paths.results / name / "*_details.json")))
        if not files:
            log.error("no details for %s", name)
            continue
        details_paths.append(Path(files[-1]))
    if not details_paths:
        return
    first = json.loads(details_paths[0].read_text("utf-8"))
    gen_model = first[0].get("generator_model") or json.loads(sorted(glob.glob(str(details_paths[0].parent / "*_summary.json")))[-1]).get("config", {}).get("generator_model")
    if gen_model and gen_model == args.judge_model:
        log.error("judge model must differ from the generator (%s)", gen_model)
        return

    sub_path = out_dir / "subsample_ids.json"
    if sub_path.exists():
        ids = json.loads(sub_path.read_text("utf-8"))["ticket_ids"]
        log.info("Using existing subsample (%d ids)", len(ids))
    else:
        ids = stratified_subsample(first, args.n, args.seed)
        sub_path.write_text(json.dumps({"n": len(ids), "seed": args.seed, "source_run": details_paths[0].parent.name,
                                        "ticket_ids": ids}, indent=2), encoding="utf-8")
        log.info("Created subsample of %d ids", len(ids))

    client = LLMClient(model=args.judge_model, temperature=0.0, cache_path=cache_dir() / "judge_pairwise_cache.db",
                       max_concurrency=args.concurrency)
    frames = []
    for d in details_paths:
        log.info("Judging %s", d.parent.name)
        frames.append(await judge_run(d, set(ids), client, log))
    scores = pd.concat(frames, ignore_index=True)
    scores_path = out_dir / "scores.csv"
    if scores_path.exists():
        old = pd.read_csv(scores_path)
        old = old[~old["run"].isin(scores["run"])]
        scores = pd.concat([old, scores], ignore_index=True)
    scores.to_csv(scores_path, index=False)

    summary = {"judge_model": args.judge_model, "generator_model": gen_model, "prompt_version": PROMPT_VERSION,
               "n_subsample": len(ids), "runs": {}}
    for run, g in scores.groupby("run"):
        judged = g[~g["identical"]]
        summary["runs"][run] = {
            "n": int(len(g)), "n_identical": int(g["identical"].sum()),
            "feedback_wins": int((judged["verdict"] == 1).sum()), "baseline_wins": int((judged["verdict"] == -1).sum()),
            "ties_or_inconsistent": int((judged["verdict"] == 0).sum()),
            "net_win_rate": float(judged["verdict"].mean()) if len(judged) else None,
            "position_consistency": float(judged["position_consistent"].mean()) if len(judged) else None,
            "agreement_with_cosine_sign": float(np.mean(np.sign(judged["delta_cosine"]) == judged["verdict"])) if len(judged) else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = write_run_manifest(out_dir, script="experiments/09_llm_judge_pairwise.py", args=args, inputs=details_paths,
                                  extra={"cache_hits": client.cache_hits, "cache_misses": client.cache_misses}, run_id=run_id)
    append_registry(run_id=run_id, phase="P1.3-answer-judge", script="09_llm_judge_pairwise.py", out_dir=out_dir, manifest=manifest,
                    headline_metric="net_win_rate", headline_value=json.dumps({k: v["net_win_rate"] for k, v in summary["runs"].items()}))
    print("\n=== PAIRWISE JUDGE ===")
    for run, s in summary["runs"].items():
        print(f"{run}: fb wins {s['feedback_wins']}  bl wins {s['baseline_wins']}  tie {s['ties_or_inconsistent']}  "
              f"net={s['net_win_rate']}  pos-consistent={s['position_consistency']}  agree-with-cosine={s['agreement_with_cosine_sign']}")
    print(f"api calls: {client.cache_misses}  cached: {client.cache_hits}   outputs: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main_async())
