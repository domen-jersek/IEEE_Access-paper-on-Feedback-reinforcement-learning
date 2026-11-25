#!/usr/bin/env python3
"""36_revision_figures.py — main figures for the IEEE Access revision.

Reads only committed aggregate tables and generates vector PDFs. No model or API
call. The figure numbers follow the planned structure in FIGURE_PLAN.md.

Outputs (revision_ieee_access/03_manuscript/figures/):
    fig2_generated_effects.pdf
    fig3_expert_evaluation.pdf
    fig4_lift_saturation.pdf
    fig5_matched_retrievers.pdf
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "revision_ieee_access" / "01_data_protocol" / "tables"
HUMAN = ROOT / "revision_ieee_access" / "02_human_study" / "results" / "alberto"
OUT = ROOT / "revision_ieee_access" / "03_manuscript" / "figures"

PANEL_LABELS = {
    "legacy_random": "Familiar\n(random split)",
    "disjoint_duplicate": "Duplicate\nremoved",
    "disjoint_prefix": "Strict\n(prefix-disjoint)",
}
SOURCE_LABELS = {
    "conditioned": "Conditioned judge",
    "blind": "Blind judge",
    "reply_embed_bge": "Reply similarity",
}
COLORS = {
    "conditioned": "#0072B2",
    "blind": "#D55E00",
    "reply_embed_bge": "#009E73",
}
MARKERS = {"conditioned": "o", "blind": "s", "reply_embed_bge": "^"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    if os.environ.get("REVISION_FIGURE_PNG"):
        fig.savefig(OUT / Path(name).with_suffix(".png"), dpi=150,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def fig2_generated() -> None:
    df = pd.read_csv(TABLES / "generated_deltas.csv")
    df = df[df["metric"] == "delta_cosine"]
    contrasts = pd.read_csv(TABLES / "generated_protocol_contrast.csv")

    panels = ["legacy_random", "disjoint_duplicate", "disjoint_prefix"]
    sources = ["conditioned", "blind", "reply_embed_bge"]
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.75, 1.0]}
    )

    y = np.arange(len(panels))[::-1]
    offsets = {"conditioned": 0.20, "blind": 0.0, "reply_embed_bge": -0.20}
    for source in sources:
        rows = df[df["source"] == source].set_index("panel")
        xs = [rows.loc[p, "mean"] for p in panels]
        lo = [rows.loc[p, "mean"] - rows.loc[p, "ci_lower"] for p in panels]
        hi = [rows.loc[p, "ci_upper"] - rows.loc[p, "mean"] for p in panels]
        ax.errorbar(
            xs, y + offsets[source], xerr=[lo, hi], fmt=MARKERS[source],
            color=COLORS[source], ecolor=COLORS[source], elinewidth=0.8,
            capsize=2.0, markersize=4.0, label=SOURCE_LABELS[source],
        )
    ax.axvline(0, color="0.35", lw=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([PANEL_LABELS[p] for p in panels])
    ax.set_xlabel(r"$\Delta$ cosine vs. no-feedback baseline")
    ax.set_title("(a) Generated-answer effect by data regime", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=1, handletextpad=0.4)
    ax.set_xlim(-0.075, 0.045)

    rows = contrasts.set_index("panel")
    xs = [rows.loc[p, "mean"] for p in panels]
    lo = [rows.loc[p, "mean"] - rows.loc[p, "ci_lower"] for p in panels]
    hi = [rows.loc[p, "ci_upper"] - rows.loc[p, "mean"] for p in panels]
    ax2.errorbar(
        xs, y, xerr=[lo, hi], fmt="D", color="#333333", ecolor="#333333",
        elinewidth=0.8, capsize=2.0, markersize=4.0,
    )
    ax2.axvline(0, color="0.35", lw=0.7, zorder=0)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Conditioned $-$ blind")
    ax2.set_title("(b) Direct protocol contrast", loc="left")
    ax2.set_xlim(-0.005, 0.085)

    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig2_generated_effects.pdf")


def fig4_saturation() -> None:
    scores = pd.read_csv(TABLES / "score_distribution.csv").set_index("protocol")
    votes = pd.read_csv(TABLES / "vote_saturation.csv")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    protocols = ["conditioned", "blind"]
    labels = ["Conditioned", "Blind"]
    neg = [scores.loc[p, "pct_negative"] * 100 for p in protocols]
    neu = [scores.loc[p, "pct_neutral"] * 100 for p in protocols]
    pos = [scores.loc[p, "pct_positive"] * 100 for p in protocols]
    x = np.arange(len(protocols))
    ax.bar(x, neg, color="#D55E00", label="negative", width=0.55)
    ax.bar(x, neu, bottom=neg, color="#CCCCCC", label="neutral", width=0.55)
    ax.bar(x, pos, bottom=np.array(neg) + np.array(neu), color="#0072B2",
           label="positive", width=0.55)
    for i in range(len(protocols)):
        ax.text(i, neg[i] / 2, f"{neg[i]:.1f}", ha="center", va="center",
                color="white", fontsize=7)
        ax.text(i, neg[i] + neu[i] / 2, f"{neu[i]:.1f}", ha="center", va="center",
                color="black", fontsize=7)
        ax.text(i, neg[i] + neu[i] + pos[i] / 2, f"{pos[i]:.1f}", ha="center",
                va="center", color="white", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of allowed pairs (%)")
    ax.set_title("(a) Judge-score distribution", loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.16))

    for protocol, color, marker in [("conditioned", "#0072B2", "o"),
                                    ("blind", "#D55E00", "s")]:
        group = votes[votes["protocol"] == protocol].sort_values("n_votes")
        ax2.plot(group["n_votes"], group["pct_at_negative_cap"] * 100,
                 marker=marker, color=color)
        last = group.iloc[-1]
        ax2.text(last["n_votes"] * 1.12, last["pct_at_negative_cap"] * 100,
                 protocol, color=color, va="center", fontsize=6.5)
    ax2.set_xscale("log")
    ax2.set_xticks([1, 2, 5, 10])
    ax2.set_xticklabels(["1", "2", "5", "10"])
    ax2.set_xlim(0.9, 22)
    ax2.set_ylim(-0.6, 17.5)
    ax2.set_xlabel("Number of votes for a candidate")
    ax2.set_ylabel("Candidates at negative cap (%)")
    ax2.set_title("(b) Negative-cap saturation", loc="left")
    ax2.text(3.0, 2.2,
             "at 2 votes: 13.2% conditioned / 9.1% blind\n"
             "positive cap: 0.15% / 0%",
             fontsize=6.3, color="0.25")

    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig4_lift_saturation.pdf")


def fig5_retrievers() -> None:
    df = pd.read_csv(TABLES / "matched_retrievers.csv")
    df = df[(df["source"] == "conditioned") & (df["scale_mode"] == "pool_std")]
    order = ["dense_minilm", "bm25", "hybrid_rrf"]
    labels = {"dense_minilm": "Dense MiniLM", "bm25": "BM25", "hybrid_rrf": "Hybrid RRF"}
    panels = [("legacy_random", "Familiar (random split)"),
              ("disjoint_prefix", "Strict (prefix-disjoint)")]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (panel, title) in zip(axes, panels):
        group = df[df["panel"] == panel].set_index("retriever")
        xs = [group.loc[r, "delta_top1"] for r in order]
        lo = [group.loc[r, "delta_top1"] - group.loc[r, "ci_lower"] for r in order]
        hi = [group.loc[r, "ci_upper"] - group.loc[r, "delta_top1"] for r in order]
        ax.errorbar(xs, y, xerr=[lo, hi], fmt="o", color="#0072B2",
                    ecolor="#0072B2", elinewidth=0.8, capsize=2.0, markersize=4.0)
        ax.axvline(0, color="0.35", lw=0.7, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[r] for r in order])
        ax.set_xlabel(r"$\Delta$ top-1 proxy")
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.005, 0.09)
    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig5_matched_retrievers.pdf")


def fig3_human() -> None:
    summary = pd.read_csv(HUMAN / "rater_alberto_summary.csv")
    summary = summary[summary["metric"] == "strict"].set_index(["panel", "source"])
    length = pd.read_csv(HUMAN / "rater_alberto_length.csv").set_index(["panel", "source"])
    item = pd.read_csv(HUMAN / "rater_alberto_item_level.csv")
    alignment = pd.read_csv(HUMAN / "rater_alberto_alignment.csv").set_index(["panel", "source"])

    arms = [("legacy_random", "blind", "Familiar\nblind"),
            ("disjoint_prefix", "blind", "Strict\nblind"),
            ("disjoint_prefix", "conditioned", "Strict\ncond.")]
    fig, (ax, ax2, ax3) = plt.subplots(
        1, 3, figsize=(7.3, 2.6), gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]}
    )

    y = np.arange(len(arms))[::-1]
    tick_labels = []
    for i, (panel, source, label) in enumerate(arms):
        row = summary.loc[(panel, source)]
        ax.plot([row["ci_lower"], row["ci_upper"]], [y[i], y[i]],
                color="#555555", lw=1.0)
        ax.plot(row["delta"], y[i], "o", color="#0072B2", markersize=4.5)
        tick_labels.append(
            f"{label}\n{row['baseline_rate']:.2f} → {row['feedback_rate']:.2f}")
    ax.axvline(0, color="0.35", lw=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim(-0.16, 0.32)
    ax.set_xlabel("Expert $\\Delta$ strict correctness")
    ax.set_title("(a) Correctness difference", loc="left")

    width = 0.34
    x = np.arange(len(arms))
    matched = [length.loc[(p, s), "delta_matched"] for p, s, _ in arms]
    unmatched = [length.loc[(p, s), "delta_unmatched"] for p, s, _ in arms]
    ax2.bar(x - width / 2, unmatched, width, color="#D55E00",
            label="length differs >100 chars")
    ax2.bar(x + width / 2, matched, width, color="#009E73",
            label="length-matched")
    ax2.axhline(0, color="0.35", lw=0.7, zorder=0)
    ax2.set_xticks(x)
    ticks = []
    for p, s, _ in arms:
        ticks.append(f"{'Familiar' if p == 'legacy_random' else 'Strict'}\n"
                     f"{'blind' if s == 'blind' else 'cond.'}\n"
                     f"({length.loc[(p, s), 'n_unmatched']}/"
                     f"{length.loc[(p, s), 'n_matched']})")
    ax2.set_xticklabels(ticks)
    ax2.set_ylim(-0.27, 0.30)
    ax2.set_ylabel("Expert $\\Delta$ correctness")
    ax2.set_title("(b) Length-matched effect", loc="left")
    ax2.legend(frameon=False, fontsize=6.2, loc="upper right")

    random = item[item["arm"] == "legacy_random/blind"].copy()
    random["jitter"] = np.linspace(-0.06, 0.06, len(random))
    ax3.scatter(random["auto_delta"], random["human_delta"] + random["jitter"],
                s=12, color="#0072B2", alpha=0.65, edgecolor="white", linewidth=0.3)
    ax3.axhline(0, color="0.35", lw=0.7, zorder=0)
    ax3.axvline(0, color="0.35", lw=0.7, zorder=0)
    ax3.set_xlabel("Automatic $\\Delta$ cosine")
    ax3.set_ylabel("Expert $\\Delta$ (jittered)")
    ax3.set_title("(c) Familiar arm: metric divergence", loc="left")
    ax3.set_yticks([-1, 0, 1])
    row = alignment.loc[("legacy_random", "blind")]
    ax3.text(0.97, 0.05, f"Spearman {row['spearman']:+.2f}\nAUC {row['auc_preference']:.2f}",
             transform=ax3.transAxes, va="bottom", ha="right", fontsize=6.5)

    fig.tight_layout(w_pad=0.9)
    _save(fig, "fig3_expert_evaluation.pdf")


def main() -> int:
    _style()
    fig2_generated()
    fig3_human()
    fig4_saturation()
    fig5_retrievers()
    return 0


if __name__ == "__main__":
    sys.exit(main())
