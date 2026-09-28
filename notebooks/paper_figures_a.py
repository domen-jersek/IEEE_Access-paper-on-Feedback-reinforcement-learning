"""Figures F1-F6 for the IEEE Access manuscript (see PAPER_OUTLINE_JOURNAL.md).

Design system: 8 pt base type at final print size, single/double column widths,
Okabe-Ito palette with fixed semantics, panel tags outside the data area, at most
one sparse annotation per panel, all prose in captions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from paper_figures_lib import (ACCENT, BLIND, COND, DOUBLE, NEUTRAL, ORACLE, OUT, SCOPE_COLORS,
                               WARN, annot, check_artists, check_overlaps, hgrid, m4_intersection_lift_values,
                               resc, resc_row, save, tag, zero)

SCOPE_ORDER = ["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"]
SCOPE_SHORT = ["Global", "Team", "Class", "Team∩Class", "Backoff"]
DEV_RUNS = {
    "M1_global": ("M1_global_dev_conditioned_continuous", "M1_global_dev_blind_continuous"),
    "M2_team": ("M2_team_dev_conditioned_continuous", "M2_team_dev_blind_continuous"),
    "M3_class": ("M3_class_dev_conditioned_continuous", "M3_class_dev_blind_continuous"),
    "M4_intersection": ("M4_intersection_dev_conditioned_continuous", "M4_intersection_dev_blind_continuous"),
}


def _ci(row, mean_col="delta_cosine_mean", lo_col="delta_cosine_ci_lo", hi_col="delta_cosine_ci_hi"):
    return (row[mean_col] - row[lo_col], row[hi_col] - row[mean_col])


def _dot(ax, x, mean, ci, color, ms=5.0, lw=1.1):
    ax.errorbar(x, mean, yerr=[[ci[0]], [ci[1]]], fmt="o", ms=ms, color=color, ecolor=color,
                elinewidth=lw, capsize=2, capthick=lw, zorder=4)


# --------------------------------------------------------------------------- F1
def fig1_judge() -> None:
    from paper_lib import load_calib_reliability, load_calib_report
    rel_c = load_calib_reliability("conditioned")
    rel_b = load_calib_reliability("blind")
    report = load_calib_report()

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7))
    ax = axes[0]
    for rel, color, label, offset in ((rel_c, COND, "conditioned", -0.024), (rel_b, BLIND, "blind", 0.024)):
        share = rel["n"] / rel["n"].sum()
        ax.bar(rel["mean_score"] + offset, share.clip(lower=1e-6), width=0.042, color=color,
               label=label)
    ax.set_yscale("log")
    ax.set_xlabel("judge score")
    ax.set_ylabel("share of judged pairs (log)")
    ax.legend(loc="upper right", ncol=1)
    annot(ax, "exact zeros:\n59 % / 48 %", x=0.30, y=0.97, ha="left", va="top", color="#333333")
    tag(ax, "(a)")

    ax = axes[1]
    ax.plot(rel_c["mean_score"], rel_c["mean_usefulness_sim"], "-o", ms=3.2, color=COND,
            label="conditioned")
    ax.plot(rel_b["mean_score"], rel_b["mean_usefulness_sim"], "-o", ms=3.2, color=BLIND,
            label="blind")
    auc_c = report["protocols"]["conditioned"]["auc_of_score"]["same_reply"]
    auc_b = report["protocols"]["blind"]["auc_of_score"]["same_reply"]
    ax.set_xlabel("judge score")
    ax.set_ylabel("candidate–reference reply similarity")
    ax.legend(loc="lower right", markerscale=0.7)
    annot(ax, f"AUC (same reply)\n{auc_c:.2f} vs {auc_b:.2f}", x=0.03, y=0.97, ha="left", va="top",
          color="#333333")
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.6)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F1_judge")


# --------------------------------------------------------------------------- F2
def fig2_reversal() -> None:
    from paper_lib import load_details
    fig = plt.figure(figsize=(DOUBLE, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.38, wspace=0.62)

    # (a) dev dot plot
    ax = fig.add_subplot(gs[0, 0])
    offsets = {"conditioned": -0.16, "blind": 0.16}
    for protocol, color in (("conditioned", COND), ("blind", BLIND)):
        for i, scope in enumerate(SCOPE_ORDER[:4]):
            row = resc_row(DEV_RUNS[scope][0 if protocol == "conditioned" else 1])
            _dot(ax, i + offsets[protocol], row["delta_cosine_mean"], _ci(row), color)
    ax.set_xticks(range(4), SCOPE_SHORT[:4])
    ax.set_ylabel("mean Δcosine (dev, 319)")
    ax.set_ylim(-0.085, 0.050)
    zero(ax)
    hgrid(ax)
    ax.plot([], [], "o", color=COND, label="conditioned")
    ax.plot([], [], "o", color=BLIND, label="blind")
    ax.legend(loc="upper left", ncol=2)
    tag(ax, "(a)")

    # (b) eval forest
    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("Team · cond.", "M2_team_eval_conditioned_continuous", COND),
        ("T∩C · cond.", "M4_intersection_eval_conditioned_continuous", COND),
        ("Team · blind", "M2_team_eval_blind_continuous", BLIND),
        ("T∩C · blind", "M4_intersection_eval_blind_continuous", BLIND),
    ]
    y = np.arange(len(rows))[::-1]
    for yi, (label, folder, color) in zip(y, rows):
        row = resc_row(folder)
        _dot(ax, row["delta_cosine_mean"], yi, _ci(row), color)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-0.075, 0.045)
    ax.set_xlabel("mean Δcosine (eval, 398)")
    zero(ax, ls="--")
    hgrid(ax)
    tag(ax, "(b)")

    # (c) per-ticket distributions
    ax = fig.add_subplot(gs[1, :])
    bins = np.linspace(-0.85, 0.85, 35)
    for folder, label, color in (
        ("M4_intersection_eval_conditioned_continuous", "conditioned", COND),
        ("M4_intersection_eval_blind_continuous", "blind", BLIND),
    ):
        deltas = np.array([r["deltas"]["delta_cosine"] for r in load_details(folder)])
        ax.hist(deltas, bins=bins, histtype="step", lw=1.5, color=color, density=True,
                label=f"{label}: mean {deltas.mean():+.3f}; helped {np.mean(deltas > 0):.0%}, "
                      f"hurt {np.mean(deltas < 0):.0%}")
        ax.axvline(deltas.mean(), color=color, lw=1.0, ls="--")
    zero(ax)
    ax.set_ylim(0, 9.5)
    ax.set_xlabel("per-ticket Δcosine (Team∩Class, eval)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    annot(ax, "sd 0.117 vs 0.175", x=0.99, y=0.88, ha="right", va="top", color="#333333")
    tag(ax, "(c)")
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F2_reversal")


# --------------------------------------------------------------------------- F3
def fig3_scope() -> None:
    from paper_lib import load_ladder_grid
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))

    ax = axes[0]
    for protocol, grid_tag, color, offset in (("conditioned", "", COND, -0.16),
                                              ("blind", "blind", BLIND, 0.16)):
        grid = load_ladder_grid(grid_tag)
        grid = grid[grid["routing"] != "none"]
        best = grid.groupby("routing")["minilm_d_proxy_top1"].max()
        values = [float(best.get(scope, np.nan)) for scope in SCOPE_ORDER]
        ax.plot(np.arange(len(SCOPE_ORDER)) + offset, values, "o", ms=5.0, color=color,
                label=protocol, zorder=4)
    ax.set_xticks(range(len(SCOPE_ORDER)), SCOPE_SHORT)
    ax.set_ylabel("best proxy gain (top-1)")
    ax.set_ylim(-0.005, 0.115)
    zero(ax)
    hgrid(ax)
    ax.legend(loc="upper left", ncol=2)
    tag(ax, "(a)")

    ax = axes[1]
    generated = {
        "M1_global": ("M1_global_dev_conditioned_continuous", "M1_global_dev_blind_continuous"),
        "M2_team": ("M2_team_dev_conditioned_continuous", "M2_team_dev_blind_continuous"),
        "M3_class": ("M3_class_dev_conditioned_continuous", "M3_class_dev_blind_continuous"),
        "M4_intersection": ("M4_intersection_dev_conditioned_continuous", "M4_intersection_dev_blind_continuous"),
        "M5_backoff": ("M5_backoff_dev_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2",
                       "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    }
    for protocol, color, offset in (("conditioned", COND, -0.16), ("blind", BLIND, 0.16)):
        for i, scope in enumerate(SCOPE_ORDER):
            row = resc_row(generated[scope][0 if protocol == "conditioned" else 1])
            _dot(ax, i + offset, row["delta_cosine_mean"], _ci(row), color)
    ax.set_xticks(range(len(SCOPE_ORDER)), SCOPE_SHORT)
    ax.set_ylabel("mean Δcosine (dev, generated)")
    ax.set_ylim(-0.09, 0.05)
    zero(ax)
    hgrid(ax)
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.6)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F3_scope")


# --------------------------------------------------------------------------- F4
def fig4_retriever() -> None:
    from paper_lib import load_ladder_curve
    curve = load_ladder_curve()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0))

    ax = axes[0]
    ax.errorbar(curve["base_minilm_proxy_top1"], curve["best_gain"],
                yerr=[curve["best_gain"] - curve["primary_ci_lo"],
                      curve["primary_ci_hi"] - curve["best_gain"]],
                fmt="o", ms=4.5, color=ACCENT, ecolor=ACCENT, capsize=2, elinewidth=1.0)
    short = {"bm25": "BM25", "dense_bge": "BGE", "hybrid_bge_rrf": "H-BGE",
             "ce_rerank": "CE", "hybrid_rrf": "H-RRF", "dense_minilm": "MiniLM"}
    label_pos = {
        "bm25": (0.7300, 0.1130, "left"),
        "dense_bge": (0.7850, 0.0880, "left"),
        "hybrid_bge_rrf": (0.7620, 0.0700, "right"),
        "ce_rerank": (0.7660, 0.0420, "right"),
        "hybrid_rrf": (0.7820, 0.0280, "right"),
        "dense_minilm": (0.8060, 0.0600, "left"),
    }
    for _, r in curve.iterrows():
        tx, ty, ha = label_pos[r["retriever"]]
        ax.annotate(short.get(r["retriever"], r["retriever"]),
                    xy=(r["base_minilm_proxy_top1"], r["best_gain"]),
                    xytext=(tx, ty), textcoords="data", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, shrinkA=1, shrinkB=2.5))
    ax.set_xlim(0.714, 0.834)
    ax.set_ylim(-0.006, 0.134)
    ax.set_xlabel("base retrieval quality (top-1 proxy)")
    ax.set_ylabel("best proxy gain")
    zero(ax)
    hgrid(ax)
    tag(ax, "(a)")

    ax = axes[1]
    runs = [
        ("dense\nMiniLM", "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std1"),
        ("hybrid\nRRF", "M4_intersection_dev_blind_continuous_liftlaplace_eb_pool_std0.5_hybrid_rrf"),
        ("cross-\nencoder", "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std2_ce_rerank"),
    ]
    x = np.arange(len(runs))
    for offset, metric, alpha, label in ((-0.14, "delta_cosine", 1.0, "cosine"),
                                          (0.14, "delta_cosine_bge", 0.55, "BGE")):
        for i, (_, folder) in enumerate(runs):
            row = resc_row(folder)
            color = ACCENT if offset < 0 else "#9db3d5"
            _dot(ax, i + offset, row[f"{metric}_mean"],
                 _ci(row, f"{metric}_mean", f"{metric}_ci_lo", f"{metric}_ci_hi"), color,
                 ms=4.6, lw=0.9)
    ax.set_xticks(x, [r[0] for r in runs])
    ax.set_ylabel("mean Δcosine (blind, dev)")
    ax.set_ylim(-0.012, 0.0175)
    zero(ax)
    hgrid(ax)
    ax.plot([], [], "o", color=ACCENT, label="cosine")
    ax.plot([], [], "o", color="#9db3d5", label="BGE")
    ax.legend(loc="lower left", ncol=2, markerscale=0.7)
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.6)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F4_retriever")


# --------------------------------------------------------------------------- F5
def fig5_mechanism() -> None:
    from paper_lib import load_calib_saturation
    from src.config import LiftConfig
    from src.feedback.lift import compute_lift
    sat = load_calib_saturation()
    fig = plt.figure(figsize=(DOUBLE, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    neg = np.linspace(0, 10, 400)
    laplace = [compute_lift(0.0, n, LiftConfig.laplace()) for n in neg]
    eb = [compute_lift(0.0, n, LiftConfig.laplace_eb(2.0), 0.228) for n in neg]
    ax.plot(neg, laplace, color=BLIND, lw=1.7, label="Laplace")
    ax.plot(neg, eb, color=COND, lw=1.7, label="empirical Bayes")
    ax.axhline(-0.20, color=NEUTRAL, lw=0.8, ls=":")
    ax.scatter([2.0], [-0.20], color=BLIND, s=26, zorder=4)
    ax.set_xlabel("negative votes (zero positive)")
    ax.set_ylabel("lift")
    ax.set_ylim(-0.23, 0.03)
    ax.legend(loc="upper right", markerscale=0.7)
    tag(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    lap, ebl = m4_intersection_lift_values("conditioned")
    bins = np.linspace(-0.21, 0.21, 43)
    ax.hist(lap, bins=bins, histtype="step", lw=1.5, color=BLIND,
            label=f"Laplace ({np.mean(lap <= -0.199):.0%} at cap)")
    ax.hist(ebl, bins=bins, histtype="step", lw=1.5, color=COND,
            label=f"empirical Bayes ({np.mean(ebl <= -0.199):.0%} at cap)")
    ax.set_yscale("log")
    ax.set_xlabel("lift per candidate (Team∩Class, conditioned)")
    ax.set_ylabel("candidates (log)")
    ax.legend(loc="upper right")
    tag(ax, "(b)")

    ax = fig.add_subplot(gs[1, :])
    pivot = sat.pivot_table(index="routing", columns="lift", values="pct_at_minus_cap").reindex(SCOPE_ORDER)
    x = np.arange(len(SCOPE_ORDER))
    for offset, col, color, label in ((-0.20, "laplace", BLIND, "Laplace"),
                                      (0.0, "laplace_eb_k2", COND, "EB κ = 2"),
                                      (0.20, "laplace_eb_k10", ACCENT, "EB κ = 10")):
        ax.bar(x + offset, pivot[col] * 100, 0.19, color=color, label=label)
    ax.set_xticks(x, SCOPE_SHORT)
    ax.set_ylabel("candidates at the negative cap (%)")
    ax.set_ylim(0, 58)
    hgrid(ax)
    ax.legend(loc="upper center", ncol=3)
    tag(ax, "(c)")
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F5_mechanism")
