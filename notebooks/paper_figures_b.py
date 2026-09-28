"""Figures F7-F12 for the IEEE Access manuscript (see PAPER_OUTLINE_JOURNAL.md).

Design system: 8 pt base type at final print size, single/double column widths,
Okabe-Ito palette with fixed semantics (blue = conditioned, vermillion = blind,
amber = variant, green = calibration-repaired, pink = oracle), panel tags outside
the data area, at most one sparse annotation per panel, all prose in captions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from paper_figures_lib import (ACCENT, BLIND, COND, DOUBLE, GREEN, NEUTRAL, ORACLE, OUT, SINGLE,
                               WARN, annot, check_artists, check_overlaps, hgrid, resc, resc_row, save, tag, zero)

SCOPE_ORDER = ["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"]
SCOPE_SHORT = ["Global", "Team", "Class", "Team∩Class", "Backoff"]


def _ci(row, mean_col="delta_cosine_mean", lo_col="delta_cosine_ci_lo", hi_col="delta_cosine_ci_hi"):
    return (row[mean_col] - row[lo_col], row[hi_col] - row[mean_col])


def _dot(ax, x, mean, ci, color, ms=5.0, lw=1.1):
    ax.errorbar(x, mean, yerr=[[ci[0]], [ci[1]]], fmt="o", ms=ms, color=color, ecolor=color,
                elinewidth=lw, capsize=2, capthick=lw, zorder=4)


# --------------------------------------------------------------------------- F6
def fig6_rescue() -> None:
    from paper_lib import load_details, load_ladder_grid
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0))

    ax = axes[0]
    grid = load_ladder_grid("blind")
    grid = grid[grid["routing"] != "none"]
    variants = [
        ("Laplace, absolute", grid["lift"].str.startswith("laplace|abs"), BLIND, -0.20),
        ("Laplace, pool-scaled", grid["lift"].str.startswith("laplace|std"), WARN, 0.0),
        ("EB, pool-scaled", grid["lift"].str.startswith("laplace_eb") & grid["lift"].str.contains("std"),
         GREEN, 0.20),
    ]
    for label, mask, color, offset in variants:
        part = grid[mask]
        for i, scope in enumerate(SCOPE_ORDER):
            value = float(part[part["routing"] == scope]["minilm_d_proxy_top1"].max())
            ax.plot([i + offset], [value], "o", ms=5.0, color=color, label=label if i == 0 else None,
                    zorder=4)
    ax.set_xticks(range(len(SCOPE_ORDER)), SCOPE_SHORT)
    ax.set_ylabel("best proxy gain (blind, dev)")
    ax.set_ylim(-0.085, 0.028)
    zero(ax)
    hgrid(ax)
    ax.legend(loc="lower right")
    tag(ax, "(a)")

    ax = axes[1]
    rows = [
        ("dev", "M4_intersection_dev_blind_continuous",
         "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
        ("eval", "M4_intersection_eval_blind_continuous",
         "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    ]
    for i, (split, legacy, calibrated) in enumerate(rows):
        row_l, row_c = resc_row(legacy), resc_row(calibrated)
        m_l, ci_l = row_l["delta_cosine_mean"], _ci(row_l)
        m_c, ci_c = row_c["delta_cosine_mean"], _ci(row_c)
        ax.plot([m_l, m_c], [i, i], color="#c9c9c9", lw=1.6, zorder=1)
        ax.errorbar(m_l, i, xerr=[[ci_l[0]], [ci_l[1]]], fmt="o", ms=5, color=BLIND, ecolor=BLIND,
                    elinewidth=1.0, capsize=2, zorder=3)
        ax.errorbar(m_c, i, xerr=[[ci_c[0]], [ci_c[1]]], fmt="o", ms=5, color=GREEN, ecolor=GREEN,
                    elinewidth=1.0, capsize=2, zorder=3)
    ax.plot([], [], "o", color=BLIND, label="legacy Laplace, Team∩Class")
    ax.plot([], [], "o", color=GREEN, label="calibrated EB + backoff")
    mde = 2.802 * np.std([r["deltas"]["delta_cosine"] for r in
                          load_details("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2")],
                         ddof=1) / np.sqrt(398)
    ax.axvline(mde, color=WARN, ls="--", lw=1.0)
    ax.annotate(f"MDE {mde:.3f}", (mde, 1.42), xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=WARN, va="center")
    ax.set_yticks([0, 1], ["dev", "eval"])
    ax.set_xlim(-0.065, 0.032)
    ax.set_ylim(-0.5, 1.65)
    ax.set_xlabel("mean Δcosine (blind)")
    zero(ax)
    hgrid(ax)
    ax.legend(loc="lower left")
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.6)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F6_rescue")


# --------------------------------------------------------------------------- F7
def fig7_negatives() -> None:
    from paper_lib import load_blend_bootstrap, load_gate_study_multi, load_ladder_grid
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 3.0))

    ax = axes[0]
    sem = load_ladder_grid("semantic")
    sem = sem[sem["routing"] != "none"].copy()
    base_best = sem[~sem["routing"].str.startswith("semantic")].groupby("retriever")["minilm_d_proxy_top1"].max()
    sem_best = sem[sem["routing"].str.startswith("semantic")].groupby("retriever")["minilm_d_proxy_top1"].max()
    order = base_best.sort_values(ascending=False).index
    short = {"dense_minilm": "dense", "dense_bge": "BGE", "hybrid_rrf": "h-RRF", "bm25": "BM25",
             "hybrid_bge_rrf": "h-BGE", "ce_rerank": "CE"}
    x = np.arange(len(order))
    for i, retriever in enumerate(order):
        ax.plot([i - 0.13, i + 0.13], [base_best[retriever], sem_best[retriever]],
                color="#c9c9c9", lw=1.2, zorder=1)
        ax.plot([i - 0.13], [base_best[retriever]], "o", ms=4.2, color=ACCENT, zorder=3,
                label="base" if i == 0 else None)
        ax.plot([i + 0.13], [sem_best[retriever]], "o", ms=4.2, color=WARN, zorder=3,
                label="+ filter" if i == 0 else None)
    ax.set_xticks(x, [short.get(r, r) for r in order], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("best proxy gain")
    ax.set_ylim(0, 0.115)
    hgrid(ax)
    ax.legend(loc="upper right", markerscale=0.7)
    tag(ax, "(a)")

    ax = axes[1]
    boot = load_blend_bootstrap("blend_eb")
    means = boot.mean()
    ax.bar(range(len(means)), means.values, width=0.55, color=ACCENT)
    ax.set_xticks(range(len(means)), [str(c) for c in means.index], rotation=30, ha="right")
    ax.set_ylabel("mean learned blend weight")
    ax.set_ylim(0, 1.05)
    hgrid(ax)
    tag(ax, "(b)")

    ax = axes[2]
    multi = load_gate_study_multi()
    colors = [NEUTRAL, NEUTRAL, BLIND, COND]
    ax.bar(range(len(multi)), multi["mean_delta"], width=0.6, color=colors)
    ax.set_xticks(range(len(multi)), ["off", "fixed", "sign", "oracle"], fontsize=7)
    ax.set_ylabel("mean Δcosine (pooled)")
    zero(ax)
    hgrid(ax)
    tag(ax, "(c)")

    ax = axes[3]
    runs = [("dense", "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std1"),
            ("hybrid", "M4_intersection_dev_blind_continuous_liftlaplace_eb_pool_std0.5_hybrid_rrf")]
    lows, highs = [], []
    for i, (label, folder) in enumerate(runs):
        row = resc_row(folder)
        color = ACCENT if i == 0 else BLIND
        ci = _ci(row)
        _dot(ax, i, row["delta_cosine_mean"], ci, color)
        lows.append(row["delta_cosine_mean"] - ci[0])
        highs.append(row["delta_cosine_mean"] + ci[1])
    ax.set_xticks([0, 1], [r[0] for r in runs])
    ax.set_ylabel("mean Δcosine (blind, dev)")
    ax.set_ylim(min(lows) - 0.0015, max(highs) + 0.0015)
    zero(ax)
    hgrid(ax)
    tag(ax, "(d)")
    fig.tight_layout(w_pad=1.4)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F7_negatives")


# --------------------------------------------------------------------------- F8
def fig8_control() -> None:
    from paper_lib import load_gate_study_decomposition, load_gate_study_learned
    learned = load_gate_study_learned()
    decomp = load_gate_study_decomposition()
    short = {"backoff_eb": "Backoff+EB", "intersection_laplace": "Team∩Class", "team_laplace": "Team"}

    fig = plt.figure(figsize=(DOUBLE, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.50, wspace=0.30)

    # (a) policy: always-on vs gated
    ax = fig.add_subplot(gs[0, :])
    focus = learned[learned["config_key"].isin(short)].copy()
    focus["pretty"] = [f"{short[k]} · {p}" for k, p in zip(focus["config_key"], focus["protocol"])]
    y = np.arange(len(focus))[::-1]
    for yi, row in zip(y, focus.itertuples()):
        color = COND if row.protocol == "conditioned" else BLIND
        ax.plot([row.always_on, row.eval_policy], [yi, yi], color="#c9c9c9", lw=1.6, zorder=1)
        ax.plot([row.always_on], [yi], "o", ms=5, color=NEUTRAL, zorder=3)
        ax.plot([row.eval_policy], [yi], "o", ms=5, color=color, zorder=3)
        ax.text(0.037, yi, f"rec {row.ceiling_recovery:+.2f}", ha="right", va="center", fontsize=7.5,
                color="#555555")
    ax.plot([], [], "o", color=NEUTRAL, label="always-on")
    ax.plot([], [], "o", color=COND, label="gated · conditioned")
    ax.plot([], [], "o", color=BLIND, label="gated · blind")
    ax.set_yticks(y, list(focus["pretty"]))
    ax.set_xlim(-0.048, 0.038)
    ax.set_ylim(-0.6, 6.15)
    ax.set_xlabel("mean Δcosine (eval)")
    ax.axvline(0, color="#444444", lw=0.8, ls="--")
    hgrid(ax)
    ax.legend(loc="upper left", ncol=3)
    tag(ax, "(a)")

    # (b) decomposition of what the gate removes
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(len(decomp))
    width = 0.34
    ax.bar(x - width / 2, decomp["mean_delta_open"], width, color=ACCENT, label="kept (open)")
    ax.bar(x + width / 2, decomp["mean_delta_closed"], width, color=BLIND, label="removed (closed)")
    # removed-set sizes are reported in Table IV and the caption (kept out of the panel
    # to keep the figure free of crowded labels)
    labels = [f"{'B+EB' if k == 'backoff_eb' else ('T∩C' if k == 'intersection_laplace' else 'Team')}\n"
              f"{'cond.' if p == 'conditioned' else 'blind'}"
              for k, p in zip(decomp["config_key"], decomp["protocol"])]
    ax.set_xticks(x, labels)
    ax.set_ylabel("counterfactual mean Δ (eval)")
    ax.set_ylim(-0.135, 0.055)
    zero(ax)
    hgrid(ax)
    ax.legend(loc="upper right")
    tag(ax, "(b)")

    # (c) substitution
    ax = fig.add_subplot(gs[1, 1])
    rec = [
        ("Uncalibrated\nblind", float(focus[(focus["config_key"].isin(["team_laplace", "intersection_laplace"])) &
                                            (focus["protocol"] == "blind")]["ceiling_recovery"].mean()), BLIND),
        ("Calibrated\nblind", float(focus[(focus["config_key"] == "backoff_eb") &
                                          (focus["protocol"] == "blind")]["ceiling_recovery"].iloc[0]), GREEN),
        ("Conditioned\n(all)", float(focus[focus["protocol"] == "conditioned"]["ceiling_recovery"].mean()), COND),
    ]
    for i, (label, value, color) in enumerate(rec):
        ax.bar([i], [value], width=0.5, color=color)
        ax.annotate(f"{value:+.2f}", (i, value), xytext=(0, 3 if value >= 0 else -10),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(range(3), [r[0] for r in rec])
    ax.set_ylabel("ceiling recovery")
    ax.set_ylim(-0.35, 0.75)
    zero(ax)
    hgrid(ax)
    tag(ax, "(c)")
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F8_control")


# --------------------------------------------------------------------------- F9
def fig9_magnitude() -> None:
    from paper_lib import load_magnitude_actions, load_magnitude_summary
    summary = load_magnitude_summary()
    actions = load_magnitude_actions()
    dev_pred = pd.read_parquet(OUT.parent / "results" / "magnitude_policy" / "dev_predictions.parquet")
    sample = dev_pred.sample(min(len(dev_pred), 6000), random_state=0)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.0))
    ax = axes[0]
    hb = ax.hexbin(sample["mag_pred"], sample["delta"], gridsize=40, cmap="viridis", mincnt=1)
    ax.axhline(0, color="white", lw=0.8)
    ax.axvline(0, color="white", lw=0.8)
    ax.set_xlabel("predicted Δ (dev, out-of-fold)")
    ax.set_ylabel("actual per-ticket Δ")
    gbr = summary["quality"]["gbr"]
    annot(ax, f"R² {gbr['r2']:.2f} · rank ρ {gbr['spearman']:.2f}", x=0.03, y=0.97, ha="left", va="top",
          color="#333333")
    tag(ax, "(a)")
    fig.colorbar(hb, ax=ax, shrink=0.85, label="rows", pad=0.02)

    ax = axes[1]
    oracle = float(actions["oracle_mean"].max())
    best_fixed = float(actions["best_fixed_mean"].max())
    vals = [float(actions["mean_delta"].iloc[0]), float(actions["mean_delta"].iloc[1]), 0.0, best_fixed]
    x = np.arange(4)
    ax.bar(x, vals, width=0.6, color=[BLIND, COND, NEUTRAL, NEUTRAL])
    for i, value in enumerate(vals):
        ax.annotate(f"{value:+.4f}", (i, value), xytext=(0, 3 if value >= 0 else -9),
                    textcoords="offset points", ha="center", fontsize=7)
    ax.errorbar([0, 1], vals[:2],
                yerr=[[vals[0] - float(actions["ci_lower"].iloc[0]), vals[1] - float(actions["ci_lower"].iloc[1])],
                      [float(actions["ci_upper"].iloc[0]) - vals[0], float(actions["ci_upper"].iloc[1]) - vals[1]]],
                fmt="none", ecolor="#333333", capsize=2.5, elinewidth=1.0)
    ax.axhline(oracle, color=ORACLE, ls=":", lw=1.0)
    ax.annotate(f"oracle {oracle:+.3f}", (3.45, oracle), xytext=(0, 3), textcoords="offset points",
                fontsize=7.5, color=ORACLE, ha="right")
    ax.set_xticks(x, ["sign\nselect", "expected\nvalue", "never-on", "best\nfixed"])
    ax.set_ylabel("mean achieved Δcosine (pooled)")
    zero(ax)
    hgrid(ax)
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.6)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F9_magnitude")


# --------------------------------------------------------------------------- F10
def fig10_validation() -> None:
    from paper_lib import load_answer_judge_summary
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.2))

    ax = axes[0]
    judge = load_answer_judge_summary()
    label_map = {
        "M2_team_eval_conditioned_continuous": "Team · conditioned",
        "M4_intersection_eval_conditioned_continuous": "Team∩Class · conditioned",
        "M4_intersection_eval_conditioned_continuous_gated": "Team∩Class · gated conditioned",
        "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2": "Calibrated blind · ungated",
        "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gated": "Calibrated blind · gated",
    }
    rows = []
    for run, s in judge["runs"].items():
        judged = s["feedback_wins"] + s["baseline_wins"] + s["ties_or_inconsistent"]
        pf, pb = s["feedback_wins"] / judged, s["baseline_wins"] / judged
        net = pf - pb
        se = np.sqrt(max(pf + pb - net ** 2, 1e-12) / judged)
        rows.append((label_map.get(run, run[:32]), net, max(net - 1.96 * se, -1),
                     min(net + 1.96 * se, 1), s["position_consistency"]))
    y = np.arange(len(rows))[::-1]
    for yi, (label, net, lo, hi, consistency) in zip(y, rows):
        color = COND if "conditioned" in label else GREEN
        _dot(ax, net, yi, (net - lo, hi - net), color)
        ax.text(0.98, yi, f"cons. {consistency:.2f}", ha="right", va="center", fontsize=7,
                color="#666666")
    ax.axvline(0, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_xlim(-0.60, 0.98)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("judge net win rate (feedback − baseline, 95 % CI)")
    tag(ax, "(a)")

    ax = axes[1]
    pairs = [
        ("Team", "M2_team_eval_conditioned_continuous", "M2_team_eval_conditioned_continuous_gengemini38flash"),
        ("Team∩Class", "M4_intersection_eval_conditioned_continuous",
         "M4_intersection_eval_conditioned_continuous_gengemini38flash"),
        ("Calibrated\nblind", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2",
         "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gengemini38flash"),
    ]
    x = np.arange(len(pairs))
    for offset, which, color, label in ((-0.15, 0, ACCENT, "luna (primary)"),
                                        (0.15, 1, WARN, "Gemini 3.8 Flash")):
        for i, (_, luna, gemini) in enumerate(pairs):
            row = resc_row(luna if which == 0 else gemini)
            _dot(ax, i + offset, row["delta_cosine_mean"], _ci(row), color, ms=4.6, lw=0.9)
    for i, (_, _, gemini) in enumerate(pairs):
        row = resc_row(gemini)
        ax.annotate(f"BGE p {row['delta_cosine_bge_wilcoxon_p']:.3f}", (i + 0.15, 0.062),
                    ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylabel("mean Δcosine (eval)")
    ax.set_ylim(-0.015, 0.082)
    zero(ax)
    hgrid(ax)
    ax.plot([], [], "o", color=ACCENT, label="luna (primary)")
    ax.plot([], [], "o", color=WARN, label="Gemini 3.8 Flash")
    ax.legend(loc="upper left", ncol=2, markerscale=0.7)
    tag(ax, "(b)")
    fig.tight_layout(w_pad=1.8)
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F10_validation")


# --------------------------------------------------------------------------- F11
def fig11_calibrated_null() -> None:
    from paper_lib import load_details
    fig, ax = plt.subplots(figsize=(SINGLE, 2.9))
    bins = np.linspace(-0.85, 0.85, 35)
    series = [
        ("M4_intersection_eval_blind_continuous", "legacy Laplace", BLIND),
        ("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "calibrated EB", GREEN),
    ]
    for folder, label, color in series:
        deltas = np.array([r["deltas"]["delta_cosine"] for r in load_details(folder)])
        ax.hist(deltas, bins=bins, histtype="step", lw=1.5, color=color, density=True, label=label)
        ax.axvline(deltas.mean(), color=color, lw=1.0, ls="--")
    zero(ax)
    ax.set_ylim(0, 7.4)
    ax.set_xlabel("per-ticket Δcosine (blind, eval)")
    ax.set_ylabel("density")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0)
    annot(ax, "means −0.038 vs +0.002\n[−0.007, +0.011]", x=0.97, y=0.97, ha="right", va="top",
          color="#333333")
    check_overlaps(fig)
    check_artists(fig)
    save(fig, "F11_calibrated_null")