"""S11-S13: why feedback fails, the free fix, and the negative results with mechanisms."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, idea, meaning, md, sec, take, why


def cells() -> list[dict]:
    return [
        md("""
# Part III — Why does feedback fail?

The blind failure is not a mystery: it is a calibration defect with a mechanical explanation and a free
fix. This part derives the mechanism, shows the repair, and explains every negative result we obtained
along the way.
"""),

        # ------------------------------------------------------------------ S11
        sec("S11. Why does naive popularity lift harm retrieval?",
            "If the failure were 'the judge is bad', nothing could fix it. If it is a calibration defect, "
            "a correction should exist. We identify the mechanism precisely.",
            "Cap saturation under Laplace versus empirical Bayes, the lift-to-score variance ratio, and a "
            "two-vote derivation of the failure."),
        idea("""Laplace turns evidence counts into a probability with a uniform prior:
`p = (pos+1)/(pos+neg+2)`, and the bonus is `(p−0.5)·min(1, n/2)·0.8`, capped at ±0.20. Because the
judge's scores are zero-inflated, most candidates have **zero** positive votes and several negative
ones. A candidate with zero positives and two negatives already has p = 1/4, so its bonus is
(0.25−0.5)·1·0.8 = −0.20 — exactly the negative cap — and it stays there no matter how many more
negative votes accumulate. If a large share of the pool sits at the negative cap, the bonus stops
being a ranking signal and becomes a constant penalty whose magnitude exceeds the retrieval-score
spread, reordering the pool almost at random. The empirical-Bayes variant replaces the uniform prior
with the scope's base rate, so a candidate is judged against how popular its scope is."""),
        md("""
**The failure in three lines of algebra.** With Laplace (α = β = 1, cap 0.20, multiplier 0.8):

| negative votes | positive votes | p = (pos+1)/(pos+neg+2) | lift = (p−0.5)·min(1,n/2)·0.8 |
|---|---|---|---|
| 1 | 0 | 1/3 = 0.333 | −0.133 |
| 2 | 0 | 1/4 = 0.250 | **−0.200 (cap)** |
| 10 | 0 | 1/12 = 0.083 | **−0.200 (cap)** |

Two negative votes are enough. With a zero-inflated judge, that condition is the *modal* state of the
pool.
"""),
        code("""
sat = L.load_calib_saturation()
pivot = sat.pivot_table(index="routing", columns="lift", values="pct_at_minus_cap") * 100
pivot = pivot.reindex(["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"])

x = np.arange(len(pivot))
width = 0.26
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for offset, col, color in ((-width, "laplace", L.BLIND_COLOR),
                           (0.0, "laplace_eb_k2", L.COND_COLOR),
                           (width, "laplace_eb_k10", L.ACCENT_COLOR)):
    if col in pivot.columns:
        ax.bar(x + offset, pivot[col], width, color=color, label=col)
ax.set_xticks(x, ["M1 global", "M2 team", "M3 class", "M4 team∩class", "M5 backoff"])
ax.set_ylabel("share of pool candidates pinned at the negative cap (%)")
L.takeaway(ax, "Naive Laplace pins 40–50% of candidates at the negative cap; EB removes it")
L.note(ax, "feedback calibration on dev pools, 319 queries × 100 candidates")
ax.legend(frameon=False)
plt.tight_layout()
L.savefig("companion_11_saturation", run_ids=[])
plt.show()

view = sat[["lift", "routing", "pct_at_minus_cap", "pct_positive", "n_distinct_lift_values",
            "lift_std_over_faiss_std"]].copy()
view["pct_at_minus_cap"] = (view["pct_at_minus_cap"] * 100).round(1)
view["pct_positive"] = (view["pct_positive"] * 100).round(1)
display(view[(view["routing"].isin(["M1_global", "M4_intersection"]))].round(3).to_string(index=False))
"""),
        code("""
# Numerical simulation: a candidate with high similarity and unlucky evidence, under each lift.
import importlib
import src.feedback.lift as liftmod
importlib.reload(liftmod)
from src.config import LiftConfig

sim = pd.DataFrame([
    {"candidate": "A (high similarity, 0 pos / 2 neg)", "faiss": 0.95, "pos": 0.0, "neg": 2.0},
    {"candidate": "B (high similarity, 2 pos / 8 neg)", "faiss": 0.94, "pos": 2.0, "neg": 8.0},
    {"candidate": "C (lower similarity, 6 pos / 2 neg)", "faiss": 0.90, "pos": 6.0, "neg": 2.0},
])
rows = []
for _, r in sim.iterrows():
    lap = liftmod.compute_lift(r["pos"], r["neg"], LiftConfig.laplace())
    eb = liftmod.compute_lift(r["pos"], r["neg"], LiftConfig.laplace_eb(2.0), 0.30)
    rows.append({"candidate": r["candidate"], "faiss": r["faiss"],
                 "laplace lift": round(lap, 3), "laplace enhanced": round(r["faiss"] + lap, 3),
                 "EB lift": round(eb, 3), "EB enhanced": round(r["faiss"] + eb, 3),
                 "EB rank vs Laplace": "promoted" if (r["faiss"] + eb) > (r["faiss"] + lap) else "demoted"})
display(pd.DataFrame(rows))
print("Note how Laplace gives A and B the same −0.20 bonus despite very different evidence;")
print("EB keeps them ordered relative to the scope base rate.")
"""),
        howto("""Top: percentage of pool candidates whose lift sits exactly at the negative cap, per
routing and formula. Bottom: a three-candidate simulation with realistic evidence patterns, showing
the bonus Laplace and EB assign and the resulting enhanced scores. The table also reports lift-to-score
variance ratios."""),
        meaning("""Under Laplace, 40–50 % of candidates are pinned at the negative cap in every
routing, and even the empirical-Bayes variants retain some saturation only in the narrow scopes (3–7 %).
Measured as dispersion (the ratio of standard deviations), the Laplace bonus has 1.9–2.2× the spread of
the retrieval scores themselves, whereas pool-relative empirical Bayes is 1.2–2.0× (1.1–1.7× at κ = 10)
and ordered by evidence. The simulation makes
the consequence concrete: candidates A and B receive the *same* −0.20 penalty despite 0/2 versus 2/8
evidence, so their relative order is destroyed; EB preserves it and keeps both below the strong,
well-supported candidate C."""),
        why("""The defect is the interaction of two design choices: a uniform prior on a signal whose
empirical distribution is zero-inflated, and a bonus scale comparable to (or larger than) the
retrieval-score spread. Laplace behaves as if 'no evidence' were a 50/50 coin flip; in this corpus it
actually means 'no one found this useful'. The uniform prior therefore systematically over-rates
candidates with few negative votes and permanently buries everyone with two or more."""),
        artefact("""Saturation is measured on the same frozen feedback database used by all runs, so it
cannot be a sampling accident. The variance ratio uses the same pools. The simulation uses the exact
production lift implementation (imported, not re-implemented)."""),
        take("""Naive Laplace lift is not a weak signal; it is a biased one. Two negative votes with no
positives already pin a candidate to the cap, which is why the harm is systematic rather than random.
"""),

        # ------------------------------------------------------------------ S12
        sec("S12. Does the calibration fix work, and how far does it go?",
            "The mechanism predicts that centring the prior and scaling the bonus to the score units "
            "should remove the harm without hurting the resolution-informed case. We test the "
            "prediction offline across the whole grid and in generated answers.",
            "Offline rescue across routings and scalings, plus the generated before/after results."),
        idea("""Two changes target the two halves of the defect. **Centring** (empirical Bayes) replaces
the uniform prior with the scope's base rate, so popularity is measured relative to its scope.
**Scaling** expresses the bonus in units of the pool's score standard deviation, so the cap no longer
dominates the ranking. If the mechanism is right, the combination should turn the blind grid from
negative to slightly positive while leaving conditioned results largely intact."""),
        code("""
grid = L.load_ladder_grid("blind")
grid = grid[grid["routing"] != "none"].copy()


def best_for(routing, mask):
    part = grid[(grid["routing"] == routing) & mask]
    return float(part["minilm_d_proxy_top1"].max()) if len(part) else np.nan


routings = ["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"]
variants = [
    ("Laplace, absolute", grid["lift"].str.startswith("laplace|abs"), L.BLIND_COLOR),
    ("Laplace, pool-scaled", grid["lift"].str.startswith("laplace|std"), L.WARN_COLOR),
    ("EB, pool-scaled", grid["lift"].str.startswith("laplace_eb") & grid["lift"].str.contains("std"), L.COND_COLOR),
]
x = np.arange(len(routings))
width = 0.26
fig, ax = plt.subplots(figsize=(11, 4.8))
for offset, (label, mask, color) in zip((-width, 0.0, width), variants):
    vals = [best_for(r, mask) for r in routings]
    ax.bar(x + offset, vals, width, color=color, label=label)
ax.set_xticks(x, ["M1 global", "M2 team", "M3 class", "M4 team∩class", "M5 backoff"])
ax.set_ylabel("best offline proxy gain (top-1), blind")
L.zero_hline(ax)
ax.legend(frameon=False)
L.takeaway(ax, "Centring + pool-relative scaling turns the blind grid from negative to positive")
L.note(ax, "offline proxy, dense MiniLM, dev 319")
plt.tight_layout()
L.savefig("companion_12_rescue", run_ids=[])
plt.show()

before = resc_row("M4_intersection_dev_blind_continuous")
after = resc_row("M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2")
eval_after = resc_row("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2")
display(pd.DataFrame([
    {"configuration": "legacy Laplace, team∩class (dev)", "mean Δcosine": before["delta_cosine_mean"],
     "BGE": before["delta_cosine_bge_mean"], "p": before["delta_cosine_wilcoxon_p"]},
    {"configuration": "EB + backoff + pool scaling (dev)", "mean Δcosine": after["delta_cosine_mean"],
     "BGE": after["delta_cosine_bge_mean"], "p": after["delta_cosine_wilcoxon_p"]},
    {"configuration": "EB + backoff + pool scaling (eval)", "mean Δcosine": eval_after["delta_cosine_mean"],
     "BGE": eval_after["delta_cosine_bge_mean"], "p": eval_after["delta_cosine_wilcoxon_p"]},
]).round(4))
"""),
        howto("""Top: for each routing, the best blind offline gain under three prior/scaling variants
(legacy Laplace absolute, Laplace pool-scaled, empirical Bayes pool-scaled). Bottom: the generated
before/after numbers, dev and eval, with BGE and paired p-values."""),
        meaning("""Offline, every routing improves when the prior is centred and the bonus is scaled:
team∩class moves from strongly negative to positive, and the best cells reach +0.008…+0.015. In
generated answers the same repair changes the legacy blind team∩class result from −0.051 (dev) to
+0.010 for the calibrated backoff on dev and +0.002 on eval, with BGE +0.004 (p=.021). The harm is
gone; the residual gain is small and, as S5 showed, below the study's detection threshold on eval."""),
        why("""The centring term removes the scope base-rate bias: a candidate is now compared with
what is normal for its scope. The scaling term makes the bonus commensurate with the retrieval score
spread, so the cap no longer acts as a constant that rearranges candidates with high confidence.
Together they restore the ranking signal without any labels."""),
        artefact("""The offline rescue selects the best cell per routing, so it is optimistic; the
generated confirmation uses a single configuration chosen on dev *before* eval and does not tune on
eval. The eval gain is not significant, which we report rather than hide. A possible objection is that
we chose the fix after seeing the failure: the mechanism (S11) predicts it beforehand, and the fix is
the standard empirical-Bayes correction for a base-rate-biased estimator, not a fitted patch."""),
        take("""Calibration removes the harm for free. Expect risk reduction, not a large gain: after
calibration the remaining ticket-only effect is small and hard to detect."""),

        # ------------------------------------------------------------------ S13
        sec("S13. What else did not work, and why?",
            "Negative results are only convincing when they come with a mechanism. We document four: "
            "the semantic relevance filter, the learned blend, the hybrid-retriever transfer, and "
            "sign-only multi-action selection.",
            "Four small multiples with the corresponding rejection reason."),
        idea("""A negative result is evidence about the mechanism, not just a failed experiment. We
expected the semantic filter to help by removing textually distant candidates; the blend to combine
the strengths of several scopes; the calibrated lift to transfer to any retriever; and the sign gate
to choose the best configuration per ticket. Each failed for a specific, diagnosable reason."""),
        code("""
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))

# (a) semantic filter
sem = L.load_ladder_grid("semantic")
sem = sem[sem["routing"] != "none"].copy()
base_best = (sem[~sem["routing"].str.startswith("semantic")]
             .groupby("retriever")["minilm_d_proxy_top1"].max())
sem_best = (sem[sem["routing"].str.startswith("semantic")]
            .groupby("retriever")["minilm_d_proxy_top1"].max())
order = base_best.sort_values(ascending=False).index
x = np.arange(len(order))
axes[0].bar(x - 0.2, base_best[order], 0.4, color=L.ACCENT_COLOR, label="base best")
axes[0].bar(x + 0.2, sem_best[order], 0.4, color=L.WARN_COLOR, label="+ semantic filter")
axes[0].set_xticks(x, order, rotation=45, ha="right", fontsize=7)
axes[0].set_ylabel("best offline gain")
L.zero_hline(axes[0])
axes[0].legend(frameon=False, fontsize=7)
L.takeaway(axes[0], "Semantic filter hurts on every retriever", fontsize=9)
L.note(axes[0], "offline proxy, conditioned", fontsize=7)

# (b) blend weights collapse
try:
    boot = L.load_blend_bootstrap("blend_eb")
    means = boot.mean()
    axes[1].bar(means.index, means.values, color=L.COND_COLOR)
    axes[1].set_ylabel("mean learned weight (bootstrap)")
    axes[1].tick_params(axis="x", rotation=30)
    L.takeaway(axes[1], "Learned blend collapses onto one scope", fontsize=9)
    L.note(axes[1], "bootstrap resamples on train", fontsize=7)
except Exception as exc:
    axes[1].text(0.5, 0.5, f"blend artifact unavailable\\n{exc}", ha="center", va="center", fontsize=8)

# (c) sign multi-action
multi = L.load_gate_study_multi()
axes[2].bar(multi["policy"], multi["mean_delta"],
            color=[L.NEUTRAL_COLOR, L.BLIND_COLOR, L.NEUTRAL_COLOR, L.COND_COLOR])
axes[2].set_ylabel("mean Δcosine (pooled protocols)")
L.zero_hline(axes[2])
axes[2].tick_params(axis="x", rotation=25, labelsize=7)
L.takeaway(axes[2], "Sign-only action selection loses", fontsize=9)
L.note(axes[2], "796 ticket×protocol pairs", fontsize=7)

# (d) hybrid transfer
x = np.arange(1)
axes[3].bar([0, 1], [resc_row("M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std1")["delta_cosine_mean"],
                     resc_row("M4_intersection_dev_blind_continuous_liftlaplace_eb_pool_std0.5_hybrid_rrf")["delta_cosine_mean"]],
            color=[L.ACCENT_COLOR, L.BLIND_COLOR])
axes[3].set_xticks([0, 1], ["dense MiniLM", "hybrid RRF"])
axes[3].set_ylabel("mean Δcosine (blind, dev)")
L.zero_hline(axes[3])
L.takeaway(axes[3], "Calibrated gain does not transfer", fontsize=9)
L.note(axes[3], "different lift configs (confound noted)", fontsize=7)

plt.tight_layout()
L.savefig("companion_13_negative_results", run_ids=[])
plt.show()
"""),
        howto("""Four independent negatives. (a) Best offline gain without vs with the semantic
relevance filter, per retriever. (b) Mean learned blend weights across train bootstrap resamples for
the conditioned calibrated blend. (c) Policy values for never-on, always-on-best-fixed, sign-only
multi-action, and the oracle action. (d) Generated blind gain on dense MiniLM versus hybrid RRF."""),
        meaning("""(a) The semantic filter is worse than the base routing on every one of the six
retrievers: fine-scope metadata already encodes procedural relevance, so filtering on query-candidate
text cosine removes exactly the candidates feedback needs. (b) The learned blend places almost all
weight on a single scope; the other weights are statistically indistinguishable from zero, so the
blend adds parameters without adding signal. (c) Sign-only action selection is negative (−0.008)
because predicted probabilities are not comparable across configurations. (d) The calibrated blind
gain on dense MiniLM does not transfer to hybrid RRF (−0.006)."""),
        why("""The common thread is that each idea tried to add *orthogonal* information where none
existed. The semantic filter duplicated the scope signal with a worse proxy; the blend tried to
combine correlated lifts; the sign selector used relative confidences as if they were common-scale
utilities; and the retriever transfer assumed the lift is scale-free, which it is not."""),
        artefact("""Alternatives were tested: the semantic filter was tried at four thresholds
(0.5–0.8) and both protocols; the blend with a coarse and a fine weight grid and with bootstrap
stability checks; the multi-action selector with two different feature sets. All variants stay
negative."""),
        take("""Every negative result has an identified mechanism; together they bound the method:
scope metadata beats text filters, one scope beats blends, expected values beat signs, and
calibration must match the retriever."""),
    ]
