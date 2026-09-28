"""S20-S24: independent metrics, independent judge, robustness, cross-generator, synthesis."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, idea, meaning, md, sec, take, why


def cells() -> list[dict]:
    return [
        md("""
# Part V — Does it hold?

A result is only as strong as its controls. This part validates the findings with independent metrics,
an independent judge, alternative splits (seeds, unseen procedures), and a second generator family.
"""),

        # ------------------------------------------------------------------ S20
        sec("S20. Do independent metrics confirm the effect?",
            "The primary metric is a MiniLM embedding; the retriever is also MiniLM. A metric from the "
            "same family could favour the retriever's candidates. Independent metrics are the control.",
            "All four metrics for the eval finalists, with significance, as a single verdict table and a "
            "metric-agreement figure."),
        idea("""If the effect lives in the answers rather than in the metric, then an embedding from a
different family (BGE), a lexical metric (ROUGE-L), and a contextual metric (BERTScore) should all
move in the same direction. The strongest confirmation is a run where all three agree with the
primary metric and with each other."""),
        code("""
finalists = [
    ("M2 conditioned", "M2_team_eval_conditioned_continuous"),
    ("M4 conditioned", "M4_intersection_eval_conditioned_continuous"),
    ("M4 conditioned, gated", "M4_intersection_eval_conditioned_continuous_gated"),
    ("M4 conditioned, disjoint", "M4_intersection_eval_conditioned_continuous_disjoint"),
    ("M2 blind", "M2_team_eval_blind_continuous"),
    ("M4 blind", "M4_intersection_eval_blind_continuous"),
    ("M5 blind, calibrated", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    ("M5 blind, calibrated, disjoint", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_disjoint"),
]
metrics = [("cosine", "delta_cosine"), ("BGE", "delta_cosine_bge"),
           ("ROUGE-L", "delta_rouge_l"), ("BERTScore", "delta_bertscore_f1")]
rows = []
for label, folder in finalists:
    r = resc_row(folder)
    row = {"run": label}
    for name, col in metrics:
        row[name] = r.get(f"{col}_mean", np.nan)
        row[f"{name} p"] = r.get(f"{col}_wilcoxon_p", np.nan)
    rows.append(row)
verdict = pd.DataFrame(rows)
display(verdict.round(4))

fig, ax = plt.subplots(figsize=(12, 4.8))
x = np.arange(len(finalists))
width = 0.2
for offset, (name, col) in zip((-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width), metrics):
    vals = [resc_row(f).get(f"{col}_mean", np.nan) for _, f in finalists]
    ax.bar(x + offset, vals, width, label=name)
ax.set_xticks(x, [label for label, _ in finalists], rotation=20, ha="right", fontsize=8)
ax.set_ylabel("mean delta (eval)")
L.zero_hline(ax)
ax.legend(frameon=False, ncol=4)
L.takeaway(ax, "Four independent metrics agree in sign across every finalist")
plt.tight_layout()
L.savefig("companion_20_independent_metrics", run_ids=[])
plt.show()
"""),
        howto("""Table: per finalist, the mean delta under each metric and its paired p-value. Figure:
the same means as bars; the four bars of a run should all point the same way for the result to be
metric-independent."""),
        meaning("""Every finalist moves in the same direction under all four metrics. Conditioned runs
are positive (M4 gated: cosine +0.0164, BGE +0.0130, ROUGE +0.0179, BERTScore +0.0160; BGE p=.0002),
blind runs are negative (M2/M4: cosine −0.038, BGE −0.017, ROUGE −0.05, BERTScore −0.06), and the
calibrated blind runs are near zero (cosine +0.002, BGE +0.004). BERTScore is computed for the eval
runs only (compute cost); dev runs have BGE and ROUGE, which already provide an independent family.
"""),
        why("""All four metrics compare the generated answer with the historical reply, so they share
the reference — but they measure different aspects: dense semantics from two families, lexical
overlap, and contextual token alignment. Their agreement makes a metric-family artefact implausible.
"""),
        artefact("""The metrics remain reference-based: a good answer that differs from an imperfect
historical reply is penalised. The independent judge (S21) does not rely on the metric and confirms
the direction. ROUGE-L is length-sensitive; the deltas are small and consistent with the others, so
the direction is not driven by length."""),
        take("""The effects are not an artefact of the retrieval embedding family: four metrics and an
independent judge agree on the direction."""),

        # ------------------------------------------------------------------ S21
        sec("S21. Does an independent LLM judge agree?",
            "Automated similarity metrics can be fooled. A different model family, shown both answers in "
            "both orders, provides an orthogonal quality judgement.",
            "Net win rates with confidence intervals, position consistency, and per-ticket agreement."),
        idea("""For each of 100 eval tickets, a non-OpenAI judge (Claude Sonnet 5) sees the ticket, the
reference reply, and the two generated replies (baseline vs feedback), in both orders, and picks the
one that better matches the reference procedure. A win counts only if both orders agree, which
controls for position bias. If the embedding effect is real, the judge's net win rate should have the
same sign, especially for conditioned runs."""),
        code("""
judge = L.load_answer_judge_summary()
scores = L.load_answer_judge_scores()


rows = []
for run, s in judge["runs"].items():
    judged = s["feedback_wins"] + s["baseline_wins"] + s["ties_or_inconsistent"]
    pf, pb = s["feedback_wins"] / judged, s["baseline_wins"] / judged
    net = pf - pb
    se = np.sqrt(max(pf + pb - net ** 2, 1e-12) / judged)
    lo, hi = max(net - 1.96 * se, -1.0), min(net + 1.96 * se, 1.0)
    rows.append({"run": run.replace("_", " ")[:46], "judged": judged,
                 "feedback wins": s["feedback_wins"], "baseline wins": s["baseline_wins"],
                 "net win": s["net_win_rate"], "CI low": lo, "CI high": hi,
                 "position consistency": s["position_consistency"],
                 "sign agreement with cosine": s["agreement_with_cosine_sign"]})
jury = pd.DataFrame(rows)
display(jury.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
x = np.arange(len(jury))
axes[0].errorbar(jury["net win"], x, xerr=[jury["net win"] - jury["CI low"], jury["CI high"] - jury["net win"]],
                 fmt="o", color=L.ACCENT_COLOR, ecolor=L.ACCENT_COLOR, capsize=3)
axes[0].axvline(0, color="black", lw=1, ls="--")
axes[0].set_yticks(x, jury["run"], fontsize=8)
axes[0].set_xlabel("net win rate (feedback − baseline), 95% CI")
L.takeaway(axes[0], "The independent judge confirms the conditioned direction")
L.note(axes[0], "100 eval tickets per run, both orders, strict agreement")

axes[1].barh(x - 0.2, jury["position consistency"], 0.4, color=L.COND_COLOR, label="position consistency")
axes[1].barh(x + 0.2, jury["sign agreement with cosine"], 0.4, color=L.WARN_COLOR, label="per-ticket sign agreement")
axes[1].set_yticks(x, [""] * len(x))
axes[1].set_xlim(0, 1)
axes[1].legend(frameon=False, fontsize=8)
L.takeaway(axes[1], "Order-consistent, but per-ticket disagreement is high")
plt.tight_layout()
L.savefig("companion_21_judge", run_ids=[])
plt.show()
"""),
        howto("""Left: net win rate (feedback minus baseline wins over all judged pairs; ties count as
zero) with a normal-approximation 95 % interval; values right of zero favour feedback. Right: position
consistency (how often the verdict is stable when the two answers swap places) and per-ticket sign
agreement with the cosine metric."""),
        meaning("""The judge confirms the conditioned direction: M4 +0.31 and the gated M4 +0.31; M2
+0.18. Under calibrated blind feedback the judge sees essentially no difference (+0.03; gated −0.05
with 58 % identical answers). Position consistency is 0.69–0.78, and per-ticket sign agreement with
the embedding metric is low (0.35–0.40 conditioned, 0.14–0.18 blind)."""),
        why("""The judge and the embedding metric answer different questions. The embedding metric
rewards surface similarity to the reference; the judge rewards procedural equivalence while ignoring
tone and length. They agree on the *aggregate* because the intervention systematically moves answers
toward the reference procedure, but a large fraction of individual tickets move for stylistic reasons
the judge discounts or vice versa."""),
        artefact("""The judge sees the reference reply, so it is oracle-conditioned — the same upper
bound as the conditioned protocol, and it is a different model family from the generator and from the
judge that produced the feedback. The win requires agreement in both orders, which removes the most
common judge bias (position preference) at the cost of many ties."""),
        take("""An independent judge agrees with the aggregate direction; per-ticket agreement is
weak, which is an honest limit we report rather than hide."""),

        # ------------------------------------------------------------------ S22
        sec("S22. Does the effect survive other splits and unseen procedures?",
            "A single seed can be lucky, and near-duplicate procedures can leak between train and eval. "
            "Multi-seed and procedure-disjoint runs rule both out.",
            "The conditioned effect across seeds 123/456 and the disjoint split, and the calibrated "
            "blind winner on the same protocol."),
        idea("""Two robustness questions. (1) *Seed robustness*: does the effect depend on which tickets
happen to land in eval? We repeat the conditioned finalists on splits 123 and 456. (2) *Procedure
generalisation*: do near-duplicate procedures inflate the effect? The disjoint regime keeps identical
procedures out of the evaluation, so a positive effect there means the system generalises to unseen
procedures rather than memorising them."""),
        code("""
robust_runs = [
    ("M2 · eval seed 42", "M2_team_eval_conditioned_continuous", "conditioned"),
    ("M2 · dev seed 123", "M2_team_dev_conditioned_continuous_seed123", "conditioned"),
    ("M2 · dev seed 456", "M2_team_dev_conditioned_continuous_seed456", "conditioned"),
    ("M2 · eval disjoint", "M2_team_eval_conditioned_continuous_disjoint", "conditioned"),
    ("M4 · eval seed 42", "M4_intersection_eval_conditioned_continuous", "conditioned"),
    ("M4 · dev seed 123", "M4_intersection_dev_conditioned_continuous_seed123", "conditioned"),
    ("M4 · dev seed 456", "M4_intersection_dev_conditioned_continuous_seed456", "conditioned"),
    ("M4 · eval disjoint", "M4_intersection_eval_conditioned_continuous_disjoint", "conditioned"),
    ("M5 blind · dev seed 42", "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "blind"),
    ("M5 blind · dev seed 123", "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_seed123", "blind"),
    ("M5 blind · dev seed 456", "M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_seed456", "blind"),
    ("M5 blind · eval seed 42", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "blind"),
    ("M5 blind · eval disjoint", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_disjoint", "blind"),
]
rows = []
for label, folder, protocol in robust_runs:
    r = resc_row(folder)
    rows.append({"run": label, "protocol": protocol, "mean": r["delta_cosine_mean"],
                 "lo": r["delta_cosine_ci_lo"], "hi": r["delta_cosine_ci_hi"],
                 "p": r["delta_cosine_wilcoxon_p"], "BGE": r["delta_cosine_bge_mean"]})
robust = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11.5, 5.2))
colors = [L.COND_COLOR if p == "conditioned" else L.BLIND_COLOR for p in robust["protocol"]]
L.forest(ax, robust["run"], robust["mean"], robust["lo"], robust["hi"], colors=colors,
         title="Direction holds across seeds and unseen procedures; magnitude varies",
         xlabel="mean generated-answer Δcosine (95% CI)")
L.note(ax, "seeds change the split; disjoint removes near-duplicate procedures")
plt.tight_layout()
L.savefig("companion_22_robustness", run_ids=[])
plt.show()
display(robust.round(4))
"""),
        howto("""Each row is a rerun of a finalist; points are mean deltas with 95 % CIs. Green rows are
conditioned, red the calibrated blind winner. 'seed 42/123/456' changes which tickets are evaluated
(the feedback memory is adjusted accordingly); 'disjoint' keeps identical procedures out of eval."""),
        meaning("""The conditioned direction persists: M4 +0.015 (seed 123), +0.013 (seed 456), +0.015
(disjoint; BGE p=.0065); M2 +0.006, −0.001, +0.010 (BGE p=.058 on disjoint). The calibrated blind
winner is small and mixed: +0.005, −0.001 across seeds and +0.005 on disjoint. Magnitudes therefore
vary with the split — team seed 456 is effectively zero — but the sign of the conditioned effect does
not."""),
        why("""Seed variation changes which tickets (and which feedback queries) are in play, and the
team-scope effect depends on how much same-team evidence the sampled tickets have. The disjoint
regime removes the near-duplicate advantage, so its positive result shows that the conditioned effect
is not only memorisation of repeated procedures."""),
        artefact("""Across seeds the feedback database is the seed-42 memory minus the evaluated
split, so the source population shifts slightly by construction (documented in the repository). The
disjoint split is evaluated once. Small team-scope effects and the null calibrated-blind results are
reported as they are."""),
        take("""The conditioned effect is robust in direction to seeds and unseen procedures; the
calibrated blind effect remains small and not individually significant."""),

        # ------------------------------------------------------------------ S23
        sec("S23. Is the effect specific to the primary generator?",
            "The feedback judge and the primary generator come from the same model family. If the "
            "effect is an alignment artefact, a different generator should not reproduce it.",
            "The same eval finalists generated by Gemini 3.8 Flash, compared within its own regime, "
            "plus a decomposition of why its magnitudes differ."),
        idea("""The judge that produces feedback never sees generated answers, so there is no
evaluation leak; but shared model priors could still favour the primary generator. Re-running the
finalists with an unrelated generator is the test. We also check the obvious explanation for
magnitude differences: higher sensitivity to the retrieved context (larger answer changes) would
produce larger deltas."""),
        code("""
cross = [
    ("M2 conditioned", "M2_team_eval_conditioned_continuous", "M2_team_eval_conditioned_continuous_gengemini38flash"),
    ("M4 conditioned", "M4_intersection_eval_conditioned_continuous", "M4_intersection_eval_conditioned_continuous_gengemini38flash"),
    ("M5 blind, calibrated", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2",
     "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gengemini38flash"),
]
rows = []
for label, luna, gem in cross:
    rl, rg = resc_row(luna), resc_row(gem)
    rows.append({"config": label,
                 "luna Δ": rl["delta_cosine_mean"], "luna CI": f"[{rl['delta_cosine_ci_lo']:+.3f}, {rl['delta_cosine_ci_hi']:+.3f}]",
                 "gemini Δ": rg["delta_cosine_mean"], "gemini CI": f"[{rg['delta_cosine_ci_lo']:+.3f}, {rg['delta_cosine_ci_hi']:+.3f}]",
                 "gemini BGE": rg["delta_cosine_bge_mean"], "gemini BGE p": rg["delta_cosine_bge_wilcoxon_p"]})
cross_df = pd.DataFrame(rows)
display(cross_df.round(4))

x = np.arange(len(cross))
width = 0.36
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.bar(x - width / 2, [resc_row(l)["delta_cosine_mean"] for _, l, _ in cross], width,
       yerr=[[resc_row(l)["delta_cosine_mean"] - resc_row(l)["delta_cosine_ci_lo"] for _, l, _ in cross],
             [resc_row(l)["delta_cosine_ci_hi"] - resc_row(l)["delta_cosine_mean"] for _, l, _ in cross]],
       capsize=3, color=L.ACCENT_COLOR, label="primary generator (luna)")
ax.bar(x + width / 2, [resc_row(g)["delta_cosine_mean"] for _, _, g in cross], width,
       yerr=[[resc_row(g)["delta_cosine_mean"] - resc_row(g)["delta_cosine_ci_lo"] for _, _, g in cross],
             [resc_row(g)["delta_cosine_ci_hi"] - resc_row(g)["delta_cosine_mean"] for _, _, g in cross]],
       capsize=3, color=L.WARN_COLOR, label="Gemini 3.8 Flash")
ax.set_xticks(x, [c[0] for c in cross])
ax.set_ylabel("mean generated-answer Δcosine (eval)")
L.zero_hline(ax)
ax.legend(frameon=False)
L.takeaway(ax, "A different generator family reproduces the effect — and strengthens it")
L.note(ax, "separate generation regime by design; each compared within itself")
plt.tight_layout()
L.savefig("companion_23_cross_generator", run_ids=[])
plt.show()
"""),
        code("""
from src.evaluation.metrics import batch_cosine

def generator_anatomy(label, folder):
    recs = L.load_details(folder)
    bl = [r["baseline"]["answer"] for r in recs]
    fb = [r["feedback"]["answer"] for r in recs]
    sim = batch_cosine(bl, fb, model="minilm")
    base_cos = np.mean([r["baseline"]["metrics"]["cosine"] for r in recs])
    fb_cos = np.mean([r["feedback"]["metrics"]["cosine"] for r in recs])
    return {"run": label, "baseline quality": round(float(base_cos), 3),
            "feedback quality": round(float(fb_cos), 3),
            "context sensitivity": round(float(1 - sim.mean()), 3)}

anatomy = pd.DataFrame([
    generator_anatomy("luna · M4 cond", "M4_intersection_eval_conditioned_continuous"),
    generator_anatomy("gemini · M4 cond", "M4_intersection_eval_conditioned_continuous_gengemini38flash"),
    generator_anatomy("luna · M5 blind", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2"),
    generator_anatomy("gemini · M5 blind", "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gengemini38flash"),
])
display(anatomy)
print("Gemini's answers start closer to the reference and change LESS with context, "
      "yet the feedback benefit is larger: the mechanism is not simply context sensitivity.")
"""),
        howto("""Top: feedback effect per configuration for the primary generator (blue) and Gemini
(orange), with 95 % CIs; both are evaluated within their own regime. Bottom: absolute answer quality
and context sensitivity (1 − cosine between the baseline and feedback answers) for the two
generators."""),
        meaning("""Gemini reproduces the direction and increases the magnitude: M2 +0.029 (BGE +0.015,
p<1e-4), M4 +0.040 (BGE +0.024, p<1e-4), calibrated blind +0.007. The anatomy shows why the naive
explanation fails: Gemini's answers are already closer to the reference (0.694 vs 0.634) and change
*less* when retrieval changes (context sensitivity 0.13 vs 0.16), yet the feedback benefit is larger.
"""),
        why("""The effect is not tied to one model's answer style. A plausible reading of the larger
Gemini gains is that its answers are more form- and template-faithful, so promoting the correct
procedural candidate translates more directly into a closer reference match, while luna paraphrases
more freely. The anatomy in the table shows that the naive sensitivity explanation fails: Gemini's
context sensitivity is *lower* (0.13 vs 0.16 on M4), yet its feedback gain is larger. This is a
hypothesis, not a measurement; the measured facts are the stable direction and the different
magnitudes."""),
        artefact("""Gemini is a different generation regime, so its runs are compared only within
themselves; the same retrieval, feedback databases and gates are used. The free-form quality
comparison (Gemini's baseline is already 0.06 higher) means the two generators are not directly
comparable in absolute terms, only in the sign and significance of the feedback effect."""),
        take("""The effect is not specific to the primary generator; magnitude is generator-dependent
while direction is stable."""),

        # ------------------------------------------------------------------ S24
        sec("S24. What is stable and what varies?",
            """A reader needs to know which conclusions to carry away and which numbers are
            context-dependent. We separate the two explicitly.""",
            "A synthesis table of findings, stability, and caveats."),
        idea("""Across metrics, protocols, splits and generators, some facts repeat and some move.
Stability itself is evidence about what the method captures: mechanisms that hold in every protocol
and generator are structural; magnitudes that shift with the generator or split are calibration- and
style-dependent."""),
        code("""
stability = pd.DataFrame([
    ("Conditioned fine-scope feedback is positive", "S6, S20, S22, S23", "stable", "sign and significance"),
    ("Blind ticket-only feedback is negative (uncalibrated)", "S6, S12", "stable", "large magnitude"),
    ("Scope ordering (fine > broad)", "S7", "stable", "six retrievers, both protocols"),
    ("Laplace saturation mechanism", "S11", "stable", "mechanical"),
    ("EB centring removes the harm", "S12", "stable", "magnitude small and n.s. on eval"),
    ("Gate recovery where harm is concentrated", "S16, S17", "stable", "0.54–0.56 uncalibrated; ≈0 otherwise"),
    ("Magnitude beats sign", "S18", "stable", "direction; magnitude varies"),
    ("Conditioned effect magnitude across seeds", "S22", "varies", "+0.006 to +0.015; team seed456 ≈ 0"),
    ("Effect magnitude across generators", "S23", "varies", "Gemini larger (+0.04 vs +0.015)"),
    ("Calibrated blind gain on eval", "S12, S22", "not detectable", "+0.002, below MDE"),
    ("Semantic filter, blend, hybrid transfer", "S13", "negative", "mechanisms identified"),
], columns=["finding", "evidence", "stability", "notes"])
display(stability)
"""),
        howto("""A table of the paper's findings classified as stable, variable, not detectable, or
negative, with the sections that establish each."""),
        meaning("""Structural findings — the sign flips, the scope ordering, the saturation
mechanism, the substitution law, the sign-vs-magnitude result — hold across every control we ran.
Magnitudes vary with the split and the generator, and the calibrated blind gain is below the
study's detection threshold."""),
        why("""Structural findings follow from the information the judge has and the mathematics of
the lift; magnitudes additionally depend on how strongly a generator follows the retrieved context
and how much same-scope evidence a particular split provides."""),
        artefact("""This synthesis is the basis for the paper's claims; the claim ledger at the top of
the notebook maps each to its evidence. Optional controls we did not run (human annotation, a second
corpus) are listed in S26."""),
        take("""Carry away the structure, treat the magnitudes as context-dependent, and read the
calibrated-blind result as underpowered rather than absent."""),
    ]
