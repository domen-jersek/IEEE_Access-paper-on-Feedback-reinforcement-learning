"""S6-S10: does feedback help? Protocol reversal, scope, retrievers, formulas, evidence."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, idea, meaning, md, sec, take, why


def cells() -> list[dict]:
    return [
        md("""
# Part II — Does feedback help?

The protocol reversal is the centre of the paper. This part measures it, then asks which modelling
choices change it: the scope of the feedback (global → team∩class), the base retriever, the lift
formula, and the amount of evidence. Each subsection ends with the mechanism behind the pattern.
"""),

        # ------------------------------------------------------------------ S6
        sec("S6. Does the sign of the effect depend on what the judge knows?",
            "This is the paper's primary analysis. If the same reranking helps with resolution-informed "
            "judging and harms with ticket-only judging, then most published 'feedback gains' obtained "
            "with a strong judge may not transfer.",
            "Conditioned vs blind results for the four scopes on dev and the finalists on eval, with "
            "confidence intervals and independent metrics."),
        idea("""If feedback simply captured community knowledge, both protocols would help. Instead, the
conditioned judge can reward candidates that match the *resolution*, while the blind judge can only
reward candidates that *look* useful. The blind signal is therefore dominated by broadly popular
form-redirects; when the correct answer is something else (a link, a different form), popularity
actively misleads. We expect a sign flip, largest for fine scopes where the judge's information is
most decisive."""),
        code("""
resc = L.load_rescore_comparison()
DEV_RUNS = [
    ("M1 global", "M1_global_dev_conditioned_continuous", "M1_global_dev_blind_continuous"),
    ("M2 team", "M2_team_dev_conditioned_continuous", "M2_team_dev_blind_continuous"),
    ("M3 class", "M3_class_dev_conditioned_continuous", "M3_class_dev_blind_continuous"),
    ("M4 team∩class", "M4_intersection_dev_conditioned_continuous", "M4_intersection_dev_blind_continuous"),
]


def resc_row(folder):
    return resc[resc["run"] == folder].iloc[0]


x = np.arange(len(DEV_RUNS))
width = 0.36
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for offset, protocol, color in ((-width / 2, "conditioned", L.COND_COLOR), (width / 2, "blind", L.BLIND_COLOR)):
    means, los, his = [], [], []
    for _, cond, blind in DEV_RUNS:
        r = resc_row(cond if protocol == "conditioned" else blind)
        means.append(r["delta_cosine_mean"])
        los.append(r["delta_cosine_mean"] - r["delta_cosine_ci_lo"])
        his.append(r["delta_cosine_ci_hi"] - r["delta_cosine_mean"])
    ax.bar(x + offset, means, width, yerr=[los, his], capsize=3, color=color, label=protocol)
ax.set_xticks(x, [name for name, _, _ in DEV_RUNS])
ax.set_ylabel("mean generated-answer Δcosine (dev)")
L.zero_hline(ax)
ax.legend(frameon=False)
for i, (_, cond, blind) in enumerate(DEV_RUNS):
    ax.annotate(f"{resc_row(cond)['delta_cosine_mean']:+.3f}", (i - width / 2, 0.004), ha="center",
                fontsize=8, color=L.COND_COLOR)
    ax.annotate(f"{resc_row(blind)['delta_cosine_mean']:+.3f}", (i + width / 2, 0.004), ha="center",
                fontsize=8, color=L.BLIND_COLOR)
L.takeaway(ax, "The same reranking helps when the judge sees the resolution and harms when it does not")
L.note(ax, "dev, 319 tickets per run, 95% CI, n=878-ticket feedback memory")
plt.tight_layout()
L.savefig("companion_06_protocol_reversal_dev", run_ids=[])
plt.show()
"""),
        code("""
EVAL_ROWS = [
    ("M2 team", "M2_team_eval_conditioned_continuous", "M2_team_eval_blind_continuous"),
    ("M4 team∩class", "M4_intersection_eval_conditioned_continuous", "M4_intersection_eval_blind_continuous"),
]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
labels, means, los, his, colors = [], [], [], [], []
for name, cond, blind in EVAL_ROWS:
    for protocol, folder, color in (("conditioned", cond, L.COND_COLOR), ("blind", blind, L.BLIND_COLOR)):
        r = resc_row(folder)
        labels.append(f"{name}\\n{protocol}")
        means.append(r["delta_cosine_mean"])
        los.append(r["delta_cosine_ci_lo"])
        his.append(r["delta_cosine_ci_hi"])
        colors.append(color)
L.forest(ax, labels, means, los, his, colors=colors,
         title="The reversal repeats on the untouched eval split",
         xlabel="mean generated-answer Δcosine (eval, 398 tickets, 95% CI)")
L.note(ax, "conditioned +0.014/+0.015 · blind −0.038", loc="lower right")
plt.tight_layout()
L.savefig("companion_06_protocol_reversal_eval", run_ids=[])
plt.show()

cols = ["run", "n", "delta_cosine_mean", "delta_cosine_ci_lo", "delta_cosine_ci_hi",
        "delta_cosine_wilcoxon_p", "delta_cosine_bge_mean", "delta_cosine_bge_wilcoxon_p",
        "delta_rouge_l_mean", "delta_bertscore_f1_mean"]
finalists = resc[resc["run"].isin([
    "M2_team_eval_conditioned_continuous", "M4_intersection_eval_conditioned_continuous",
    "M4_intersection_eval_conditioned_continuous_gated",
    "M2_team_eval_blind_continuous", "M4_intersection_eval_blind_continuous",
    "M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2",
    "M4_intersection_eval_conditioned_continuous_gengemini38flash",
])][cols].round(4)
display(finalists)
"""),
        howto("""Bars are mean per-ticket Δcosine with 95 % bootstrap CIs; the dashed line is zero. Green
is the resolution-informed (conditioned) protocol, red the deployable (blind) protocol. The table
adds the independent metrics (BGE cosine, ROUGE-L, BERTScore) and paired Wilcoxon p-values for the
eval finalists."""),
        meaning("""On dev the pattern is unambiguous: conditioned M2 +0.022 (BGE +0.013, p=.005) and M4
+0.031 (BGE +0.018, p<1e-4); the same methods blind −0.034 and −0.051. On eval it repeats: M2 +0.014
(p=.012), M4 +0.015 (p=.057, BGE p=.001), while blind is −0.038 for both. Broad scopes are negative
under both protocols (M1 −0.028/−0.036, M3 −0.027/−0.060 on dev), so the failure is specific to the
combination of *fine scope* and *faithful judge*. The gain is concentrated: the gated conditioned run
reaches +0.0164 (BGE +0.0130, p=.0002)."""),
        why("""Three ingredients explain the flip. (1) Fine scopes give the judge a small, relevant
candidate set, so the score reflects procedural equivalence rather than generic plausibility.
(2) A conditioned judge can certify procedural equivalence because it sees the resolution; the blind
judge instead rewards replies that *look* like useful answers to similar tickets — very often the
popular onboarding/offboarding boilerplate — which is exactly wrong for tickets whose resolution is a
link or a different form (see the manually checked examples in Appendix A). (3) Broad scopes mix
unrelated procedures and add noise under every protocol."""),
        artefact("""The reversal could be a metric artefact (MiniLM scoring MiniLM retrievals). BGE
cosine, an independent embedding family, reproduces it in both direction and significance; an
independent LLM judge (S21) agrees in direction on the same runs. The comparison uses the same
candidate pools, retriever, lift and generator in both protocols, so the only difference is the
judge's information."""),
        take("""Resolution-informed feedback improves first replies for fine scopes; the deployable
ticket-only feedback harms them. The sign is controlled by what the judge knows, not by the
reranking mechanism."""),

        # ------------------------------------------------------------------ S7
        sec("S7. Which scope should receive the feedback?",
            "The scope determines the evidence-vs-noise trade-off: global pools have the most votes but "
            "mix unrelated procedures; team∩class has the most relevant votes but starves for data.",
            "Best offline gain per scope under both protocols, plus the evidence-coverage explanation."),
        idea("""Narrowing the scope should improve relevance but reduce the number of votes. We expect
a monotone pattern: broad scopes hurt, fine scopes help, with the optimum where relevance and
evidence balance. If instead the pattern were noise, it would not be stable across retrievers and
protocols."""),
        code("""
rows = []
for protocol, tag in (("conditioned", ""), ("blind", "blind")):
    grid = L.load_ladder_grid(tag)
    grid = grid[grid["routing"] != "none"]
    best = grid.groupby("routing")["minilm_d_proxy_top1"].max()
    for routing in ["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"]:
        rows.append({"protocol": protocol, "routing": routing, "best_gain": float(best.get(routing, np.nan))})
scope = pd.DataFrame(rows)
pivot = scope.pivot(index="routing", columns="protocol", values="best_gain").reindex(
    ["M1_global", "M2_team", "M3_class", "M4_intersection", "M5_backoff"])

generated = []
for name, cond, blind in DEV_RUNS:
    generated.append({"routing": name, "conditioned": resc_row(cond)["delta_cosine_mean"],
                      "blind": resc_row(blind)["delta_cosine_mean"]})
generated.append({
    "routing": "M5 backoff+EB",
    "conditioned": resc_row("M5_backoff_dev_conditioned_continuous_liftlaplace_eb_pool_std0.5_minev2")["delta_cosine_mean"],
    "blind": resc_row("M5_backoff_dev_blind_continuous_liftlaplace_eb_pool_std0.5_minev2")["delta_cosine_mean"],
})
gen = pd.DataFrame(generated)

x = np.arange(5)
width = 0.36
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].bar(x - width / 2, pivot["conditioned"], width, color=L.COND_COLOR, label="conditioned")
axes[0].bar(x + width / 2, pivot["blind"], width, color=L.BLIND_COLOR, label="blind")
axes[0].set_xticks(x, ["M1 global", "M2 team", "M3 class", "M4 team∩class", "M5 backoff"])
axes[0].set_ylabel("best offline proxy gain (top-1)")
L.zero_hline(axes[0])
axes[0].legend(frameon=False)
L.takeaway(axes[0], "The proxy only ranks finer scopes higher", fontsize=10)
L.note(axes[0], "offline proxy; every scope looks positive", fontsize=8)

axes[1].bar(x - width / 2, gen["conditioned"], width, color=L.COND_COLOR, label="conditioned")
axes[1].bar(x + width / 2, gen["blind"], width, color=L.BLIND_COLOR, label="blind")
axes[1].set_xticks(x, gen["routing"], fontsize=8)
axes[1].set_ylabel("mean generated-answer Δcosine (dev)")
L.zero_hline(axes[1])
axes[1].legend(frameon=False)
L.takeaway(axes[1], "Generated answers reveal that broad pooling actively harms", fontsize=10)
plt.tight_layout()
L.savefig("companion_07_scope", run_ids=[])
plt.show()

coverage = L.load_volume_curve()
print("Evidence coverage on dev (share of pool judged, by scope):")
print(coverage[["coverage_global", "coverage_class", "coverage_team", "coverage_intersection"]]
      .iloc[-1].round(3).to_string())
"""),
        howto("""Left: the best offline proxy gain per scope over all lift formulas and scalings
(offline proxy = similarity of the top-1 retrieved reply to the reference; used for configuration
selection, not claims). Right: the same scopes evaluated on *generated* answers on dev. The two panels
disagree in sign for the broad scopes, and that disagreement is the point."""),
        meaning("""The offline proxy ranks finer scopes higher under both protocols (global
+0.012/+0.008, team +0.075/+0.010, class +0.029/+0.011, team∩class +0.087/+0.015, backoff
+0.097/+0.013) but assigns a *positive* gain to every scope, including global and class. Generated
answers tell a different story: global and class are negative under both protocols (dev −0.028/−0.036
and −0.027/−0.060) while team, team∩class and the calibrated backoff are positive (+0.022/+0.031/+0.005
conditioned, −0.034/−0.051/+0.010 blind). The
reconciliation is the proxy's blind spot: it scores the top-1 candidate's reply text against the
reference and cannot see that a broadly popular but procedurally wrong candidate poisons the generated
answer. The proxy is valid for ordering configurations (B1), not for sign-level claims — which is why
every claim in this paper is computed from generated answers."""),
        why("""A vote from a same-team, same-class neighbour is far more likely to reflect the same
procedure. Broad scopes mix procedures, and their popularity signal is dominated by boilerplate; the
proxy's top-1 reply may even look similar to the reference, but the generator follows the wrong
procedure. Fine scopes therefore win both the proxy ordering and, decisively, the generated-answer
comparison."""),
        artefact("""The generated comparison uses the same candidate pools and generator in both
protocols and includes 95 % CIs and independent BGE metrics (S6, S20). The proxy/generated divergence
is not hidden: it is exactly the validity boundary quantified in B1 and is why the proxy was never
used for a claim."""),
        take("""Use fine scopes, never global pooling, and handle sparsity with hierarchical backoff;
sign-level conclusions require generated answers, not the proxy."""),

        # ------------------------------------------------------------------ S8
        sec("S8. Does the feedback gain survive a stronger retriever?",
            "If a better retriever already finds the right procedure, feedback may be redundant; if "
            "feedback captures something a retriever cannot, its gain should persist. Either answer "
            "changes how the method should be deployed.",
            "Offline gain vs base quality across six retrievers, plus generated runs on a hybrid and a "
            "cross-encoder retriever."),
        idea("""Two competing hypotheses. (a) Feedback substitutes for weak retrieval: it gains most
where similarity is poor (BM25) and little where it is strong. (b) Feedback supplies procedural
information orthogonal to similarity, so it should gain everywhere. We test both by pairing each
retriever with its own gain, then confirming with generated answers on a hybrid and a cross-encoder
base."""),
        code("""
curve = L.load_ladder_curve()

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
ax = axes[0]
ax.errorbar(curve["base_minilm_proxy_top1"], curve["best_gain"],
            yerr=[curve["best_gain"] - curve["primary_ci_lo"], curve["primary_ci_hi"] - curve["best_gain"]],
            fmt="o", ms=6, color=L.ACCENT_COLOR, ecolor=L.ACCENT_COLOR, capsize=3)
for _, r in curve.iterrows():
    ax.annotate(r["retriever"], (r["base_minilm_proxy_top1"], r["best_gain"]), fontsize=8,
                xytext=(5, 3), textcoords="offset points")
L.zero_hline(ax)
ax.set_xlabel("base retrieval quality (top-1 reply-similarity proxy)")
ax.set_ylabel("best offline feedback gain")
L.takeaway(ax, "Feedback gains are largest where retrieval is weakest")
L.note(ax, "offline proxy, dev 319, best cell per retriever")

ax = axes[1]
generated = [
    ("dense MiniLM", "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std1"),
    ("hybrid RRF", "M4_intersection_dev_blind_continuous_liftlaplace_eb_pool_std0.5_hybrid_rrf"),
    ("cross-encoder", "M4_intersection_dev_blind_continuous_liftlaplace_eb_k10_pool_std2_ce_rerank"),
]
means, colors = [], []
for label, folder in generated:
    r = resc_row(folder)
    means.append(r["delta_cosine_mean"])
    colors.append(L.ACCENT_COLOR if label == "dense MiniLM" else L.WARN_COLOR)
ax.bar([g[0] for g in generated], means, color=colors)
L.zero_hline(ax)
ax.set_ylabel("mean generated-answer Δcosine (dev)")
L.takeaway(ax, "But the calibrated gain does not transfer to better retrievers")
L.note(ax, "generated answers, blind, calibrated EB, 319 tickets")
plt.tight_layout()
L.savefig("companion_08_retriever", run_ids=[])
plt.show()

base_quality = []
for label, folder in generated:
    recs = L.load_details(folder)
    base_quality.append({"retriever": label,
                         "baseline answer cosine": round(float(np.mean([r["baseline"]["metrics"]["cosine"] for r in recs])), 4),
                         "feedback answer cosine": round(float(np.mean([r["feedback"]["metrics"]["cosine"] for r in recs])), 4)})
display(pd.DataFrame(base_quality))
"""),
        howto("""Left: each point is one retriever; x is base quality without feedback, y is the best
offline gain, with 95 % CI; annotated names. Right: generated blind runs with calibrated
empirical-Bayes feedback on three retrievers; the table lists the absolute answer quality per
retriever."""),
        meaning("""The offline ladder shows the substitution pattern: BM25 (weakest base) gains most
(+0.097), dense MiniLM (best base, 0.799) gains +0.053, and stronger dense/hybrid retrievers gain
between +0.050 and +0.072. On generated answers the story sharpens: calibrated blind feedback is
+0.007 on dense MiniLM, but −0.006 on hybrid RRF and +0.004 on a cross-encoder base, while baseline
answer quality is essentially identical across the three (cosine 0.654–0.658). The gain is not
retriever-independent."""),
        why("""Two mechanisms interact. Weak retrievers leave clear room for a procedural signal to fix
rankings; strong retrievers already put the right procedure near the top, so the lift mostly perturbs
an already-good order. In addition, the lift was calibrated for cosine-scale scores: RRF and
cross-encoder scores live on different scales, and although pool-relative scaling adjusts the
magnitude, the interaction between lift and the baseline ordering changes with the score geometry."""),
        artefact("""The three generated runs use different lift configurations (κ, λ, scaling), so
retriever and calibration are partly confounded; this is stated in the paper's limitations and is why
the claim is directional ('does not transfer') rather than a clean retriever ablation. The offline
ladder, which holds the calibration grid constant per retriever, supports the same direction."""),
        take("""Feedback helps most on weaker retrievers and cannot be assumed to transfer to stronger
ones; if a strong retriever is in place, validate feedback before deploying it."""),

        # ------------------------------------------------------------------ S9
        sec("S9. How much does the lift formula matter?",
            "If the result depends on the arbitrary shape of the popularity bonus, the method is not "
            "robust. We compare four formulas under both protocols.",
            "Lift shapes as a function of evidence, and the best gain per formula family."),
        idea("""The lift converts evidence into a score bonus. Laplace adds a uniform prior; the
empirical-Bayes variant centres on the scope's base rate; tanh is smooth and bounded; the Bayesian
lower-confidence bound penalises uncertainty. If evidence accumulation is the mechanism, all
reasonable shapes should help; if one shape fails, it should fail for an identifiable reason."""),
        code("""
from src.config import LiftConfig
from src.feedback.lift import compute_lift

pos = np.linspace(0, 20, 200)
neg = np.full_like(pos, 5.0)
configs = [
    ("laplace", LiftConfig.laplace(), None, L.NEUTRAL_COLOR),
    ("laplace_eb", LiftConfig.laplace_eb(2.0), 0.3, L.COND_COLOR),
    ("tanh", LiftConfig.tanh(), None, L.ACCENT_COLOR),
    ("bayesian_lcb", LiftConfig.bayesian_lcb(), None, L.WARN_COLOR),
]
fig, ax = plt.subplots(figsize=(9, 4.6))
for name, cfg, prior, color in configs:
    lifts = [compute_lift(p, 5.0, cfg, prior) for p in pos]
    ax.plot(pos - 5.0, lifts, color=color, label=name)
L.zero_hline(ax)
ax.set_xlabel("net evidence (positive − negative votes), negative votes fixed at 5")
ax.set_ylabel("lift applied to the retrieval score")
ax.legend(frameon=False)
L.takeaway(ax, "The formulas differ mainly in how they treat weak or negative evidence")
L.note(ax, "cap = ±0.20; EB centred on a scope prior of 0.3")
plt.tight_layout()
L.savefig("companion_09_lift_shapes", run_ids=[])
plt.show()
"""),
        code("""
def family(frame):
    return (frame["lift"].str.split("|").str[0]
            .str.replace(r"_k[0-9.]+$", "", regex=True)
            .str.replace(r"_s[0-9.]+$", "", regex=True))


rows = []
for protocol, tag in (("conditioned", "liftablation"), ("blind", "liftablation_blind")):
    grid = L.load_ladder_grid(tag)
    grid = grid[grid["routing"] != "none"].copy()
    grid["family"] = family(grid)
    for fam, part in grid.groupby("family"):
        best = part.sort_values("minilm_d_proxy_top1", ascending=False).iloc[0]
        rows.append({"protocol": protocol, "family": fam, "best_gain": float(best["minilm_d_proxy_top1"]),
                     "routing": best["routing"], "lift": best["lift"],
                     "p": float(best["primary_wilcoxon_p"])})
ablation = pd.DataFrame(rows)
display(ablation.round(4))

order = ["laplace_eb", "tanh", "laplace", "bayesian_lcb"]
x = np.arange(len(order))
width = 0.36
fig, ax = plt.subplots(figsize=(9.5, 4.4))
for offset, protocol, color in ((-width / 2, "conditioned", L.COND_COLOR), (width / 2, "blind", L.BLIND_COLOR)):
    vals = [ablation[(ablation["protocol"] == protocol) & (ablation["family"] == f)]["best_gain"].iloc[0] for f in order]
    ax.bar(x + offset, vals, width, color=color, label=protocol)
ax.set_xticks(x, order)
ax.set_ylabel("best offline proxy gain (top-1)")
L.zero_hline(ax)
ax.legend(frameon=False)
L.takeaway(ax, "Empirical Bayes > tanh > Laplace > LCB, under both protocols")
L.note(ax, "offline proxy, dense MiniLM, best cell per family")
plt.tight_layout()
L.savefig("companion_09_lift_ablation", run_ids=[])
plt.show()
"""),
        howto("""Top: the lift each formula assigns as a function of net evidence, with negative evidence
fixed at 5 votes and a cap of ±0.20. Bottom: the best offline gain achieved by each formula family
across routings and scalings, per protocol. The table lists the winning cell and its p-value."""),
        meaning("""The ordering is stable: empirical Bayes (0.053 conditioned, 0.011 blind) beats tanh
(0.046/0.007), then Laplace (0.044/0.006), then the Bayesian LCB (0.036/0.005). The differences are
modest but systematic, and the same ordering holds under both protocols."""),
        why("""Laplace treats a candidate with zero evidence as a coin flip and lets the cap pull it
down; tanh damps weak evidence smoothly; the LCB penalises small samples twice and therefore
over-regularises. EB centring removes the *scope-level* base rate — a ticket in a scope where
everything is popular no longer looks artificially good — which is the dominant bias on this corpus
(S11)."""),
        artefact("""The ordering could reverse at different caps or multipliers; both were fixed at
the documented defaults (±0.20, ×0.80) which were chosen on train. Section S12 shows that the
*centring* component is the operating ingredient: without it, no formula shape rescues blind
feedback."""),
        take("""Use empirical-Bayes centring; the exact smoothing shape is secondary. Model selection
should centre the prior, not tune the curve."""),

        # ------------------------------------------------------------------ S10
        sec("S10. Is more feedback always better, and what about sparse narrow scopes?",
            """Feedback needs evidence; narrow scopes are precise but sparse. If a ticket has no
            same-scope votes, its lift is uninformative, which is exactly why backoff exists.""",
            "Evidence coverage per scope and the feedback-volume curves for both protocols."),
        idea("""Two effects work against each other: more judged queries mean better estimates, but
within a narrow scope most candidates still have few votes. We expect gains to saturate with volume,
and starvation to appear as a mass of tickets with zero intersection evidence — the motivation for
hierarchical backoff."""),
        code("""
dev_feat = L.load_gate_study_features("dev")
sub = dev_feat[(dev_feat["config_key"] == "intersection_laplace") & (dev_feat["protocol"] == "conditioned")]
evidence = sub["evidence_intersection_top5"]
shares = {
    "no evidence in\\nbaseline top-5": float((evidence == 0).mean()),
    "< 2 observations\\n(backoff threshold)": float((evidence < 2).mean()),
    "< 10 observations": float((evidence < 10).mean()),
}
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
axes[0].bar(list(shares.keys()), list(shares.values()), color=[L.WARN_COLOR, L.ACCENT_COLOR, L.COND_COLOR])
for i, value in enumerate(shares.values()):
    axes[0].annotate(f"{value:.1%}", (i, value), ha="center", va="bottom", fontsize=9)
axes[0].set_ylim(0, max(shares.values()) * 1.5)
axes[0].set_ylabel("share of dev tickets")
L.takeaway(axes[0], "Narrow-scope starvation is a small tail here, not a mass")
L.note(axes[0], "dev, team∩class; baseline top-5 evidence")

cond = L.load_volume_curve()
blind = L.load_volume_curve("blind")
for curve, protocol, color in ((cond, "conditioned", L.COND_COLOR), (blind, "blind", L.BLIND_COLOR)):
    for scope, ls in (("M1_global", ":"), ("M2_team", "--"), ("M4_intersection", "-")):
        col = f"laplace_eb_{scope}_d_proxy_top1"
        if col in curve.columns:
            axes[1].plot(curve["fraction"], curve[col], ls=ls, color=color, label=f"{protocol} · {scope}")
axes[1].set_xlabel("fraction of the train feedback memory used")
axes[1].set_ylabel("offline proxy gain (top-1)")
L.zero_hline(axes[1])
axes[1].legend(frameon=False, fontsize=8)
L.takeaway(axes[1], "Gains saturate quickly; more feedback stops helping")
plt.tight_layout()
L.savefig("companion_10_evidence_volume", run_ids=[])
plt.show()
"""),
        howto("""Left: for team∩class on dev, the share of tickets with no same-scope evidence in their
baseline top-5, fewer than two observations (the backoff threshold), and fewer than ten. Right:
offline gain as a function of how much of the train feedback memory is used, for
global/team/intersection and both protocols (`laplace_eb`)."""),
        meaning("""On this corpus starvation is a tail, not a mass: only 2.5 % of dev tickets have no
team∩class evidence in their baseline top-5 and 3.4 % fall below the backoff threshold of two
observations — far below the ~33 % reported in the earlier 250-query study. The reason is the judging
budget: 200 candidates per train query make narrow-scope evidence dense. The dominant evidence
problem in this environment is therefore *quality* (zero-inflation, S11), not absence. Volume curves
still show that gains saturate by roughly the first 20–40 % of the memory, so a modest judging budget
suffices. Backoff remains the safe default: it never emits a zero-evidence lift for the tail."""),
        why("""With 878 train queries and 200 judged candidates each, a same-team, same-class candidate
usually has at least some votes; the full corpus simply has more neighbours per scope than the
250-query pilot. The remaining sparsity is a tail that backoff covers."""),
        artefact("""The 21.3 % figure in the calibration report measures a different quantity — how
often a zero-evidence candidate outranks a judged one — and is not a starvation rate. The numbers
here come directly from the gate feature table. The volume curves are offline proxies, used for
ordering only."""),
        take("""Feedback saturates quickly and severe starvation does not replicate on this corpus,
but backoff still protects the tail and is the safe default."""),
    ]
