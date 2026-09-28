"""Appendices: manually checked examples, additional analyses, provenance."""
from __future__ import annotations

from companion_helpers import artefact, code, howto, meaning, md, take, why


def cells() -> list[dict]:
    return [
        md("""
# Appendix A — Manually checked examples

Numbers can hide what actually happens. These five tickets were read and checked by hand; each one
shows a mechanism from the paper in the raw data. Quotes are truncated for readability; the full
records are in the result JSON files.
"""),

        md("""### A1. Conditioned feedback fixes a semantic-vs-procedural confusion
*Query: "Connect to Client SW" (M4 team∩class, conditioned, eval).* The baseline retrieves a
network-access ticket because the words "client" and "connect" match; the reference is a local
software-support form. Conditioned feedback promotes the candidate whose reply is that form."""),
        code("""
show_case(RUN_M4_COND_EVAL, "R-377", top=2, chars=150)
"""),
        why("""The failure is lexical-semantic, not random. \"Client software\" and \"client network\" share
the word *client*; the correct procedure is distinguished by the resolution (a software form), which
only the conditioned judge can see. This is the canonical example of why feedback is needed and why
the deployable signal must be calibrated before it can help."""),

        md("""### A2. When the judge is right, the ranking can be exact
*Query: "Q2 AMAZON — 2x cavi HDMI 10MT + MIC" (M4 team∩class, conditioned, eval).* The reference is
a one-line accounting instruction; the baseline returns a different invoice registration, while
feedback returns a candidate whose reply is word-for-word the reference."""),
        code("""
record = show_case(RUN_M4_COND_EVAL, "R-662", top=2, chars=150)
print()
print("Delta:", round(record["deltas"]["delta_cosine"], 3))
"""),
        why("""Note that the promoted candidate's lift is slightly *negative*: the ranking is restored by
the *relative* ordering of lifts, not by a large positive bonus. This is important for interpreting
the mechanism: centring makes the whole scope comparable, so the right candidate can win even when
the scope is conservative. It also warns against reading a single lift value as 'importance'."""),

        md("""### A3. Blind feedback promotes popular boilerplate over the correct auto-link
*Query: onboarding device assignment (M4 team∩class, blind, eval).* The reference is an automatic
\"additional ticket created\" link. The blind judge rates a popular \"extra information needed\" form
reply as useful — it is genuinely useful for *other* onboarding tickets — and feedback promotes it,
destroying the answer."""),
        code("""
record = show_case("M4_intersection_eval_blind_continuous", "R-789", top=2, chars=150)
cand = record["feedback"]["retrieval"][0]
print()
print(f"Promoted candidate {cand['seq_id']}: intersection evidence = {cand['evidence']['intersection']}")
print(f"faiss similarity: {cand['faiss_score']:.3f}   lift: {cand['feedback_lift']:+.3f}")
print(f"Delta: {record['deltas']['delta_cosine']:+.3f}")
"""),
        why("""The promoted candidate had accumulated 30 same-scope votes (19.55 positive, 10.45
negative) because many *other* onboarding tickets genuinely need it. Without the resolution, the
judge cannot distinguish this ticket's auto-link resolution from the general onboarding form. The
lift is small (+0.114) but sufficient to outrank the correct, low-evidence link. This is the
popularity-without-resolution mechanism, measured on a real ticket."""),

        md("""### A4. The gate closes a harmful ticket (and the live pipeline agrees)
*Same failure family, calibrated blind configuration (M5 backoff+EB, eval).* The gate assigns a low
probability of help and keeps the baseline ranking; the ungated counterfactual for this ticket was a
large loss."""),
        code("""
show_case("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "R-1325", top=2, chars=150)
gated = load_records("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2_gated")["R-1325"]
print()
print(f"Gate decision: {'CLOSED (baseline used)' if gated['gate_active'] else 'open'}   "
      f"P(help) = {float(gated['gate_proba']):.3f}")
print(f"Gated delta (by construction): {gated['deltas']['delta_cosine']:+.3f}")
"""),
        howto("""The first block shows the ungated run (what would have happened). The second block shows
the live gated record for the same ticket: it was closed, so the pipeline returned the baseline and
the delta is exactly zero."""),
        why("""This ticket is the archetype the gate is designed to catch: the intersection scope has
some evidence, the promoted candidate is popular but wrong for this resolution, and the predicted
probability of help (0.22) is below the dev-frozen threshold. Closing it converts a −0.58 loss into a
no-op. Across the split, such closures are exactly the decomposition of Section S16."""),

        md("""### A5. Expected-value modelling closes a ticket the sign gate left open
*Calibrated blind (M5 backoff+EB, eval, ticket R-1354).* The sign gate assigns P(help) = 0.51 — above
its threshold, so it opens — while the magnitude model predicts a clearly negative expected delta and
closes. The ungated delta is −0.141."""),
        code("""
pred = pd.read_parquet(L.RESULTS / "magnitude_policy" / "eval_predictions.parquet")
row = pred[(pred["config_key"] == "backoff_eb") & (pred["protocol"] == "blind") &
           (pred["ticket_id"] == "R-1354")].iloc[0]
print(f"Sign gate:      P(help) = {row['sign_proba']:.3f}  -> open (threshold 0.30)")
print(f"Magnitude model: predicted Δ = {row['mag_pred']:+.3f} -> close (cutoff −0.015)")
print(f"Realized delta:  {row['delta']:+.3f}")
print()
show_case("M5_backoff_eval_blind_continuous_liftlaplace_eb_pool_std0.5_minev2", "R-1354", top=2, chars=140)
"""),
        why("""The sign gate answers \"is feedback more likely than not to help?\" and gets a 0.51 —
almost a coin flip. The magnitude model answers \"how large is the expected change?\" and sees a
clearly negative value because the configuration would promote boilerplate on a ticket whose
resolution is a link. This is precisely the sign-versus-magnitude gap: a 51 % chance of a small gain
loses to a 49 % chance of a large loss, but a sign classifier cannot see the difference."""),

        md("""
# Appendix B — Additional analyses
"""),

        md("""### B1. The offline proxy: valid for ordering configurations, weak per ticket
The proxy replaces generated answers with the similarity of the top-1 retrieved reply to the
reference. It was used to sweep configurations cheaply; its validity boundary matters."""),
        code("""
proxy_tickets = L.load_proxy_per_ticket()
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
sample = proxy_tickets.dropna(subset=["delta_cosine", "minilm_d_proxy_top1"])
axes[0].hexbin(sample["minilm_d_proxy_top1"], sample["delta_cosine"], gridsize=45, cmap="magma", mincnt=1)
axes[0].axhline(0, color="white", lw=1)
axes[0].axvline(0, color="white", lw=1)
axes[0].set_xlabel("proxy Δ (top-1 reply similarity)")
axes[0].set_ylabel("generated Δcosine")
from scipy.stats import spearmanr
rho = spearmanr(sample["minilm_d_proxy_top1"], sample["delta_cosine"]).statistic
L.takeaway(axes[0], f"Per ticket the proxy is weak (Spearman ρ = {rho:.2f})")
L.note(axes[0], f"n = {len(sample):,} ticket-run pairs")

runs = proxy_tickets.groupby("run")[["minilm_d_proxy_top1", "delta_cosine"]].mean().dropna()
axes[1].scatter(runs["minilm_d_proxy_top1"], runs["delta_cosine"], color=L.ACCENT_COLOR)
for name, row in runs.iterrows():
    axes[1].annotate(name.split("_dev_")[0], (row["minilm_d_proxy_top1"], row["delta_cosine"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
axes[1].axhline(0, color="black", lw=1)
axes[1].axvline(0, color="black", lw=1)
run_rho = spearmanr(runs["minilm_d_proxy_top1"], runs["delta_cosine"]).statistic
axes[1].set_xlabel("mean proxy Δ per run")
axes[1].set_ylabel("mean generated Δcosine per run")
L.takeaway(axes[1], f"Across runs the proxy orders methods (ρ = {run_rho:.2f})")
plt.tight_layout()
L.savefig("companion_24_proxy_validity", run_ids=[])
plt.show()
"""),
        why("""The proxy uses the *reference reply* to score candidates, so it knows which candidate
was procedurally right but cannot reproduce the generator's behavior (paraphrase, length, form
fidelity). Configuration-level comparisons therefore transfer (ρ = 1.0 in the validation report),
while per-ticket decisions do not. This is why every claim in the paper is computed from generated
answers and the proxy is used only for selection."""),

        md("""### B2. Which intent classes benefit (and which inherit the SIKDD finding)?
The earlier conference work found onboarding/offboarding dominating aggregate effects; we re-check
this on the generated eval runs."""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
for ax, folder, label, color in (
    (axes[0], "M4_intersection_eval_conditioned_continuous", "M4 conditioned", L.COND_COLOR),
    (axes[1], "M4_intersection_eval_blind_continuous", "M4 blind", L.BLIND_COLOR),
):
    recs = L.load_details(folder)
    frame = pd.DataFrame([{"class": r["expected_class"], "delta": r["deltas"]["delta_cosine"]} for r in recs])
    agg = (frame.groupby("class")["delta"].agg(["mean", "count"])
           .query("count >= 10").sort_values("mean"))
    ax.barh(agg.index, agg["mean"], color=color)
    for i, (name, row) in enumerate(agg.iterrows()):
        ax.annotate(f"n={int(row['count'])}", (row["mean"], i), va="center", fontsize=7,
                    ha="left" if row["mean"] >= 0 else "right",
                    xytext=(3 if row["mean"] >= 0 else -3, 0), textcoords="offset points")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("mean Δcosine")
    L.takeaway(ax, f"{label}: class heterogeneity with n ≥ 10")
    ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
L.savefig("companion_25_classes", run_ids=[])
plt.show()
"""),
        meaning("""The class pattern reproduces the qualitative SIKDD result: onboarding and offboarding
tickets dominate the tails in both directions because their resolutions are templated, while classes
with link-based or permission-based resolutions behave differently. This is a heterogeneity analysis,
not a routing rule: class-scope routing (M3) was negative under both protocols because class identity
alone is too coarse (S7)."""),

        md("""### B3. Identical-prompt audit
For some tickets the feedback reranking leaves the top-5 unchanged, so the baseline and feedback
prompts are identical. The pipeline generates once for those tickets (preventing nondeterministic
false deltas); the audit quantifies how many tickets this affects and what correcting for it changes."""),
        code("""
audit = L.load_identical_prompt_audit()
display(audit.round(4))
print("The correction is small: identical-prompt tickets cannot produce a delta, and the pipeline "
      "already handles them by generating once.")
"""),

        md("""### B4. All generated runs with independent metrics
The complete master table of the 39 evaluated runs (dev, eval, robustness, gated, disjoint, and
cross-generator), with the primary and independent metrics."""),
        code("""
master = L.load_rescore_comparison()
cols = ["run", "n", "delta_cosine_mean", "delta_cosine_wilcoxon_p",
        "delta_cosine_bge_mean", "delta_cosine_bge_wilcoxon_p",
        "delta_rouge_l_mean", "delta_bertscore_f1_mean"]
display(master[cols].round(4).to_string(index=False))
"""),

        md("""
# Appendix C — Provenance and artifacts
"""),
        code("""
provenance = pd.DataFrame([
    ("S1", "worked example records", "results/M4_intersection_eval_conditioned_continuous/*_details.json"),
    ("S2", "corpus and splits", "data/processed/dataset.parquet; splits/split_seed42.json; procedure_groups.json"),
    ("S3", "judge calibration", "results/feedback_calibration/{report.json, reliability_*.csv}"),
    ("S4", "generation regimes", "results/<run>/manifest_*.json; summaries' generation_regime blocks"),
    ("S5", "metrics + MDE", "results/rescored/*; results/<run>/*_details_rescored.csv"),
    ("S6-S7", "generated + ladder results", "results/rescored/method_comparison_v2.csv; results/retriever_ladder{,_blind}/grid.csv"),
    ("S8", "retriever runs", "results/retriever_ladder/ladder_curve.csv; generated alt-retriever runs"),
    ("S9", "lift ablation", "results/retriever_ladder_liftablation{,_blind}/grid.csv"),
    ("S10", "volume + coverage", "results/feedback_volume{,_blind}/*.csv; gate features"),
    ("S11", "saturation", "results/feedback_calibration/saturation.csv"),
    ("S12", "rescue", "results/retriever_ladder_blind/grid.csv; generated EB runs"),
    ("S13", "negative results", "semantic grids; blend bootstrap; multi_action.csv; hybrid run"),
    ("S15", "gate features + ablation", "results/gate_study_general/features_dev.parquet; gate_model.joblib"),
    ("S16-S17", "gate policy + decomposition", "results/gate_study_general/learned_gate{,_decomposition}.csv"),
    ("S18", "magnitude policy", "results/magnitude_policy/{policy_table,decomposition,action_selection}.csv"),
    ("S19", "live gated runs", "results/*_gated/*_summary.json; rescored gated runs"),
    ("S20", "independent metrics", "results/rescored/method_comparison_v2.csv"),
    ("S21", "independent judge", "results/answer_judge/{summary.json, scores.csv}"),
    ("S22", "seeds + disjoint", "results/M*_seed*; results/M*_disjoint"),
    ("S23", "cross-generator", "results/*_gengemini38flash"),
    ("B1-B4", "additional analyses", "results/proxy_validation/*; results/identical_prompt_audit.csv"),
], columns=["section", "content", "artifact"])
display(provenance)
status = L.artifact_status()
print(f"Artifact checks available: {int(status['available'].sum())} / {len(status)}")
"""),
        take("""Every number in this notebook traces to a committed artifact, and every artifact to a
command in S28; the notebook is the paper's evidence base, not an illustration of it."""),
    ]
