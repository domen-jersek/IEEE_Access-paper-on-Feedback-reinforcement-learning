"""Front matter for the paper companion notebook."""
from __future__ import annotations

from companion_helpers import code, md


def cells() -> list[dict]:
    return [
        md("""
# When Does Feedback Help Retrieval-Augmented IT Support?

## A companion notebook: the complete evidence behind the paper

**What this notebook is.** This is the executed laboratory notebook behind the manuscript. Every
figure, every table, and every number in the paper appears here, in the same order, together with the
code that produced it and the reasoning that explains it. The notebook is deliberately written as a
scientific argument: each section states a question, explains the mechanism we expect *before*
looking at the data, shows the evidence, explains **why** the result comes out that way, checks
whether it could be an artefact, and closes with a one-line takeaway. No result is reported as "it
worked" or "it did not work" without a mechanism.

**Scope.** 1,595 anonymized enterprise IT support tickets; 878 train / 319 dev / 398 eval; five random
seeds; a procedure-disjoint regime. Two feedback protocols produced by an LLM judge: *conditioned*
(the judge sees the historical ground-truth reply) and *blind* (ticket-only). All generated runs are
anchored to one recorded generation regime (model + system prompt + response cache), and every number
in this notebook comes from an executed, warm-cache run.

---

## TL;DR — the paper in ten bullets

1. **The question.** Two tickets can look textually similar yet require different procedures. Feedback
   — promoting historical tickets that were judged useful — is the natural fix. When does it help?
2. **The design that separates the signal from the oracle.** The same retrieval is re-ranked twice:
   with a judge that sees the historical resolution (*conditioned*, an upper bound) and with a judge
   that does not (*blind*, deployable).
3. **The sign flips.** Conditioned fine-scope feedback improves generated replies on the held-out
   split (**+0.014** team, **+0.015** team∩class; BGE p ≤ .001); the same methods blind **harm**
   (−0.038). Broad scopes are negative under both.
4. **The failure is calibration, not the idea.** Judge scores are zero-inflated (59 % / 48 % exact
   zeros); a naive Laplace lift saturates at its negative cap (40–50 % of candidates) and overwhelms
   similarity.
5. **The fix is free.** Empirical-Bayes centering + pool-relative scaling removes the harm
   (offline blind −0.054 → +0.008…+0.015; generated +0.002, p=.86 — harm gone, residual gain not
   significant).
6. **The lift formula ordering is stable:** empirical Bayes > tanh > Laplace > Bayesian LCB, under both
   protocols.
7. **A gate is risk control, not gain.** Where harm is concentrated (uncalibrated blind) it recovers
   **0.54–0.56** of the oracle ceiling; once calibration removes the concentration it is neutral
   (−0.003) and under conditioned feedback it matches always-on (± 0.0005). Calibration and control are
   substitutes.
8. **Magnitude beats sign.** Expected-delta regression closes small, sharply harmful subsets and turns
   per-ticket action selection positive (**+0.0116**, 95 % CI [0.0034, 0.0205]) where the sign-only
   selector is negative (−0.0079).
9. **It replicates.** Independent metrics (BGE, BERTScore) and an independent LLM judge agree in
   direction; conditioned effects persist across seeds and unseen procedures, and a second generator
   (Gemini 3.8 Flash) reproduces and strengthens them (M4 +0.040).
10. **What we do not claim.** The deployable (blind) mean effect is null; feedback is LLM-judge
    simulated, the conditioned protocol is oracle-informed, the corpus is one organization, and the
    effects are small. All of that is documented in Section 26 and the FAQ.

---

## How to read this notebook

Every section has the same shape:

| Block | Meaning |
|---|---|
| **Why it matters / What you will see** | the question and the plan |
| **The idea** | the mechanism we expect *before* the data |
| *code* | the analysis that produces the evidence |
| **How to read it** | axes, units, sample size, reference lines |
| **What it shows** | the numbers and the claim |
| **Why this happens** | the causal explanation |
| **Could this be an artefact?** | controls, alternatives, robustness |
| **Takeaway** | one sentence |

Figures follow a message-first rule: their title is the conclusion, not the variable names. Sample
sizes, units, and splits are annotated on the figure.
"""),

        md("""
## Metric glossary

| Symbol | Meaning |
|---|---|
| **Δcosine** | per-ticket difference `cos(generated reply, historical reply) − cos(baseline reply, historical reply)`, averaged. Baseline = no feedback, so its mean is 0 by definition. |
| **[BGE]** | the same difference measured with BAAI/bge-base-en-v1.5 — an embedding family independent of the retriever (all-MiniLM). |
| **p** | paired Wilcoxon signed-rank test over tickets. |
| **CI** | 95 % bootstrap interval; for pooled dev statistics, tickets are resampled (clustered), not rows. |
| **Recovery** | `(policy − always-on) / (oracle − always-on)`: the fraction of the attainable ceiling a policy captures. 0 = always-on, 1 = oracle. |
| **AUC** | probability that a helped ticket receives a higher predicted score than a non-helped one; 0.5 = chance. |
| **PPV** | precision: among tickets the policy opens, the fraction that actually helped. |
| **Net win** | pairwise judge: `(feedback wins − baseline wins) / decided pairs`, consistent across both presentation orders. |
| **Context sensitivity** | `1 − cos(baseline answer, feedback answer)`: how much the generator's output changes when retrieval changes. |
| **MDE** | minimum detectable effect at 80 % power (paired): `2.80 · σ / √n`. Tells us which effects the study could ever confirm. |
"""),

        md("""
## Notebook map (paper section → here)

| Paper | Here | Content |
|---|---|---|
| 1 Introduction | S1 | problem, running example, contributions |
| 2 Related work | — | pointers only (no bibliography in the notebook, by design) |
| 3 Data & protocols | S2–S4 | corpus, splits, protocols, judge reliability, generation regime |
| 4 Method | S5, S14 | metrics, statistics, lift/routing/gate/magnitude construction |
| 5 Analysis I | S6–S10 | protocol reversal, scope, retrievers, formulas, evidence |
| 6 Analysis II | S11–S13 | zero-inflation, saturation, the fix, negative results |
| 7 Control | S15–S19 | gate, decomposition, substitution, magnitude, live runs |
| 8 Results & validation | S20–S24 | eval, independent metrics, judge, seeds/disjoint, cross-generator |
| 9 Discussion | S25 | practical recipe |
| 10 Limitations | S26–S27 | what we do not claim + reviewer FAQ |
| Appendix | A–C | worked examples, additional analyses, all runs, provenance |
"""),

        md("## Claim ledger — every paper claim and where it is proven"),

        code("""
import pandas as pd
claims = pd.DataFrame([
    ("C1  Protocol reversal (conditioned helps, blind harms)", "S6", "generated eval + BGE", "proven"),
    ("C2  Fine scopes help; broad pooling harms", "S7", "generated dev + six-retriever ladder ordering", "proven"),
    ("C3  Judge scores are zero-inflated and naive lift saturates", "S11", "judge distributions, cap saturation", "proven"),
    ("C4  EB centering + pool-relative scaling removes the harm", "S12", "offline rescue + generated", "proven (harm removal; gain n.s.)"),
    ("C5  Lift-formula ordering: EB > tanh > Laplace > LCB", "S9", "ablation under both protocols", "proven"),
    ("C6  A gate recovers harm only where harm is concentrated", "S16", "gate policy + decomposition", "proven"),
    ("C7  Calibration and control are substitutes", "S17", "recovery vs prior state", "proven"),
    ("C8  Magnitude modelling beats sign; action selection positive", "S18", "regressor + action selection", "proven"),
    ("C9  Independent metrics and an independent judge agree in direction", "S20–S21", "BGE/BERTScore + Claude judge", "proven"),
    ("C10 The calibrated blind gain does not transfer to stronger retrievers", "S8", "hybrid/CE generated runs", "proven (lift-config confound noted)"),
    ("C11 Conditioned effects persist across seeds and unseen procedures", "S22", "seeds 123/456 + disjoint", "proven (smaller; team seed456 ~0)"),
    ("C12 Cross-generator replication (Gemini 3.8 Flash)", "S23", "second generator family", "proven (stronger)"),
    ("N1  Semantic filter is worse than base routings", "S13", "offline, all retrievers", "negative result"),
    ("N2  Learned blend collapses to one scope", "S13", "bootstrap weights", "negative result"),
    ("N3  Sign-only multi-action selection loses", "S18", "action selection", "negative result"),
], columns=["claim", "section", "evidence", "status"])
display(claims)
"""),

        md("## Reproducibility at a glance"),

        code("""
resc = L.load_rescore_comparison()
status = L.artifact_status()
n_ok = int(status["available"].sum())
print(f"Frozen generated runs with independent metrics: {len(resc)}")
print(f"  dev runs:   {int(resc['run'].str.contains('_dev_').sum())}")
print(f"  eval runs:  {int(resc['run'].str.contains('_eval_').sum())}")
print(f"  generated by a second model family: {int(resc['run'].str.contains('gengemini38flash').sum())}")
print(f"Artifact checks available: {n_ok} / {len(status)}")
display(status[["section", "available"]].set_index("section").T)
"""),
    ]
