# Paper storyboard — narrative outline for the IEEE Access manuscript

Companion to `PAPER_OUTLINE.md` (claim matrix). This file is the *story*: what each section does,
the turn it delivers, and the numbers that carry it. Every number comes from a warm-cache artifact
of the frozen analysis; the gemini block is a separate generation regime by design.

---

## Positioning in one paragraph

The paper is a **conditional, mechanistic study**, not a "new method wins" paper. The story is a
trap and its resolution: feedback looks powerful when the judge knows the answer (resolution-informed
*conditioned* protocol) and turns harmful when it does not (*blind*, deployable protocol). We show
why (a zero-inflated, uncalibrated popularity signal), how to fix the harm (empirical-Bayes centering
+ pool-relative scaling), and where the remaining decision value lives (a pre-generation policy that
is **risk control, not gain**; and an expected-value decision layer that turns per-ticket action
selection positive). The tone is honest: the deployable mean effect is null; the contribution is the
conditional structure, the failure mechanism, and the control theory that follows.

## Title candidates (pick one)

1. *When Does Feedback Help Retrieval-Augmented IT Support? Oracle Conditioning, Calibration, and
   the Limits of Control*
2. *Feedback as a Conditional Retrieval Prior: A Systematic Study of LLM-Judge Signals for IT
   Support First Replies*
3. *Risk, Not Gain: Calibrating and Controlling Feedback-Driven Retrieval for IT Support Tickets*

## Abstract skeleton (≈180 words)

> Retrieval-augmented first-reply generation can promote historical tickets that a judge rated as
> useful even when they are less textually similar. We study when this helps and when it hurts on
> 1,595 enterprise IT tickets with two LLM-judge protocols: *conditioned* (the judge sees the
> historical resolution; an expert-with-answer upper bound) and *blind* (ticket-only; deployable).
> On identical retrieval, conditioned fine-scope feedback improves first replies (eval +0.014/+0.015
> cosine; BGE p ≤ .001) while blind feedback harms them (−0.038). We trace the failure to
> zero-inflated judge scores that saturate a naive popularity lift (40–50% of candidates at the
> negative cap), and remove it with empirical-Bayes centering and pool-relative scaling. A
> pre-generation gate is then **risk control rather than gain**: it recovers 0.54–0.56 of the oracle
> ceiling where harm is concentrated (uncalibrated blind) and is neutral once calibration removes
> that concentration. Finally, expected-value modelling beats sign-only gating and turns per-ticket
> action selection positive (+0.0116, 95% CI [0.0034, 0.0205]). We report independent metrics, an
> independent LLM judge, multi-seed, procedure-disjoint, and cross-generator evidence, and we state
> the limits of simulated feedback.

## Contribution list (for the introduction)

1. **Oracle conditioning quantified.** The same retriever, lift, and pools flip sign with the judge's
   information: conditioned +0.014/+0.015 vs blind −0.038 on the held-out split.
2. **A calibration mechanism and a free fix.** Judge scores are zero-inflated (58.7% exact zeros
   conditioned, 48.3% blind); a naive Laplace lift saturates at its negative cap; empirical-Bayes
   centering + pool-relative scaling removes the harm without labels.
3. **Control as risk management.** A pre-generation gate helps only where harm is concentrated;
   the open/closed decomposition makes this precise and shows calibration and control are
   substitutes.
4. **A magnitude-aware decision layer.** Expected-value regression explains substantial dev variance
   (R² 0.47), closes smaller and far more harmful subsets than the sign gate, and makes action
   selection positive where sign-only selection is negative.
5. **Rigorous, reproducible evidence.** Grouped cross-validation by ticket, dev-frozen thresholds,
   ticket-clustered intervals, generation-regime provenance, independent metrics, an independent
   judge, multi-seed, disjoint, and cross-generator replication.

---

# Section-by-section storyboard

## 1. Introduction — "the seductive signal and its trap" (≈1.5 pp)

**Job.** Motivate feedback re-ranking; state the diagnostic question; summarize the trap and the
resolution; list contributions.

**Beats.**
- Enterprise IT support is repetitive but not identical; dense similarity ≠ procedural correctness
  (the `.ica`-file vs form-redirect example from the SIKDD work is the running anecdote).
- The seductive idea: past judgments of usefulness should promote procedurally right candidates.
- The trap, stated as three questions:
  1. **When** does the signal help? (Answer preview: only when it is resolution-informed; the
     deployable variant harms.)
  2. **Why** does it fail? (Preview: zero-inflated scores + an uncalibrated lift; a calibration bug,
     not a rotten idea.)
  3. **Can we control it?** (Preview: a gate is risk control; expected-value modelling extracts the
     remaining per-ticket value.)
- One-sentence honest positioning: this is a systematic study with positive, null, and negative
  results, all reported.

**Figure/table.** None (or a small schematic of the pipeline + protocols).

## 2. Related work — "three conversations we join" (≈1.5–2 pp)

**Job.** Position the paper precisely; avoid overclaiming novelty of feedback reranking per se.

**Beats.**
- RAG and dense retrieval/reranking (Lewis; Reimers/MiniLM; FAISS; cross-encoders).
- Learning-to-rank from implicit/feedback signals (the idea is old; what is new here is the
  *reliability* of the feedback and its *calibration*).
- LLM-as-judge and judge reliability (Zheng et al.; judge-score biases; why oracle-conditioned
  judging is an upper bound).
- Selective prediction / abstention / control policies and decision-theoretic thresholds.
- Gap statement: prior feedback-reranking work rarely separates *what the judge knows* from *how the
  signal is turned into a score*, and almost never reports the counterfactual decomposition of where
  a gate helps or hurts. That is this paper's niche.

## 3. Data and protocols — "setting the stage honestly" (≈2.5 pp)

**Job.** Establish the corpus, the two protocols, and the rules that keep the study leak-free.

**Beats.**
- **Corpus.** 1,595 anonymized enterprise IT tickets (title, description, historical first reply;
  37 teams, 17 intent classes). Train 878 / dev 319 / eval 398; 5 random seeds; a procedure-disjoint
  regime (no near-duplicate procedure across splits, 390 eval tickets).
- **Two judge protocols** (the heart of the paper's identity):
  - *conditioned*: the judge sees query + **ground-truth historical reply** + candidate — an
    expert-with-answer simulation, an **upper bound**;
  - *blind*: query + candidate only — the **deployable** user simulation.
  - 175,600 judged pairs per protocol; judge `openai/gpt-5.6-luna`, temperature 0.
- **Judge reliability evidence**: score AUC vs same-reply 0.90 (conditioned) / 0.84 (blind);
  distribution is zero-inflated → median 0, 58.7% / 48.3% exact zeros. This is foreshadowing, not a
  digression — it is the mechanism of the later failure.
- **Generation regime and provenance**: one generator model + system prompt + response cache; every
  summary records `regime_id`, cache hits/misses, and a `warm` flag; the pre-cache dev runs were
  regenerated so all compared runs share a regime. State this as a methodological contribution to
  reproducible LLM experiments.
- **Statistical protocol**: grouped CV by ticket, dev-frozen thresholds, ticket-clustered bootstrap
  intervals, open/closed counterfactual decomposition. (This paragraph buys the reader's trust for
  everything later.)

**Figures/tables.** Corpus/split composition; judge reliability curve; protocol schematic.

## 4. Method — "the machinery" (≈2.5–3 pp)

**Job.** Describe retrieval, lift, routing, scaling, gate, and the decision layer, with equations but
no results yet.

**Beats.**
- Retrieval: MiniLM dense index (top-100 pool → top-5 context); ladder of six retrievers for
  sensitivity.
- Lift: Laplace, empirical-Bayes-centered Laplace, tanh, Bayesian LCB; cap and multiplier; the
  pool-relative scaling option (`lift = (p−0.5)·min(1,n/2)·m`, scaled to λ·pool-std).
- Routing scopes: global / class / team / team∩class / hierarchical backoff / learned blend (null).
- Gate features (pool-distribution, cosine-correlation, routing evidence) and the control policy;
  the expected-value (magnitude) policy and cost-sensitive cutoff.
- Evaluation: generated first replies compared to the historical reply with MiniLM cosine (primary),
  BGE cosine, ROUGE-L, BERTScore; paired Wilcoxon; the independent LLM judge; oracle ceilings.

**Figure/table.** Pipeline schematic; lift-shape figure.

## 5. Analysis I — "the sign flips with what the judge knows" (≈2.5 pp)

**Job.** Deliver the headline discovery.

**Beats.**
- **Protocol reversal.** Dev, one generation regime: M2 team +0.0220 (BGE +0.0133, p=.005), M4
  intersection +0.0314 (BGE +0.0183, p<1e-4) conditioned; the same methods blind: −0.0336 and
  −0.0506 (BGE p=.008/.001). Broad scopes (global, class) are negative under both.
- **Held-out confirmation.** Eval: M2 +0.0138 (p=.0117; BGE p<1e-4), M4 +0.0151 (p=.057; BGE
  p=.001); blind −0.0383 (BGE p=.03).
- **Scope ordering** is stable across six retrievers: fine scopes > broad pooling; the calibrated
  EB backoff is mildly positive under both protocols (+0.005 conditioned, +0.010 blind).
- **Retriever strength**: best base top-1 is dense MiniLM (0.799); blind gains are largest on the
  weakest base (BM25 +0.097) and shrink on stronger ones; generated checks show the calibrated blind
  gain does **not** transfer to hybrid RRF (−0.006, BGE p=.017) or cross-encoder base (+0.004),
  while baseline answer quality across retrievers is nearly identical (0.654–0.658).

**Figures.** Protocol-reversal bar chart; scope-ordering; retriever-strength curve.

## 6. Analysis II — "why it fails" (≈2.5 pp)

**Job.** Turn the reversal into a mechanism, then fix it.

**Beats.**
- **Zero inflation and saturation.** Judge scores cluster at 0; a naive Laplace lift saturates
  40–50% of pool candidates at the negative cap under global routing; the lift std exceeds the FAISS
  score std (≈2.1×), so feedback overwhelms similarity.
- **The harm is mechanical.** Blind feedback therefore pushes good candidates down; the effect is a
  calibration defect, not evidence that feedback is useless.
- **The fix.** Empirical-Bayes centering (per-scope prior) + pool-relative scaling: offline blind
  lift recovers from −0.051 to +0.007…+0.011; generated blind dev +0.0096 (p=.42), eval +0.0021
  (p=.86), disjoint +0.0051 (p=.16). **Honest framing:** the fix removes harm; the residual gain is
  not individually significant.
- **Formula ablation.** Ordering under both protocols: EB > tanh > Laplace > LCB (conditioned
  0.053/0.046/0.044/0.036; blind 0.011/0.007/0.006/0.005). The formula is a free choice; centering
  is what matters.
- **Volume and evidence.** More feedback is better only when it is trustworthy; fine scopes are
  evidence-limited, which motivates backoff.

**Figures.** Saturation; centering/scaling rescue; lift ablation.

## 7. Modeling the control policy — "who should intervene?" (≈2.5–3 pp)

**Job.** The modeling centerpiece. Deliver the "risk, not gain" result and its explanation.

**Beats.**
- **Framing.** A gate replaces feedback with the baseline on closed tickets, so it can only beat
  always-on if the closed set has a below-zero mean. This makes the counterfactual decomposition
  the natural diagnostic (gain = −(n_closed/N)·mean_closed).
- **Construction.** One general gate across all routing signals and both protocols, trained on dev
  (ticket × configuration rows) with grouped CV, thresholds frozen on dev; dev OOF AUC **0.699
  [0.676, 0.721]**; eval AUC 0.56–0.74.
- **Result.** Uncalibrated blind: policy +0.001…+0.002 vs always-on −0.0383 → recovery **0.54–0.56**
  (gains +0.039/+0.041). Calibrated blind: −0.003 (slightly worse). Conditioned: ±0.0005 (neutral).
- **Why — the decomposition.** The gate closes 152–235 tickets whose counterfactual mean is
  −0.07…−0.10 under uncalibrated blind (56–60% harmful); under the calibrated prior it closes 220
  tickets that are **62% no-ops** (mean +0.006): it removes a small positive tail. Conditioned closed
  sets are 55–90% no-ops (mean +0.003). **Law: calibration and control are substitutes.**
- **Magnitude-aware layer.** Regress expected delta (HistGradientBoosting) on the same features:
  within-configuration rank correlation ρ=0.17; it beats the sign gate on every blind configuration
  (intersection +0.0074 [−0.0020, +0.0177] recovery 0.63; team +0.0024; backoff +0.0035) because it
  closes only 46–66 tickets with means −0.28…−0.35. **Action selection turns positive: +0.0116
  [0.0034, 0.0205] with 19% abstention, vs −0.0079 for sign-only, +0.0002 best-fixed, +0.0571
  oracle.** Headroom remains, but the decision layer is settled: predict magnitude, not sign.
- **Live deployment.** The gate is applied inside the pipeline; live numbers match the dev-frozen
  post-hoc policy (e.g. intersection conditioned eval +0.0164 live), confirming integration.

**Figures/tables.** Gate AUC; decomposition; magnitude policy and action selection.

## 8. Results and validation — "does it survive?" (≈2 pp)

**Job.** Freeze the final story on the untouched split and harden it.

**Beats.**
- **Locked configuration** (chosen on dev): EB-centered Laplace, backoff minimum evidence 2,
  pool-relative scaling in the ticket-only regime; gate only as risk control; expected-value layer
  for per-ticket actions.
- **Eval table** with CIs and p-values (conditioned positive, blind negative, calibrated blind null).
- **Independent metrics**: BGE agrees in direction everywhere; the gated conditioned run is the
  strongest: cosine +0.0164 (p=.028), BGE +0.0130 (p=.0002), BERTScore +0.0160.
- **Independent judge** (Claude Sonnet 5, 100-ticket subsample, both orders): M2 net +0.18, M4
  +0.31, gated +0.31, calibrated blind +0.03 (gated −0.05 with 58% identical answers); position
  consistency 0.69–0.78. Honest caveat: per-ticket sign agreement with the embedding metric is low
  (0.35–0.40 conditioned) — aggregate direction agrees, per-ticket does not.
- **Robustness.** Conditioned seeds: M4 +0.0154/+0.0126, M2 +0.0063/−0.0006; procedure-disjoint:
  M4 +0.0152 (BGE p=.0065), M2 +0.0103 (BGE p=.058). Calibrated blind seeds +0.0049/−0.0009,
  disjoint +0.0051.
- **Cross-generator** (Gemini 3.8 Flash, separate regime): M2 +0.0290 (BGE +0.0151, p<1e-4), M4
  +0.0403 (BGE +0.0241, p<1e-4), calibrated blind +0.0073. The effect is not a single-model artifact.

**Figures/tables.** Eval table; independent-metrics figure; cross-generator figure; robustness table.

## 9. Discussion — "what a practitioner should take away" (≈1.5 pp)

**Job.** Convert results into guidance and boundaries.

**Beats.**
- Never evaluate feedback reranking with resolution-informed judging and call it a gain; the
  oracle-conditioning gap is the effect size (here, a sign flip).
- If you use LLM-judge feedback, calibrate before you re-rank: center on scope base rates and scale
  relative to the pool; otherwise the uncalibrated lift harms.
- A gate is insurance, not alpha: deploy it when feedback is untrustworthy or uncalibrated; once the
  prior is calibrated, the gate is neutral.
- For per-ticket decisions, model the expected delta; sign classifiers over-close no-ops and miss
  magnitude.
- Retriever choice matters as much as feedback: the feedback bonus shrinks as the base retriever
  strengthens, and its retention across retrievers is not guaranteed.

## 10. Limitations and conclusion — "the honest reckoning" (≈1 pp)

**Job.** State the boundaries without undermining the contribution; close the loop.

**Beats.**
- Feedback is **LLM-judge-simulated**, not human; conditioned is oracle-informed and is an upper
  bound; blind is the deployable setting and its mean effect is null.
- One organizational corpus and language; effects are small (+0.01–0.04 cosine); the qualitative
  metric is reference similarity, not resolution success.
- The judge and the primary generator are the same family; cross-generator replication mitigates but
  does not eliminate the concern.
- Per-ticket agreement between embedding metrics and the LLM judge is low even where aggregate
  directions agree.
- **Conclusion.** Feedback is a conditional prior: powerful when resolution-informed, harmful when
  uncalibrated, and controllable only where harm is concentrated — a finding about *when and why*,
  with a calibrated, risk-aware recipe that transfers.

---

## Where the optional/bonus material goes (if produced)

| Optional item | Placement |
|---|---|
| Human annotation study | new subsection in §8 (validation) and a limitation in §10 |
| Clean same-lift strong-retriever runs | §5 (retriever-strength subsection) |
| Downstream resolution/ticket-closure proxy | §8 (validation) and §9 (practical guidance) |

## Three sentences a reviewer should remember

1. Conditioned-judging gains do not transfer to the deployable (blind) setting: the sign flips.
2. The failure is a calibration defect — zero-inflated judge scores saturate a naive lift — and
   centering + pool-relative scaling fixes it for free.
3. A gate is risk control, not gain, and expected-value (magnitude) modelling is the right layer for
   per-ticket decisions.
