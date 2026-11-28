# Response to reviewers — draft

**Manuscript:** Access-2026-52910
**Submitted title:** The Same Reranking, the Opposite Sign: When Feedback Helps
Retrieval-Augmented Generation for IT Support and When It Harms
**Proposed revised title:** When Feedback Gains Are Borrowed: Procedure Overlap,
Calibration, and the Limits of Similarity-Based Evaluation in RAG for IT Support
**Date:** [date of resubmission]

**Status:** working draft. It follows the IEEE Access three-part template:
(a) reviewer concern, (b) response, (c) action. Section references follow the
planned structure in `PAPER_SPINE.md` and must be updated after the rewrite.
Section, table and figure numbers refer to the revised manuscript. The items
still waiting for the second expert or a production step are listed at the end.

We thank both reviewers for a detailed and technically engaged reading. The
revision is a restructure rather than a patch. The main change is that the data
regime becomes the experimental variable: the submitted "procedure-disjoint"
split is shown not to be procedure-disjoint, and the headline reversal is
re-evaluated on history-only panels with group-aware splits. The calibrated
result is reported as harm removal, not as a confirmed gain. We also add the
controlled ablation, the reply-similarity baseline, the matched-retriever
comparison, the statistical sensitivity analysis, and a blinded expert
assessment of procedural correctness.

All values below come from frozen artifacts. The public provenance record is in
`results/revision/frozen/frozen_runs_v1.json`; aggregate tables are in
`revision_ieee_access/01_data_protocol/tables/`; the human analysis is in
`revision_ieee_access/02_human_study/`.

## Summary of the main changes

1. **Data regime.** Structural groups (identical full reply or identical
   title+subcategory; a conservative prefix variant), group-aware splits, and
   history-only retrieval, BM25 statistics, feedback rows and priors. The strict
   prefix panel is the primary control.
2. **Split audit.** The submitted "procedure-disjoint" split shares an identical
   full reply for 12.8 % and a reply opening for 53.3 % of its eval tickets with
   training. The random split shares 14.8 % and 58.3 %.
3. **Controlled repair.** One-factor-at-a-time dev ablation: tuned Laplace,
   empirical Bayes, pool-relative scaling, scope, backoff. The published legacy
   cell ranks 23/25 on the strict panel.
4. **Measurement.** A free reply-similarity baseline; a matched feedback
   configuration across dense, BM25 and hybrid retrieval; a blinded expert
   assessment with length control.
5. **Claims.** The calibrated gain is not confirmed; the strict protocol contrast
   is small and below its 80 % MDE; the retriever-strength conclusion is
   withdrawn; the gate is risk control, not a gain mechanism.

---

## Reviewer 1 — major comments

### R1-M1 — Title and abstract overclaim; no calibrated gain

**Concern.** The title promises an opposite sign caused by judge information,
but the sign flip exists only for the legacy Laplace lift at fine scope. After
calibration, blind is +0.002 (p=.86) and conditioned is also null. The accurate
summary is that an uncalibrated popularity lift helps under one judge and hurts
under another, while a calibrated lift does nothing under either.

**Response.** We agree. The submitted framing attributes to the judge a
difference that belongs mostly to the interaction between the legacy lift and
procedure overlap. The revision keeps the legacy reversal as a documented
result, but no longer presents it as the paper's causal claim. The new question
is whether any benefit survives when evaluation tickets do not share a procedure
or reply template with the historical base. On the strict panel, blind harm
disappears (+0.0008, p=.86), the conditioned effect is small (+0.0093), its
cosine interval excludes zero but the rank test is marginal (p=.066), ROUGE-L is
not significant, and the effect is below the 80 % MDE (0.0103). The calibrated
gain is therefore not confirmed. The title, abstract, RQ1 and conclusion are
rewritten around this result.

**Action.** New title and abstract; new RQ1; Section 5 documents the random-split
baseline and its scope; Section 6 makes the strict panel primary; Section 10
states that no calibrated gain is claimed. The words "opposite sign" and
"for free" are removed where they imply a causal or confirmed benefit.

### R1-M2 — The repair is not a controlled comparison

**Concern.** Table 3 and Section VI-C change scope, prior, scaling and the
Backoff fallback at once. Report EB with pool scaling at Team and
Team∩Class, and Laplace at Backoff, so each factor is isolated.

**Response.** We agree, and we replaced the bundled comparison with a
one-factor-at-a-time dev ablation under a matched selection budget. On the
duplicate panel the retrieval-level top-1 proxy is: legacy Laplace +0.0370
[0.0180, 0.0567]; tuned Laplace +0.0423 [0.0248, 0.0614]; Laplace + pool scaling
+0.0273 [0.0146, 0.0420]; EB κ=2 absolute +0.0497 [0.0319, 0.0688]; EB + pool
scaling +0.0238 [0.0126, 0.0368]; EB + pool scaling at Team +0.0132 [0.0009,
0.0271]; EB + pool scaling at Backoff +0.0217 [0.0076, 0.0370]. On the strict
panel the same sequence is −0.0083, −0.0023, −0.0042, −0.0008, −0.0002, −0.0002,
−0.0052. The scope change costs about half of the apparent duplicate-panel gain;
backoff does not recover it; nothing is positive on the strict panel.

We state the scope of this evidence explicitly: the ablation is a dev selection
experiment at retrieval level, and the generated panels confirm the frozen
finalist (EB κ=2, intersection, absolute scaling) on all three data regimes. We
did not generate answers for every ablation cell: the retrieval-level ablation is
the controlled selection experiment, and the generated panels confirm the selected
finalist. This scope is stated explicitly in Section 7.1.

**Action.** Section 7.1 and Table 4 report the full one-factor ablation; the
generated panel is labelled as confirmation of the finalist, not as a factorial
generation study. The text no longer credits "calibration" for the joint change.

### R1-M3 — The legacy baseline is a strawman

**Concern.** The legacy lift is added with m=0.80 and cap=0.20; its SD is
1.9–2.2 times the retrieval SD. m and cap were not tuned on dev while the
proposed variant had κ, λ and scaling tuned. Tune m and cap on dev with the same
budget, and support or drop the "common practice" claim.

**Response.** We agree. We ran a 25-cell Laplace sweep on dev with the same
candidate pools, feedback rows and selection metric used for the proposed
variant. The published cell (m=0.80, cap=0.20) ranks 9/25 on the duplicate panel
and 23/25 on the strict panel. Tuning moves the duplicate proxy from +0.0370 to
+0.0423 and the strict proxy from −0.0083 to −0.0023. Pool-relative scaling
alone reduces the lift/SD ratio from 2.30 to 0.50 on duplicate and from 0.95 to
0.21 on strict, which confirms the reviewer's reading: scale is a major cause,
and the prior alone is not. The legacy cell is now described as the inherited
configuration of the submitted study, not as "standard practice"; the sentence
and the missing citation are removed.

**Action.** Section 7.1 and Table 4 report the sweep, the published cell's rank, and the
tuned baseline; Section 4.2 removes "standard"; the abstract no longer implies
that the legacy cell represents common practice.

### R1-M4 — The two-vote algebra is mis-stated

**Concern.** Eq. 1 is antisymmetric: one positive gives +0.067, one negative
−0.067; two of either reach the cap. The observed asymmetry comes from the
zero-inflated judge, not from the prior. Pair the negative-cap sentence with the
symmetric positive statement.

**Response.** We agree; the original sentence confused the formula with the data
distribution. The revised text separates three facts. (i) The lift is
antisymmetric in (pos, neg), and the symmetry table is now in the paper. (ii)
The judge is strongly negative-heavy: conditioned scores are 62.9 % negative,
12.1 % positive and 25.0 % neutral; blind scores are 57.3 %, 7.8 % and 34.9 %.
(iii) At two votes, 13.2 % of conditioned candidates sit at the negative cap and
0.15 % at the positive cap; blind is 9.1 % and 0 %. At five votes the negative
cap is 15.6 % (conditioned) and 7.0 % (blind), while the positive cap is 0 %.
The asymmetry is therefore a property of the evidence distribution interacting
with the ramp, not of the prior alone.

**Action.** Section 4.2 rewrites both equations with the clamp explicit and adds
the symmetry table; the abstract says that two negative votes reach the cap
*and* that two positive votes do the same, while the data supply far more
negatives.

### R1-M5 — Conditioned is deployable; reply-similarity baseline is missing

**Concern.** The conditioned judge scores a candidate against the recorded reply
of a training ticket; both are historical. Only the query ticket lacks a
resolution at inference. The real contrast is a judge comparing two known
resolutions versus a judge guessing from text. Add an exact/embedding
reply-similarity baseline and clarify what is available at deployment.

**Response.** We agree with the data-flow correction. The conditioned judge uses
only historical resolutions; it is deployable for historical pairs and is not an
unrealizable upper bound. At inference on a new ticket, no resolution exists yet, so
only blind votes can be generated for that ticket; conditioned votes for the
historical base can be precomputed. We now call it a *resolution-informed*
judge and state this in the data-flow figure and in the limitations. We also
added exact and BGE reply-similarity baselines on exactly the same allowed pairs,
scopes and lift. On the random panel, reply similarity is at least as large as
conditioned feedback (+0.0143 [0.0047, 0.0242] versus +0.0105 [0.0005, 0.0212]).
On the duplicate panel it is smaller (+0.0081 [0.0016, 0.0152] versus +0.0149
[0.0059, 0.0246]); on strict it is +0.0026 [−0.0039, 0.0097] versus +0.0093
[0.0026, 0.0169], with the reply interval including zero. The simple baseline is
competitive only where procedure overlap remains, which is the paper's point.

**Action.** New data-flow paragraph and Figure 1; Section 3 states what is
available at feedback-construction, retrieval and evaluation time; Section 6 and
Table 3 add the reply-similarity rows. The phrase "unrealizable oracle" is
removed.

### R1-M6 — Statistical support is thin and internally inconsistent

**Concern.** Mean and SD imply t≈2.6 and t≈4.3, but the reported Wilcoxon
p-values are .017 and .010; the mean and median tell different stories. Report
paired bootstrap CIs for mean and median, state which statistic the claims rest
on, correct for multiplicity, and provide verifiable pre-specification.

**Response.** We agree that the original presentation mixed estimands. The
pre-specified statistic is the mean paired difference with a ticket-level
bootstrap interval; the median is reported beside it because the distribution is
zero-inflated (under conditioned feedback, 55.0 % of strict-panel and 22.6 % of
random-panel tickets have an unchanged answer). The median is exactly zero in most cells, so a rank test has
little power even when the mean interval excludes zero; this is a property of
the outcome, not an inconsistency. The two submitted blind eval runs, cited in
the concern, differ per ticket: mean absolute difference 0.0671 cosine units,
median 0.0337, only 11.8 % of per-ticket deltas identical, r=.799. Their means
coincide because both inherit the same saturated prior.

We now report direct protocol contrasts: random +0.0549 [0.0389, 0.0710];
duplicate +0.0363 [0.0217, 0.0517]; strict +0.0085 [0.0016, 0.0158]. Three
families were fixed before adjustment (conditioned and blind across panels;
reply similarity; protocol contrast) and Holm-corrected within family. Blind
harm on the random panel survives (p=.0009); conditioned on random does not
(p=.1165); duplicate conditioned does (p=.0106); strict conditioned does not
(p=.1965); the strict protocol contrast survives (p=.0322) but is below its
80 % MDE (0.0101). We describe it as small and imprecisely estimated, not as
evidence of absence.

We also withdraw the claim that the original primary comparisons were
prospectively pre-specified. The original study was not registered. The revision
has a dated protocol (`PROTOCOL.md`) and frozen artifacts, but the eval split had
already been used; the new analyses are revision-stage, not preregistered.

**Action.** Section 9.2 and Table 3 report mean and median intervals, the share of
unchanged tickets, direct contrasts, Holm-adjusted p-values and MDEs. Section 3
adds the provenance and pre-specification statement, including what cannot be
verified retrospectively.

### R1-M7 — Train/eval overlap; the disjoint split should be primary

**Concern.** A random split lets eval tickets share a procedure with training.
The procedure-disjoint split is the right control; its conditioned gain and
calibrated blind result should be primary, not a robustness row.

**Response.** We agree, and the revision goes further than moving a row. We
audited what the submitted splits guaranteed. The submitted "procedure-disjoint"
split combined reply prefix, title and subcategory into one hash; simultaneous
equality does not keep related tickets apart. Measured on the canonical corpus,
12.8 % of its 390 eval tickets share an identical full reply, 53.3 % share a
300-character reply opening, 9.7 % share title+subcategory, and 22.6 % are in
the same duplicate-union component as training. The random split shares 14.8 %,
58.3 %, 15.3 % and 29.1 %.

We rebuilt the control with versioned structural groups over connected
components, assigned whole groups to train/dev/eval, and produced a duplicate
panel (1,218 groups; 878/319/398) and a conservative prefix panel (676 groups;
856/312/427). Every retriever, BM25 document frequency, feedback row and prior is
restricted to training; self-pairs are removed. Five seeds were generated. The
prefix panel is the primary control, and the duplicate panel is the intermediate
one.

On the strict panel, blind harm disappears (−0.0444 → +0.0008, p=.86), the
protocol contrast shrinks six-fold (+0.0549 → +0.0085, Holm p=.0322), and the
conditioned effect is +0.0093 [0.0026, 0.0169] with a marginal rank test
(p=.066), non-significant ROUGE-L, and a value below the MDE. We also correct
the submitted "922 identical-reply tickets" statement: 922 tickets share a
300-character reply opening; 238 share an identical full reply.

**Action.** New Section 3 (structural groups, splits, allowed history), Table 2
(overlap audit) and Table 3 (generated effects by regime); the strict panel is
the primary result; the earlier claim is corrected in the text and in the
response to R1-M5.

### R1-M8 — Metric and judges are close to circular; add a human check

**Concern.** The metric is MiniLM cosine to one historical reply; MiniLM is also
the retriever; both judges are reference-conditioned. A small human check, even
100 tickets by two support engineers, would do more for validity than another
automated metric.

**Response.** We agree and added a blinded expert assessment of procedural
correctness. The sample contains 105 changed pairs (35 per arm): familiar
procedures with blind feedback, new procedures with blind feedback, and new
procedures with conditioned feedback. A/B order is randomised and balanced
within arm; the expert rated each reply (correct / partially correct / incorrect
/ cannot judge), flagged unsafe and missing steps, and chose the better reply,
with equal, both incorrect and cannot judge as explicit options. One GFT expert
(E1) has completed all 105 items; the second is in progress. We report E1
descriptively and will add agreement when E2 finishes.

The expert does not confirm the automatic harm. Of the 105 items, 101 entered
the paired strict-correctness analysis and four were cannot-judge. On the strict
arms the differences are −0.030 [−0.121, +0.061] (blind) and 0.000 [−0.088,
+0.088] (conditioned), with 3 and 2 discordant pairs. On the familiar arm the expert
prefers feedback by +0.147 [0.000, +0.294] (7 discordant; McNemar p=.25), while
cosine penalises it (−0.088). The feedback replies there are 173 characters
longer; the expert preference tracks length (Spearman +0.63), and among
length-matched pairs the effect is exactly zero (n=13). Across the sample, 42
items were rated equal, 32 both incorrect, and 2 cannot judge. With 2–7
discordant pairs per arm, this is a validity check, not a confirmatory test; we
do not claim that the expert confirms either harm or benefit.

**Action.** New Section 8, Table 6 and Figure 5; the abstract and conclusion say
that the expert does not confirm a difference on new procedures and that the
familiar-procedure preference is length-driven. Inter-rater agreement will be
added when the second expert completes; the analysis reports percent agreement,
Cohen's κ with bootstrap interval, weighted κ, and PABAK/AC1 on the common items.

### R1-M9 — Gate and expected-value results are overclaimed

**Concern.** Eq. 3 is a bookkeeping identity, not an exact decomposition law.
The gate helps only where always-on is negative, which follows almost directly.
The EV result is on 796 ticket-by-protocol rows, which are not independent, and
pools conditioned and blind configurations. State whether the CI is clustered
and why a deployed system would choose a protocol it cannot run.

**Response.** We agree on all three points. The bookkeeping identity is removed
from the main text; the control analysis is reduced to the hindsight bound. The
gate is no longer presented as a general gain mechanism: it is
risk control that pays only where the uncalibrated prior concentrates harm. On
the strict blind panel, always-on is +0.0008 and the hindsight upper bound is
+0.0190, so the maximum headroom is 0.0182; on the random blind panel,
always-on is -0.0444 and the hindsight bound is +0.0258, headroom 0.0701. We do not
claim that a feasible gate recovers the ceiling. The expected-value analysis is
removed from the main claims and moved to an exploratory appendix; the text
states that its rows are two per ticket and not independent, so the old interval
cannot be read as a clustered CI. We do not compare a deployable protocol with
one that cannot be run; the blind-only numbers are reported separately.

**Action.** Section 7.4 reports the hindsight bound and the blind-only numbers;
Section 10 states the limits; the EV numbers are removed from the abstract and
conclusion; the "exact decomposition law" and "calibration and control are
substitutes" sentences are deleted.

### R1-M10 — Single-corpus generalisation

**Concern.** One corpus, one language, one organisation; general lessons and a
five-step recipe are overreach. A public ticket dataset would allow an external
replication of the saturation diagnostic.

**Response.** We agree. The revision narrows the claims to one enterprise corpus
and treats the practical recipe as hypotheses to test elsewhere. We did not run
an external replication; no public ticket corpus with comparable procedure labels
and resolutions was available within the revision period, and we prefer to state
that than to substitute a different task. The saturation diagnostic (share of
candidates at the cap, share of exact zeros, lift/SD ratio) is cheap and is
released as a function so that others can compute it on their own corpora.

**Action.** Section 10 delimits the claim; Section 11 replaces the recipe with
conditional hypotheses; the appendix gives the diagnostic recipe and the code
path.

### R1-M11 — Reproducibility, license, model names, parameters

**Concern.** No license file; tickets are confidential, so headline numbers
cannot be regenerated; state exactly what a third party can rerun; name the
generator in the main text; give λ, κ and π_s per cell; give API identifiers and
dates.

**Response.** We agree and add the missing provenance. The generator and judge
are both `openai/gpt-5.6-luna` at temperature 0; the judge differs only in the
protocol prompt. The generation regime is `51fd9f4b37e900bb`, the system prompt
hash is `524098066b407de3`, and the run dates are September 2026 for the
submitted cache and November 2026 for the revision panels. The judge cache dates
are 12–13 September 2026. The finalist uses κ=2 and absolute scaling; λ=1 in the
pool-scaled ablation cells. π_s is the mean judge score of the scope over the
allowed training rows, with a global fallback when a scope has no rows. The per-cell values and the scope counts are in the
provenance table.

A third party can rerun three levels. (i) From the public repository, without
confidential data: all aggregate tables and figures, the human-evaluation
analysis from aggregate CSVs, the unit tests, and the data-protocol validation on
synthetic data. (ii) With the confidential corpus on premises: the full pipeline,
retrieval panels and generated panels with an API key. (iii) With new API calls:
generation and judging can be repeated, but hosted models may drift; the regime
and hashes are pinned rather than claiming bit-level reproducibility.

**Action.** Section 3 names the models at the start of the methods; Section 4
and the provenance table give κ, λ, π_s and the per-cell settings; the
reproducibility appendix states the three levels. License: the repository now contains `LICENSE` (MIT for code) and
`LICENSE-DOCS.md` (CC BY 4.0 for documentation and aggregate tables). The
confidential tickets, feedback pairs and generated answers are not released.

## Reviewer 1 — minor comments

### R1-m1 — Equations omit π_s, κ, λ and the clamp

**Concern.** Eq. 2 omits how π_s is estimated, how κ was chosen, and the value
of λ; Eq. 1 mixes cap and multiplier.

**Response.** Both equations are rewritten with the clamp explicit. The legacy
lift is p=(pos+α)/(n+α+β), L=clamp((p−0.5)·min(1,n/2)·m, −c, c). The
empirical-Bayes lift is p̄=π_s, a=κp̄, p_post=(pos+a)/(n+κ),
L=clamp((p_post−p̄)·m, −c, c). π_s is the mean judge score of the scope over the
allowed training rows, with a global fallback. κ was swept over {2,5,10,20,50}
and λ over {0.25,0.5,1,2,4} on dev; the finalist is κ=2 with absolute scaling,
and the pool-scaled cells use λ=1. Pool-relative scaling multiplies L by
λ·SD(pool)/c.

**Action.** Section 4.2, Eqs. (1)-(3); Table 4 lists the selected values and the
sweep grid.

### R1-m2 — Abstract too long and dense

**Concern.** State the main finding in plain terms first; remove terms defined
only in the body.

**Response.** The abstract is rewritten to ~200 words: problem, data regime,
three findings, limitation. "Pool-relative scaling" and "ceiling recovery" are
removed from the abstract.

**Action.** New abstract.

### R1-m3 — Nearly equal blind means

**Concern.** The two blind eval means agree to four decimals; one sentence on
the mean absolute per-ticket difference would settle it.

**Response.** The two runs differ per ticket: mean absolute difference 0.0671
cosine units, median 0.0337, 11.8 % of per-ticket deltas identical, r=.799.
Their means coincide because both inherit the same saturated prior; the
four-decimal agreement is a coincidence of aggregation, not duplicate runs.

**Action.** Section 5, one sentence and the correlation; the repeated
explanations in Sections V-A, VI-B and IX-D are consolidated.

### R1-m4 — Retriever comparison is confounded

**Concern.** Alternative-retriever runs use different lift configurations, so
retriever strength is confounded with calibration; soften or run one matched
configuration.

**Response.** We ran one matched configuration (EB κ=2, pool-relative scaling
λ=1) across dense MiniLM, BM25 and hybrid RRF. The feedback gain is random
+0.033, +0.064, +0.060 and strict +0.008, +0.009, +0.006. The earlier conclusion
that stronger retrieval reduces the benefit is a scaling artefact: absolute lift
units are cosine units and are not meaningful on BM25 scores. We withdraw the
conclusion and report the matched table.

**Action.** Section 7.2 and Table 5; the retriever-strength sentence is removed
from the introduction and conclusion.

### R1-m5 — Citations do not support sentences

**Concern.** [30] GPT-4 is cited for RAG on ticket reuse; [29] Transformers for
passing five candidates; MiniLM source missing; [28] BERT cited for dense
bi-encoders; arXiv-only entries; "standard" Laplace lift uncited.

**Response.** We removed or replaced the unsupported citations. RAG is cited to
Lewis et al. (NeurIPS 2020); the bi-encoder recipe to Reimers and Gurevych
(EMNLP 2019); the Transformers library only for implementation; the top-5
context choice is described as a design decision. Venue versions are used where
they exist. The legacy lift is described as the inherited configuration of the
submitted study, not as standard practice, and no citation is invented for it.

**Action.** Section 2 and the reference list; a citation audit table is in
`REFERENCES_TO_ADD.md`.

### R1-m6 — Missing related work

**Concern.** Add work on LLM relevance judgments and human agreement,
counterfactual/unbiased learning to rank, and Bayesian-average/Wilson ranking
with few ratings.

**Response.** We add a short related-work subsection with these threads: LLM
relevance judgments and their biases (Faggioli et al., ICTIR 2023; Zheng et al.,
NeurIPS 2023; Wang et al., ACL 2024; Panickssery et al., NeurIPS 2024);
unbiased learning to rank (Joachims et al., WSDM 2017); Bayesian averages and
Wilson intervals (Wilson, JASA 1927, plus a production-ranking reference); and
RAG evaluation (RAGAS, ARES, RGB, BEIR). Each citation is used in a sentence it
supports.

**Action.** Section 2; references verified before insertion.

### R1-m7 — Comparison with [31] and [32] is unclear

**Concern.** Give ticket counts and judge details side by side in a table.

**Response.** We add a systematic comparison table: corpus, evaluation queries,
judge and protocol, feedback pairs, lift, scopes, split, retrieval/evidence
boundary, baselines, statistics and main finding for SIKDD 2025, SIKDD 2026,
the submitted manuscript and the revision. The disclosure paragraph is expanded.

**Action.** Section 2.5 and Table 1; cover-letter disclosure.

### R1-m8 — Repetition across sections

**Concern.** Near-identical numbers repeat across Sections V, VII, VIII and
Tables 2, 4, 5; Section IX restates the abstract and Section X.

**Response.** We consolidated the numbers into the main tables and removed repeated
sentences. Section 9 now reports only validation, not a re-listing of the main
effects; Section 10 interprets rather than repeats.

**Action.** Editorial pass; a number-to-table map is used during the rewrite.

### R1-m9 — Template residue and layout

**Concern.** "VOLUME 11, 2023", DOI placeholder, LaTeX fragments, introduction
breaking across columns.

**Response.** We rebuilt the manuscript from the IEEE Access class, removed the
placeholder DOI and template residue, and checked the abstract and equations for
LaTeX fragments. The introduction is rewritten and the column flow checked in
the compiled PDF.

**Action.** Clean build; visual check of the highlighted PDF.

### R1-m10 — Name the generator and judge in Section III

**Concern.** The generator is only in Table 1; name it and the judge near the
start of Section III.

**Response.** Section 3 now names `openai/gpt-5.6-luna` with temperature,
prompt hash and dates for both roles, and states that the judge differs only in
what it sees.

**Action.** Section 3, first paragraph.

### R1-m11 — One success example is not enough

**Concern.** Add a concrete blind failure and a calibrated-neutral case.

**Response.** The appendix adds two anonymized cases selected by mechanism, not
by effect size: one where blind feedback promotes a wrong but popular candidate
and the generated reply keeps the wrong procedure; one where the calibrated
prior leaves the ranking unchanged. The conditioned success case is shortened.
Raw ticket text stays private; the published examples are paraphrased and
checked for identifying details.

**Action.** Appendix A.2; the two cases are anonymized and paraphrased.

### R1-m12 — "For free", "exactly", "standard"

**Concern.** Calibration needs a base rate and κ; saturation percentages are
measured, not exact.

**Response.** We removed "for free", "exactly" and "standard" where they imply
no cost or mathematical identity. Calibration is described as requiring π_s and
κ; saturation is described as measured on the allowed rows.

**Action.** Wording pass; the abstract and Section 6 are checked.

---

## Reviewer 2

### R2-1 — Independent human evaluation of procedural correctness

**Concern.** Evaluation is predominantly semantic similarity; add an independent
human evaluation of procedural correctness.

**Response.** Same study as R1-M8: 105 blinded changed pairs, three arms, one
expert complete and the second scheduled. The expert rates each reply and the
pair; strict and lenient correctness, unsafe/missing flags, ties and
both-incorrect are reported separately. On new procedures the expert sees no
difference; on familiar procedures the apparent preference is length-driven.
We do not claim confirmation of either automatic effect.

**Action.** Section 8, Table 6, Figure 5; agreement added when E2 completes.

### R2-2 — Conditioned judge and evaluation reference depend on each other

**Concern.** The conditioned protocol uses the historical response to generate
feedback, while the same response is the evaluation reference. Clarify the
impact.

**Response.** The dependency is real but narrower than direct circularity. The
conditioned judge scores a historical pair (q′, c′) using q′'s recorded
resolution; it never sees the evaluation ticket's reference. The evaluation
metric compares the generated reply for q to q's own reference. The data
boundary keeps q's reference out of retrieval, feedback construction and
generation. The remaining dependency is that both the judge and the metric
reward similarity to a recorded resolution, which is why the conditioned result
is an upper bound on what resolution-informed evidence can buy, and why we add
the blind protocol, the reply-similarity baseline and the expert check. The
metric still measures similarity to one recorded reply, not resolution success;
we state this in the limitations.

**Action.** Section 3 data-flow paragraph and Figure 1; Section 8 and Section 10.

### R2-3 — Simulated feedback is not real user feedback

**Concern.** Justify the representativeness of LLM-simulated feedback for
practical IT support.

**Response.** The feedback is simulated and the paper says so. We use it because
the corpus has no user click or rating stream, and because the protocol contrast
requires controlling what the judge knows. The judge is instructed to score
procedural equivalence, not fluency; it runs at temperature 0 with a pinned
prompt. The results are therefore about simulated feedback, not about the
prevalence of real user signals. The expert study validates the outcome side,
not the judge's labels; we do not claim judge-human agreement. A deployment
should recalibrate π_s on its own feedback and validate on a small expert sample.

**Action.** Section 3 and Section 10; the abstract says "simulated feedback" in
the limitations sentence.

### R2-4 — Single organisational corpus

**Concern.** Validate on an independent dataset or delimit applicability.

**Response.** We delimit applicability. The revision makes no cross-organisation
claim; it provides a diagnostic that can be computed elsewhere and states that
the magnitudes are corpus-specific.

**Action.** Section 10 and Section 11.

### R2-5 — +0.0021 is not significant; separate harm reduction from benefit

**Concern.** Clearly distinguish reducing degradation from demonstrating a
positive benefit.

**Response.** We agree and separate the two claims throughout. Calibration
removes the blind harm: on the strict panel the effect is +0.0008 [−0.0063,
+0.0084], p=.86. It does not demonstrate a gain: the submitted +0.0021 (p=.86)
and the strict +0.0008 are far below the MDE. The paper calls calibration a risk
fix, not an alpha generator, and the conclusion says no calibrated gain is
confirmed.

**Action.** Abstract, Section 6, Section 11; the phrase "removes the harm for
free" is deleted.

### R2-6 — Confidence intervals, direct protocol tests, multiplicity

**Concern.** Report CIs for main comparisons, directly test protocol
differences, justify multiple-comparison handling.

**Response.** We report ticket-clustered paired bootstrap intervals for every
main cell, direct conditioned−blind contrasts (random +0.0549 [0.0389, 0.0710];
duplicate +0.0363 [0.0217, 0.0517]; strict +0.0085 [0.0016, 0.0158]), and Holm
correction within three pre-specified families. Exploratory comparisons are
labelled as such and are not used for confirmation.

**Action.** Section 9.2 and Table 3; the statistical protocol paragraph.

### R2-7 — MDE is not evidence of absence

**Concern.** Clarify MDE interpretation; do not equate non-significance with no
effect.

**Response.** We state that the MDE is a sensitivity floor for the design, not
an equivalence boundary. The strict conditioned effect (0.0093) and the strict
protocol contrast (0.0085) are below their 80 % MDEs (0.0103 and 0.0101), so we
describe them as small and imprecisely estimated. We do not claim equivalence;
an equivalence claim would need a pre-specified margin and a corresponding test.

**Action.** Section 9, one paragraph; the word "null" is used only for
"no confirmed effect", not for "no effect".

### R2-8 — Novelty versus reranking, empirical Bayes and intervention selection

**Concern.** Distinguish the contribution from existing reranking, EB
calibration and intervention selection.

**Response.** We do not claim a new reranker or a new empirical-Bayes smoother.
The contribution is diagnostic and measurement-oriented: (i) procedure overlap
as the experimental variable, with an audit showing the submitted disjoint split
was not disjoint; (ii) a controlled decomposition showing that scale and scope,
not only the prior, drive the apparent repair; (iii) a reply-similarity baseline that uses no LLM judge that is competitive only where procedures overlap; (iv) a blinded
expert check showing that similarity to one reply and procedural correctness
diverge, and that expert judgement is itself length-sensitive. The gate is
reduced to risk control; intervention selection is no longer a headline
contribution.

**Action.** Introduction contributions, Section 2, Section 10; the abstract lists
the four contributions.

### R2-9 — Relationship to [31] and [32]

**Concern.** Systematic comparison of contributions and experimental results.

**Response.** Same as R1-m7: a side-by-side table with corpus, queries, judge,
pairs, lift, scopes, split, baselines, statistics and findings, plus an expanded
disclosure paragraph.

**Action.** Section 2.5 and Table 1.

### R2-10 — Keep feedback configurations constant across retrievers

**Concern.** Run additional experiments with the same feedback configuration for
all retrieval methods.

**Response.** Done. With EB κ=2 and pool-relative scaling λ=1, the feedback gain
is random +0.033 (dense), +0.064 (BM25), +0.060 (hybrid); strict +0.008, +0.009,
+0.006. The earlier retriever-strength conclusion is withdrawn as a scaling
artefact.

**Action.** Section 7.2, Table 5 and Figure 3.

### R2-11 — Evaluate the blind scenario separately

**Concern.** The intervention selection combines conditioned and blind results;
present the blind scenario separately.

**Response.** All main tables report blind and conditioned separately; no cell
pools them. The decision table is blind-only. The expected-value analysis that
pooled protocols is removed from the main claims and moved to an exploratory
appendix with the non-independence caveat.

**Action.** Tables 3-6; Section 7.4.

### R2-12 — Substitutability of calibration and control

**Concern.** Phrase the substitutability conclusion cautiously; it holds only
for the tested configurations.

**Response.** We remove the general substitutability claim. The revision says
that in these configurations the gate adds little once the prior is calibrated,
and that the gate pays mainly where always-on is harmful. It does not claim that
calibration and control are generally interchangeable.

**Action.** Section 7 and Section 10; the sentence is deleted from the abstract.

### R2-13 — Detailed ablation of EB, scaling and backoff

**Concern.** Present the individual contributions of empirical Bayes,
pool-relative scaling and backoff.

**Response.** Same one-factor dev ablation as R1-M2. On duplicate: legacy
+0.0370, tuned Laplace +0.0423, Laplace+pool +0.0273, EB absolute +0.0497,
EB+pool +0.0238, EB+pool Team +0.0132, EB+pool Backoff +0.0217. On strict:
−0.0083, −0.0023, −0.0042, −0.0008, −0.0002, −0.0002, −0.0052. The table
separates prior, scale, scope and fallback; the text states that the generated
panels confirm the finalist, not every cell.

**Action.** Section 7.1 and Table 4.

### R2-14 — Complete documentation for reproducibility

**Concern.** Document model versions, prompts, hyperparameters and
configurations; give clear reproduction instructions.

**Response.** Same as R1-M11: named models and dates, prompt hashes, κ, λ, π_s,
scope, top-k, pool size, allowed pairs, frozen run manifest and the three
reproduction levels. The repository adds a `REPRODUCE.md` with commands and
expected outputs.

**Action.** Section 3, Section 4, provenance table, `REPRODUCE.md`, `LICENSE`.

### R2-15 — Computational cost and latency

**Concern.** Analyse the cost and latency of feedback generation to show
feasibility.

**Response.** We add an end-to-end cost table from the frozen runs: 611–749
requests per panel and source, 356–617 seconds per run, 0.48–0.87 seconds per
request on CPU, with two requests per ticket for baseline and feedback. Feedback
database preparation is a separate one-off cost and is reported separately. The
calibrated lift adds no model call at inference; its cost is the offline
feedback construction.

**Action.** Section 9.3 and Table 7.

### R2-16 — Semantic similarity versus procedural correctness

**Concern.** Discuss why higher similarity does not guarantee better support
quality.

**Response.** The expert study provides the direct evidence: on the familiar
arm the expert prefers feedback (+0.147) while cosine penalises it (−0.088);
Spearman −0.48, AUC 0.22. The expert's preference is explained by length
(+173 characters, Spearman +0.63, matched effect 0), so neither measure is a
clean criterion. The revision discusses what each captures: cosine measures
similarity to one recorded reply; the expert measures procedure but is sensitive
to answer length; neither measures resolution success.

**Action.** Section 8, Figure 5, Section 10.

### R2-17 — Simplify and reorganise the abstract

**Concern.** State objective, methodology, main results and limitations;
present insignificant results without overestimating them.

**Response.** New ~200-word abstract with that structure; no "for free", no
"opposite sign" causal claim, no ceiling recovery. It states that the calibrated
gain is not confirmed.

**Action.** New abstract.

### R2-18 — Recent RAG evaluation, judge reliability, calibration literature

**Concern.** Supplement the literature review with recent work on RAG
evaluation, LLM evaluator reliability and feedback calibration.

**Response.** Same references as R1-m6: LLM-as-judge reliability and bias, RAG
evaluation suites, unbiased learning to rank, Bayesian/Wilson ranking. Each is
used in the related-work text, not appended as a list.

**Action.** Section 2; verified reference list.

### R2-19 — Dense figures and small labels

**Concern.** Improve readability and distinguish main from exploratory results.

**Response.** The figure plan keeps four main figures (data regime; generated
effects; expert evaluation; lift saturation) and moves exploratory figures to
the appendix. Vector PDF, fonts ≥8 pt at final width, at most four series per
panel, direct labels, colour-blind-safe palette, captions with regime, source,
sample size and interval method.

**Action.** Figures 1–5; appendix; `FIGURE_PLAN.md`.

### R2-20 — Conclusions must reflect the evidence

**Concern.** Reword the conclusions, especially the absence of a significant
positive benefit of the calibrated method in the blind scenario.

**Response.** The conclusion states: procedure overlap explains part of the
measured gain; calibration removes harm without a confirmed gain; similarity to
one recorded reply is not procedural correctness; the practical recipe is a set
of hypotheses, not a general law. No positive calibrated benefit is claimed.

**Action.** Section 11.

---

## Evidence index

| Artifact | Path |
|---|---|
| Split overlap audit | `revision_ieee_access/01_data_protocol/tables/r1m7_split_overlap.csv` |
| Generated effects | `.../tables/generated_deltas.csv` |
| Protocol contrasts | `.../tables/generated_protocol_contrast.csv` |
| Holm and MDE | `.../tables/power_multiplicity.csv` |
| Dev ablation | `.../tables/dev_ablation.csv` |
| Matched retrievers | `.../tables/matched_retrievers.csv` |
| Decision headroom | `.../tables/decision_headroom.csv` |
| Cost | `.../tables/generation_cost.csv` |
| Lift diagnostics | `.../LIFT_DIAGNOSTICS.md` and `.../tables/score_distribution.csv`, `vote_saturation.csv` |
| Human evaluation | `revision_ieee_access/02_human_study/HUMAN_EVAL_RESULTS.md` |
| Frozen runs | `results/revision/frozen/frozen_runs_v1.json` |

## Open items before this letter is final

1. **Second expert E2.** The first expert is complete; the second is in progress.
   Add percent agreement, κ and the combined estimate when the ratings arrive.
2. **Generated ablation.** The retrieval-level ablation and the generated
   finalist are reported; the eight generated factor cells are optional and would
   strengthen R1-M2 further. The manuscript states the current scope explicitly.
3. **Final title approval.** The working title is the one in the manuscript.
4. **Compile with the IEEE Access class.** The local check used a standard LaTeX
   class because the Formata fonts are not installed; the final compile should be
   done in Overleaf or with the IEEE Access class and fonts.
5. **DOI and volume.** The manuscript sets volume 14 and year 2026; the DOI remains
   a placeholder assigned by IEEE Access production.
6. **Author photos and biographies.** The biography text is in place; the photos
   must be added before submission.
