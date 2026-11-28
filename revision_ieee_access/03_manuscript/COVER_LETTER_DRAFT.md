# Cover letter — draft

**Manuscript:** Access-2026-52910
**Title (submitted):** The Same Reranking, the Opposite Sign: When Feedback Helps
Retrieval-Augmented Generation for IT Support and When It Harms
**Title (revised):** When Feedback Gains Are Borrowed: Procedure Overlap,
Calibration, and the Limits of Similarity-Based Evaluation in RAG for IT Support
**Journal:** IEEE Access
**Date:** [date]

Dear Editor,

We thank you and the reviewers for the careful assessment of our manuscript. We
have revised it substantially and resubmit it for consideration. The point-by-point
response, with the reviewer concern, our response and the action taken for every
comment, is uploaded as the Author's Response file. A highlighted PDF and a clean
manuscript are included.

The revision is a restructure, not a new study. The reviewers identified that the
submitted title claimed a causal effect of the judge, while the evidence supported
a narrower result: an uncalibrated popularity lift helped under a
resolution-informed judge and harmed under a ticket-only judge, and the calibrated
lift produced no confirmed gain under either. They also identified that the
"procedure-disjoint" split was not procedure-disjoint, that the repair changed
several factors at once, that the legacy baseline was untuned, and that the
automatic metric lacked an independent validity check.

We rebuilt the evaluation around those concerns. The data regime is now the
experimental variable. We define structural groups over identical full replies or
identical title and subcategory, and a conservative prefix variant; we assign
whole groups to train, dev and eval; and we restrict the candidate index, BM25
statistics, feedback rows and scope priors to the training partition. The
submitted "procedure-disjoint" split shares an identical reply for 12.8 % and a
reply opening for 53.3 % of its evaluation tickets with training; the new strict
panel shares neither. Under that control, blind harm disappears, the conditioned
effect is small and metric-dependent, and a reply-similarity baseline that uses no LLM judge is
competitive only where procedure overlap remains.

We also add the controlled one-factor ablation, the tuned Laplace baseline, the
matched feedback configuration across dense, BM25 and hybrid retrieval, direct
protocol contrasts with Holm correction and minimum detectable effects, and a
blinded expert assessment of procedural correctness on 105 changed pairs. The
expert study is reported as a validity check, not as confirmation: on new
procedures it finds no difference; on familiar procedures the apparent preference
for feedback is explained by answer length, and expert and cosine judgement
diverge. One expert has completed the sample; the second is in progress, and inter-rater
agreement will be added before publication.

The revised claims are narrower and, we believe, more useful. Procedure overlap
explains part of the gain reported on random splits; calibration removes harm
without a confirmed benefit; and similarity to one recorded reply is not
procedural correctness. The practical recipe is presented as a set of hypotheses
to test on other corpora, not as a general law. We state the limits of the single
enterprise corpus, the simulated feedback, the reference-based metric and the
expert sample explicitly.

The manuscript is an expanded version of two conference papers by the authors.
The relationship is disclosed in Section 2 and in the prior-work table: the
earlier papers introduced the pipeline and an exploratory conditioned-only
analysis; the present article contributes the data-regime control, the split
audit, the controlled repair, the reply-similarity baseline, the matched-retriever
comparison, the statistical sensitivity analysis and the expert evaluation. No
text, tables or figures are reused.

The code and aggregate tables are available in the public repository. The ticket
text, feedback pairs and generated answers remain confidential under the data
agreement; the manuscript states exactly what a third party can rerun and what
requires on-premises access. The code is released under MIT; documentation and aggregate tables under CC BY
4.0; the confidential data are not released. The expert evaluation used
anonymized tickets and stored only aggregate ratings; no ticket text or raw
ratings are released.

We hope the revision meets the concerns of both reviewers and the editor. We
would be glad to provide any additional material.

Sincerely,

[corresponding author]
[affiliation]
[email]

_on behalf of all authors_
