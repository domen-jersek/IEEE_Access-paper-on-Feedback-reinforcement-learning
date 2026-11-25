# Paper spine — revision of Access-2026-52910

**Status:** working document, 2026-11-25. Not the manuscript. It fixes the claim,
the structure and the evidence before the sections are rewritten. The submitted
`access.tex` is not modified.

## Title

Recommended:

> **When Feedback Gains Are Borrowed: Procedure Overlap, Calibration, and the Limits
> of Similarity-Based Evaluation in RAG for IT Support**

Shorter alternative:

> **Borrowed Gains: How Procedure Overlap Inflates Feedback Reranking in RAG for IT
> Support**

Why not “The Same Reranking, the Opposite Sign”: the submitted title promises a
causal effect of the judge. The revision shows the sign flip is confined to the
legacy uncalibrated lift and is largely explained by procedure overlap.

## Abstract (target, ~200 words)

> Feedback from earlier ticket–candidate pairs is widely used to rerank retrieval
> candidates in retrieval-augmented generation for IT support. We show that the
> measured benefit depends less on the judge than on whether evaluation tickets share
> procedures with the historical base. Using 1,595 enterprise tickets, we rebuild the
> evaluation so that retrieval, feedback and priors use only the training partition,
> and compare familiar-procedure, exact-duplicate-removed and strict
> procedure-disjoint regimes. Under strict disjointness, blind feedback no longer
> harms (−0.044 to +0.001) and the conditioned effect is small (+0.009),
> metric-dependent and not robust to multiplicity correction. A free reply-similarity
> baseline matches the LLM judge only when procedures overlap. A blinded expert
> assessment of 105 tickets finds no difference on new procedures; on familiar
> procedures the expert’s apparent preference for feedback is explained by answer
> length, and expert and cosine metric disagree. Gains reported on overlapping splits
> are therefore partly borrowed from procedure overlap; calibration removes harm
> without a confirmed gain; and similarity to one recorded reply is not procedural
> correctness.

## Research questions

- **RQ1 (data regime).** Does the feedback gain survive when evaluation tickets do
  not share a procedure or reply template with the historical base?
- **RQ2 (components).** Which part of the repair matters: prior, scale, scope, or
  backoff?
- **RQ3 (measurement).** Does expert procedural correctness agree with the automatic
  similarity metric, and what does each measure miss?
- **RQ4 (practice).** What should a team deploying feedback reranking do, and what
  should it not claim?

## Contributions

1. **Data regime as the experimental variable.** Structural groups (full reply or
   title+subcategory; a conservative prefix variant), group-aware splits, history-only
   retrieval for every retriever, feedback and priors restricted to the same allowed
   rows. The submitted “procedure-disjoint” split is shown not to be procedure-disjoint.
2. **A controlled decomposition of the repair.** Tuned Laplace under a matched search
   budget, empirical Bayes, pool-relative scaling, scope and backoff varied one at a
   time; matched retrievers under one feedback configuration. The retriever-strength
   conclusion is a scaling artefact.
3. **A calibrated outcome.** Harm is removed on disjoint history; the remaining
   conditioned effect is small, metric-dependent and not confirmed after multiplicity
   correction. A free reply-similarity prior is competitive only where procedures
   overlap.
4. **An expert validity check.** 105 blinded ratings from one GFT expert (second
   pending): no difference on new procedures; on familiar procedures the apparent
   preference for feedback is explained by reply length; expert and cosine metric
   diverge. This is direct evidence that similarity is not procedural correctness.

## Structure

| Section | Purpose | Main evidence |
|---|---|---|
| 1 Introduction | problem, findings, contributions | — |
| 2 Related work | positioning, prior-work table | SIKDD 2025/2026 comparison |
| 3 Data regimes | groups, splits, history-only, priors | Table 1 |
| 4 Method | lift, scopes, calibration, evaluation | equations, π_s |
| 5 The submitted result | legacy sign flip, documented | legacy Table 2 |
| 6 Where the gain comes from | main data-regime result | Table 2, Figure 2 |
| 7 Controlled repair | ablation, tuned Laplace, matched retrievers | Table 3 |
| 8 Expert evaluation | correctness, length, metric divergence | Table 4, Figure 3 |
| 9 Validation | independent metrics, seeds, power/MDE | tables |
| 10 Discussion | implications, limitations | — |
| 11 Conclusion | narrowed claim | — |
| Appendix | examples, recipes, reproduction | — |

## Tables and figures

- **Table 1 — data and overlap.** Submitted random, submitted disjoint, duplicate
  panel, prefix panel; full-reply / prefix / title-subcategory / union overlap.
- **Table 2 — generated effects.** Three regimes × three evidence sources, strict
  cosine effect with intervals; protocol contrast.
- **Table 3 — controlled repair.** Dev ablation: legacy Laplace, tuned Laplace,
  Laplace + pool scaling, EB absolute, EB + pool scaling, scope and backoff; plus
  matched retrievers.
- **Table 4 — expert evaluation.** Per arm: baseline/feedback strict correctness,
  interval, McNemar, discordant pairs; preferences; length-matched effect. Agreement
  column added when the second expert completes.
- **Table 5 — prior work.** Corpus, evaluation size, judge, protocols, methods,
  results (SIKDD 2025, SIKDD 2026, submitted, revised).
- **Figure 1 — data regime.** Schematic of allowed history, candidates, feedback rows
  and priors.
- **Figure 2 — generated effects.** Forest plot by regime and source.
- **Figure 3 — expert diagnostics.** Correctness intervals, length-matched effect,
  expert-vs-cosine scatter.
- **Figure 4 — lift saturation.** Score distribution and cap frequencies.

## Claims and evidence

| Claim | Status | Evidence | Reviewer |
|---|---|---|---|
| Same reranking flips sign because of the judge | **Reinterpreted**: only legacy Laplace; mostly overlap | Table 2, seed robustness | R1-M1, R2-5 |
| Legacy Laplace harms blind feedback | **Supported on overlapping data**, disappears on disjoint | generated effects | R1-M7 |
| EB + pool scaling repairs harm | **Supported for harm removal**; no confirmed gain | generated effects, ablation | R1-M2, R2-13 |
| Retriever strength reduces the gain | **Retracted**: scaling artefact | matched retrievers | R1-m4, R2-10 |
| 922 tickets in identical-reply groups | **Corrected**: 922 share an opening; 238 share a full reply | group audit | R1-M7, M5 |
| Conditioned is an unrealizable oracle | **Reinterpreted**: deployable for historical pairs; resolution-informed upper bound | data-flow section | R1-M5, R2-2 |
| Gate recovers ceiling; EV beats sign | **Reduced**: oracle bound and blind-only table; EV not re-estimated | decision headroom | R1-M9, R2-11/12 |
| Expert confirms harm | **Not supported** | human study | R1-M8, R2-1 |
| Expert confirms benefit | **Not supported**; length-driven on familiar | human study | R1-M8, R2-16 |
| Similarity to one reply measures correctness | **Rejected** | expert-vs-cosine divergence | R1-M8, R2-16 |

## Limitations to state

- One corpus, one organisation, one language.
- One expert so far; second pending; no agreement yet.
- 105 items, 35 per arm, changed pairs only; underpowered for small effects.
- Procedural correctness, not ticket resolution.
- Automatic metric compares to one recorded reply.
- Simulated LLM feedback, not real user feedback.
- Post-hoc routing labels; label-conditional results.
- Hosted-model drift; frozen regime and hashes.

## Relationship to SIKDD

The revised paper is further from SIKDD than the submitted version: SIKDD had a
conditioned-only judge, 250 leave-one-out queries and no blind protocol,
procedure-disjoint regime, reply baseline, matched retrievers, power analysis or
human evaluation. The revised contributions are the data regime and the measurement
critique. No text, tables or figures are reused; the overlap table and disclosure
sentence are updated. See `SIKDD_OVERLAP_AUDIT.md` and
`PRIOR_WORK_TABLE.md`.

## What remains before submission

1. Second expert and agreement.
2. Rewrite sections on this spine (drafts in `DRAFT_SECTIONS.md`).
3. References (three missing topics), figures, license, model identifiers.
4. Similarity check against SIKDD and final disclosure.
5. Response letter with the evidence map.
