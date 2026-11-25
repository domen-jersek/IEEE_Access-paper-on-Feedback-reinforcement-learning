# Prior-work comparison — SIKDD 2025, SIKDD 2026, submitted and revised IEEE Access

Used for the “Relationship to prior work” subsection (R1-m7, R2-9) and the cover
letter disclosure. Values are taken from the audited artifacts, not from memory.

| Dimension | SIKDD 2025 | SIKDD 2026 | Submitted IEEE Access | Revised IEEE Access |
|---|---|---|---|---|
| Role | retrieval + generation pipeline | exploratory feedback-aware retrieval | protocol as experimental variable | data regime and measurement validity |
| Corpus | enterprise IT tickets | 1,596 tickets, 37 teams (different cleaning pass) | 1,595 tickets, 37 teams, 17 classes | same 1,595; canonical IDs unchanged |
| Evaluation queries | — | 250 leave-one-out | 878/319/398 train/dev/eval + 5 seeds; 390 disjoint eval | same splits for legacy; new history-only regimes: 398/398/427 eval |
| Judge | — | gpt-4o-mini, **conditioned only** | gpt-5.6-luna, conditioned **and blind** | same two protocols, restricted to history-only rows |
| Feedback pairs | — | 102,722 conditioned pairs | 175,600 pairs per protocol | 94,913 (random) / 52,321 (duplicate) / 58,826 (prefix) admissible pairs |
| Lift | — | Laplace α=β=1, m=0.80, cap=0.20; tanh/LCB fidelity | legacy Laplace as object of study; EB + pool scaling | tuned Laplace, EB, scaling, scope and backoff varied one at a time |
| Scopes | — | global, team, 5-class type, team∩type | global, team, 17-class, team∩class, backoff | same scopes; M-notation dropped |
| Procedure-disjoint split | no | no | claimed, 390 eval; **not actually disjoint** | versioned structural groups; duplicate and conservative prefix panels; 5 seeds |
| History-only retrieval and evidence | no | no | no | yes: index, BM25, feedback rows and priors restricted to train |
| Reply-similarity baseline | no | no | no | exact and BGE reply similarity on the same allowed pairs |
| Matched retriever comparison | no | no | different lift settings (confounded) | one configuration across dense, BM25, hybrid |
| Human evaluation | no | no | no | 105 blinded items, three arms; one expert complete, second pending |
| Statistical plan | — | bootstrap CIs; no always-on significance | dev-frozen thresholds, ticket-clustered bootstrap, MDE, 39 runs without multiplicity correction | pre-specified families, Holm, MDE, exact McNemar for human |
| Main finding | pipeline works | conditioned feedback helps weak baselines; simple gates fail; CIs cross zero | same reranking helps conditioned and harms blind | harm is overlap-driven; calibrated gain small and metric-dependent; expert and cosine diverge |

## What is genuinely new in the revision

1. **Procedure overlap as the explanatory variable.** The submitted “disjoint” split
   still shared an identical reply for 12.8 % and a reply opening for 53.3 % of eval
   tickets. The revision measures, removes and reports this.
2. **History-only evidence.** Retrieval, BM25 IDF, feedback rows and priors use the
   same training partition; the submitted pipeline did not.
3. **Controlled component isolation.** Tuned Laplace under a matched budget, EB,
   scaling, scope and backoff one at a time; matched retrievers.
4. **Free reply-similarity baseline.** Competitive with the LLM judge only when
   procedures overlap; not on strict history.
5. **Expert validity check.** No difference on new procedures; familiar-procedure
   preference is length-driven; expert and cosine metric diverge.
6. **Reproducibility.** Frozen runs with hash manifest, pinned model and prompt
   regime, tests, and a public provenance record.

## Disclosure paragraph for the cover letter

> This manuscript is an expanded version of two conference papers by the authors:
> [SIKDD 2025] introduced the retrieval and generation pipeline, and [SIKDD 2026]
> reported an exploratory, conditioned-only feedback analysis on 250 leave-one-out
> queries. The present revision re-uses the pipeline and the legacy lift formula as
> fixed components, but contributes a new history-only data regime, a controlled
> decomposition of the calibration repair, a free reply-similarity baseline, a
> matched retriever comparison, a statistical sensitivity analysis, and an
> independent expert evaluation. No text, tables or figures from the conference
> papers are reused; the corpus is the same enterprise ticket data. The connection is
> disclosed in Section II and in this letter.

## Relationship paragraph for the manuscript

> Two earlier papers by the authors are the starting point. [SIKDD 2025] introduced
> the retrieval and generation pipeline. [SIKDD 2026] added LLM-judge feedback as a
> retrieval modifier and reported an exploratory, conditioned-only analysis on 250
> leave-one-out queries, in which no always-on method was significant and simple
> retrieval-score gates failed. The present article re-uses the pipeline and the
> legacy lift formula as fixed components, but its contribution is different: it
> treats the data regime as the experimental variable, restricts retrieval, feedback
> and priors to an allowed history, isolates the components of the calibration
> repair, and adds an expert assessment of procedural correctness. No text, tables or
> figures are reused; the overlap is limited to the cited background, and the
> connection is disclosed here as required for expanded conference versions.
