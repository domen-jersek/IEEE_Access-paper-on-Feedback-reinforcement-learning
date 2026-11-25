# SIKDD overlap audit and differentiation plan

Status: for review (document A of the journal-revision work order). **Updated
2026-11-25 for the post-review revision: see §6; the table in §1 describes the
submitted version.**
Sources audited:
- `notebooks/test_results_modular_looe/SIKDD_2026_paper_final.tex` (SIKDD 2026, 4 pp, 2,940 words)
- `notebooks/SIKDD-Automated First-Reply Generation ... Multi-Modal Response Synthesis.pdf`
  (SIKDD 2025, 4 pp; Jeršek, Kenda, Klančič, Frattini)
- Journal artifacts: `experiments/`, `results/`, `notebooks/companion_*.py`, `data/processed/`

IEEE Access policy (verified on the author pages): expanded versions of an author's own
referenced conference article are accepted **if similarity is below 35 %**; the manuscript must
describe the connection and the submission must disclose it. Plan: keep well under that budget by
(i) reusing no text, tables, or figures, (ii) compressing all shared background to cited
summaries, and (iii) adding an explicit "Relationship to prior work" subsection plus a
disclosure sentence for the cover letter.

## 1. What is already published (must cite, not re-claim)

| # | Item | SIKDD 2025 | SIKDD 2026 | Journal paper |
|---|------|-----------|------------|---------------|
| 1 | Domain, corpus | RAG for first replies | 1,596 tickets, 37 teams, 250 LOO queries | 1,595 tickets, 37 teams, 17 classes, 878/319/398 + 5 seeds + procedure-disjoint |
| 2 | Pipeline | MiniLM + FAISS + multi-modal synthesis | MiniLM-L6 + FAISS top-5 + luna generation | same pipeline, compressed to a cited summary |
| 3 | Lift formula | — | Laplace, α=β=1, m=0.80, C=0.20; fidelity vs tanh/LCB | same legacy formula as the object of study; mechanism + EB fix are new |
| 4 | Scopes | — | M1 global, M2 team, M3 type (5 labels), M4 team∩type | global, team, class (17), team∩class, backoff; **M-notation dropped** |
| 5 | Feedback DB | — | gpt-4o-mini judge, 250 queries × ~1,580 candidates, 102,722 pairs, **conditioned only** | gpt-5.6-luna judge, 878 × 200 = **175,600 pairs per protocol**, conditioned **and blind** |
| 6 | Evaluation design | — | leave-one-out, 250 folds, bootstrap CIs | train/dev/eval, dev-frozen thresholds, grouped CV, ticket-clustered bootstrap, MDE, paired Wilcoxon |
| 7 | Headline numbers | — | M4 +0.010 [−0.006, +0.029]; all CIs cross zero; oracle gate +0.026 | conditioned M4 +0.0151 (p=.057), blind −0.038, gated +0.0164; report as controlled extension |
| 8 | Oracle trend | — | feedback helps weak baselines (r=−0.41; weakest decile +0.21) | cited as prior observation; journal adds deployable prediction (pool-reshaping gate) |
| 9 | Gating | — | retrieval-score/cosine gates fail; in-sample pool gate +0.017 (circular); future work = learned gate | multivariate gate AUC 0.699; risk control, decomposition law, substitution; explicit contrast with the failed retrieval-score gate |
| 10 | Pool analysis | — | query diversity r=−0.38; procedural mismatch r=0.21; team/type deltas | not reused; cited only |
| 11 | Worked examples | — | R-163 (.ica rescue, +0.85), R-143 (disruption, −0.71) | R-377 (new, +0.540), kept to one appendix page; prior examples cited |
| 12 | Figures | pipeline diagram | architecture, tickets_replaced, oracle_retrieval, team_type_delta | 12 new figures, none reused; the pipeline figure is **not** repeated (F1 becomes a study-design matrix, see outline B) |

## 2. New in this paper (the differentiation list)

1. **Protocol as an experimental variable.** The blind (deployable) protocol; the conditioned
   protocol is re-cast from "the only protocol" (SIKDD) to an explicit upper bound. The sign of the
   effect is shown to be controlled by the judge's information.
2. **New feedback database at scale.** Full training split re-judged (878 × 200 = 175,600 pairs)
   with a stronger judge under both protocols; SIKDD's 102,722 conditioned-only pairs are prior art.
3. **Mechanism.** Zero-inflation plus the two-vote Laplace cap; pool saturation (40–50 % at cap);
   variance-ratio argument. SIKDD used the same formula but did not diagnose it.
4. **Free repair.** Empirical-Bayes centering + pool-relative scaling; offline blind grid turns
   positive; generated harm removed (residual below MDE).
5. **Control as risk management.** Multivariate pre-generation gate, ceiling recovery where harm is
   concentrated, the exact decomposition law, and calibration–control substitution.
6. **Decision layer.** Expected-value regression and per-ticket action selection beating sign
   classification.
7. **Validation battery.** Independent metrics, independent non-OpenAI judge, seeds,
   procedure-disjoint split, second generator family.
8. **Negative-results map.** Semantic filter, scope blend, hybrid transfer, sign multi-action —
   each with a mechanism.

## 3. Ready-to-use reconciliation sentences

**For the Introduction (positioning paragraph).**
> An earlier exploratory study of the same corpus [SIKDD26] showed that LLM-judge popularity and
> semantic similarity are orthogonal, that feedback helps weak baselines, and that simple
> retrieval-score gates fail; its judge saw the ground-truth reply, it evaluated 250 leave-one-out
> queries, and its confidence intervals crossed zero. This paper turns the protocol itself into the
> experimental variable. We re-judge the corpus at scale, twice, changing only what the judge sees
> — conditioned (with the historical resolution) and blind (ticket only) — and show that the sign
> of the effect is controlled by the judge's information. We then identify why naive lift harms,
> repair it without labels, and show that the remaining value of control is risk reduction rather
> than gain.

**For Related Work ("Relationship to prior work").**
> This work extends two of our earlier conference papers. [SIKDD25] introduced the retrieval and
> generation pipeline for first-reply generation; [SIKDD26] added LLM-judge feedback as a retrieval
> modifier and reported an exploratory, conditioned-only analysis on 250 leave-one-out queries in
> which no always-on method was significant. The present article re-uses the pipeline and the
> legacy lift formula as fixed components, but contributes a new judging run (two protocols,
> 175,600 pairs each), a controlled train/dev/eval evaluation, the mechanism of failure, the
> calibration repair, the control analysis, and the validation battery. No tables, figures, or text
> are reused; the corpus is the same enterprise ticket data.

**Gate reconciliation (for the control section).**
> [SIKDD26] tested gates that use retrieval scores and found them indistinguishable from always-on.
> Our gate is different in kind: its dominant feature family is *intervention shape* (how the
> candidate pool would be reshaped), and its use is different: it is a risk-control device, not a
> source of gain. It recovers 0.54–0.56 of the ceiling exactly where the uncalibrated blind prior
> concentrates harm, and it is neutral once calibration removes that harm.

**Corpus count (decision item, see §4).**
> The corpus contains 1,595 tickets (1,597 exported rows; two duplicate ticket references removed),
> the same enterprise corpus studied in [SIKDD26].

## 4. Open/decision items

1. **Corpus count.** Journal artifacts: source export 1,597 rows → dedup → **1,595** (manifest
   `data/processed/dataset_manifest.json`); splits 878+319+398 = 1,595. SIKDD 2026 states **1,596**
   from a different cleaning pass over 1,989 raw rows. Recommendation: report 1,595 and add the
   parenthetical above; do not claim 1,596, since every artifact and split in this paper uses
   1,595. (If exact unification is required, we would need the SIKDD corpus export to compare
   reference IDs; not available in this repository.)
2. **Intent classes.** SIKDD's M3 used 5 coarsened labels; the journal uses the 17-class intent
   taxonomy of the ticket metadata. State the difference explicitly so the scope comparison is not
   misread.
3. **Naming.** Drop M1–M5; use Global, Team, Class, Team∩Class, Backoff throughout (also removes the
   SIKDD notation clash).
4. **Disclosure.** Add the connection statement to the manuscript (Section II) and the cover letter,
   and cite both prior papers.

## 5. Citation entries to add

- D. Jeršek, K. Kenda, R. Klančič, M. Frattini, "Automated first-reply generation for IT support
  tickets using retrieval-augmented generation and multi-modal response synthesis," in Proc.
  Information Society (SIKDD), Ljubljana, Slovenia, 2025.
- D. Jeršek, B. Kažič, K. Kenda, "Exploring feedback-aware retrieval for IT support first-reply
  generation," in Proc. Information Society (SIKDD), Ljubljana, Slovenia, 2026.
## 6. Revision update (2026-11-25)

The IEEE Access manuscript is being revised after review. The revision moves the
contribution further away from SIKDD, so the overlap budget improves rather than
tightens. The table in §1 and the sentences in §3 describe the **submitted** version;
the text below supersedes them where they differ.

### New contributions that did not exist in either SIKDD paper

1. **Data regime as the experimental variable.** History-only dense index, BM25
   statistics, feedback rows and priors; structural groups; duplicate and conservative
   prefix panels; five group-aware splits.
2. **Audit of the submitted split.** The claimed procedure-disjoint split shared an
   identical reply for 12.8 % and a reply opening for 53.3 % of its evaluation
   tickets; the random split shared 14.8 % and 58.3 %.
3. **Controlled repair.** Tuned Laplace under a matched budget (published cell ranks
   23/25), empirical Bayes, pool scaling, scope and backoff varied one at a time.
4. **Free reply-similarity baseline.** Competitive with the LLM judge only when
   procedures overlap.
5. **Matched retriever comparison.** One feedback configuration across dense, BM25 and
   hybrid retrieval; the submitted retriever-strength claim is a scaling artefact.
6. **Statistical sensitivity.** Pre-specified families, Holm correction, MDE,
   zero-inflation explanation.
7. **Expert evaluation.** 105 blinded items in three arms; one expert complete, second
   pending; expert and cosine metric diverge, and the familiar-procedure preference is
   length-driven.
8. **Reproducibility.** Frozen runs with a hash manifest, pinned model and prompt
   regime, tests, public provenance records.

### Consequences for self-overlap

- Shared with SIKDD: the corpus, the retrieval/generation pipeline, the legacy lift
  formula and the general motivation. All are cited and compressed to background.
- The main claims of the revision (data regime, split audit, controlled repair,
  reply baseline, matched retrievers, expert evaluation) did not appear in SIKDD.
- No text, tables or figures are reused.
- The disclosure paragraph and the prior-work table are in
  `revision_ieee_access/03_manuscript/PRIOR_WORK_TABLE.md`.
- Run a similarity check only after the manuscript is rewritten; target well below
  the 35 % budget, and keep the disclosure in the cover letter.
