# Review response map — evidence and status

Compact map of the 43 comments to the artifact that answers each one. Full wording
is in `IEEE_ACCESS_KOMENTARJI_IN_POPRAVKI.md`; this file is the working checklist for
the response letter and the manuscript rewrite.

Status: **done** = evidence exists and is documented; **partial** = evidence exists
but a manuscript edit or a decision is pending; **planned** = not started.

## Reviewer 1 — major

| # | Concern | Evidence / action | Status |
|---|---|---|---|
| 1 | Title/abstract overclaim; calibrated gain is null | New title, abstract, RQ and contributions in `PAPER_SPINE.md`; calibrated results in `GENERATED_RESULTS.md` | partial (manuscript edit) |
| 2 | Repair not controlled; scope, prior, scaling, backoff change together | One-factor ablation in `dev_ablation.csv`; dev lab section | done |
| 3 | Legacy Laplace untuned | 25-cell sweep; published cell ranks 23/25; tuned cell still ≤0 on strict | done |
| 4 | Two-vote algebra mis-stated; formula is symmetric | `LIFT_DIAGNOSTICS.md`: symmetry table, score distribution, cap frequencies | done |
| 5 | Conditioned may be deployable; reply-similarity baseline missing | Data-flow section; exact and BGE reply baselines in `generated_deltas.csv` | done |
| 6 | Statistical support thin; tests inconsistent; no multiplicity; no pre-registration | Paired bootstrap, exact McNemar, Holm, MDE, zero-inflation | done |
| 7 | Procedure overlap; disjoint split should be primary | Structural groups, group-aware splits, history-only panels, generated panel | done |
| 8 | Metric and judges close to circular; need human check | Expert evaluation, 105 items, three arms; length and metric divergence | partial (second expert pending) |
| 9 | Gate/EV overclaimed; rows not independent; protocol choice unrealistic | Oracle-bound headroom, blind-only table; EV claims reduced/removed | partial (decide EV) |
| 10 | Single-corpus generalisation | Limitations; external diagnostic optional | planned (text) |
| 11 | Reproducibility and provenance; no license | `PROVENANCE_TABLE.md`; frozen runs; license decision | partial (license) |

## Reviewer 1 — minor

| # | Concern | Evidence / action | Status |
|---|---|---|---|
| m1 | Eq. 2 missing π_s, κ, λ; clamp unclear | `LIFT_DIAGNOSTICS.md`; revised equations draft | done |
| m2 | Abstract long and dense | New abstract in `PAPER_SPINE.md` | partial (manuscript edit) |
| m3 | Nearly equal blind means need mean absolute difference | `mean_abs_delta` in `generated_deltas.csv`; add one sentence | done |
| m4 | Retriever comparison confounded with lift settings | Matched retriever table; conclusion retracted | done |
| m5 | Citations do not support sentences; MiniLM missing | `REFERENCES_TO_ADD.md` | partial (verify) |
| m6 | Missing work on LLM judgments, unbiased LTR, Bayesian/Wilson ranking | `REFERENCES_TO_ADD.md` | partial (verify) |
| m7 | Comparison with [31]/[32] unclear | `PRIOR_WORK_TABLE.md` | done |
| m8 | Repetition across sections | Editorial pass | planned |
| m9 | Template residue, DOI, layout | Editorial pass | planned |
| m10 | Name generator and judge in Section III | `PROVENANCE_TABLE.md`; methods paragraph | done |
| m11 | One success example is not enough | Appendix plan: blind failure + neutral calibration; select cases privately | planned |
| m12 | “for free”, “exactly”, “standard” | Wording pass | planned |

## Reviewer 2

| # | Concern | Evidence / action | Status |
|---|---|---|---|
| 1 | Human evaluation of procedural correctness | Expert study; second expert pending | partial |
| 2 | Conditioned uses the same reference as the metric | Data-flow and metric-dependency paragraph; separate historical vs evaluation reference | done |
| 3 | Simulated feedback not real user feedback | Limitations; judge-as-simulation statement | planned (text) |
| 4 | Single organisational corpus | Limitations; no external claim | planned (text) |
| 5 | +0.0021 not significant; distinguish harm removal from benefit | Calibrated null; harm-removal vs gain separated | done |
| 6 | CIs, direct protocol tests, multiplicity | Power/multiplicity outputs; protocol contrast | done |
| 7 | MDE interpretation; no equivalence claim | MDE reported as sensitivity, not absence | done |
| 8 | Novelty vs reranking, EB, intervention selection | Contribution framing in `PAPER_SPINE.md` | partial (manuscript edit) |
| 9 | Systematic comparison with [31]/[32] | `PRIOR_WORK_TABLE.md` | done |
| 10 | Matched feedback configuration across retrievers | `matched_retrievers.csv` | done |
| 11 | Blind-only policy | `decision_headroom.csv`; blind arms separate | done |
| 12 | Calibration/control substitutability overstated | Soften to tested configurations | planned (text) |
| 13 | Ablation of EB, scaling, backoff | `dev_ablation.csv` | done |
| 14 | Model versions, prompts, hyperparameters, reproduction | `PROVENANCE_TABLE.md` | partial (license/refs) |
| 15 | Cost and latency | `generation_cost.csv` | done |
| 16 | Semantic similarity vs procedural correctness | Expert study; length and metric divergence | partial (E2) |
| 17 | Abstract structure and cautious claims | `PAPER_SPINE.md` | partial (manuscript edit) |
| 18 | Recent RAG evaluation, judge reliability, calibration literature | `REFERENCES_TO_ADD.md` | partial (verify) |
| 19 | Dense figures and small labels | Figure plan; redraw with larger labels | planned |
| 20 | Conclusions must reflect no confirmed benefit | Draft conclusion; no positive claim | partial (manuscript edit) |

## Author decisions still needed

1. Confirm the title and the narrowed claim.
2. Confirm the license (code MIT/Apache, docs CC BY, data closed).
3. Confirm the second expert and the analysis deadline.
4. Decide whether the expected-value analysis is re-estimated or moved to
   limitations. Recommendation: move to limitations; keep the oracle-bound table.
5. Verify the new references and their venues.
6. Select the two concrete examples for the appendix (privately; anonymise in the
   paper).
7. Approve the figure redraw.
