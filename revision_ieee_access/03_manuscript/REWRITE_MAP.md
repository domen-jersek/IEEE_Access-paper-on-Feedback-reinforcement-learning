# Rewrite map for `access.tex`

Working document, not the manuscript. It maps every submitted section, table and
figure to the revised structure. Line numbers refer to `access.tex` at commit
`07dec06`; they will shift after the first edit. The revised structure is the one
in `PAPER_SPINE.md`.

**Principle.** The pipeline, the lift definitions (corrected), the corpus and the
legacy results stay. The claim, the data regime, the main tables and the
measurement section change. Nothing is deleted merely because it is negative; it
is moved to the appendix if it still informs the boundary of the method.

## Section map

| Submitted | Lines | Action | Revised location |
|---|---:|---|---|
| Title | 58 | Replace | New title: procedure overlap, calibration, limits of similarity |
| Abstract | 83–85 | Replace | ~200-word abstract, harm removed / no confirmed gain |
| 1 Introduction | 97–116 | Rewrite | 1 Introduction: overlap, strict control, four contributions |
| 2 Related Work | 118–132 | Keep 2.1–2.4; rewrite 2.5–2.6; add 2.7 | 2 Related work |
| 2.1 RAG and reranking | 120 | Keep, update citations | 2.1 |
| 2.2 Learning to rank from feedback | 122 | Keep, add unbiased LTR | 2.2 |
| 2.3 LLM-as-judge validity | 124 | Keep, add judge-bias work | 2.3 |
| 2.4 Calibration/selective prediction | 126 | Keep, add Bayesian/Wilson ranking | 2.4 |
| 2.5 Relationship to prior work | 128 | Replace | 2.5 + Table 5 (prior-work comparison) |
| 2.6 Gap | 130 | Rewrite | 2.6: data regime and measurement, not another reranker |
| — | — | New | 2.7: RAG evaluation and procedural correctness |
| 3 Preliminaries | 134–180 | Rewrite in parts | 3 Data regimes and structural disjointness |
| 3.1 Corpus and splits | 136 | Keep corpus; rewrite splits | 3.1 structural groups; 3.2 group-aware splits |
| 3.2 Feedback protocols and judges | 138–170 | Rewrite data flow | 3.3 allowed history; 3.4 protocols and what is deployable |
| 3.3 Generation regime and provenance | 172 | Keep, add model names/dates | 3.5 generation regime and provenance |
| 3.4 Statistical protocol | 174 | Rewrite | 3.6 estimand, intervals, families, MDE, pre-spec caveat |
| 3.5 Data availability and ethics | 176–180 | Keep, update license/artifacts | 3.7 |
| 4 Method | 182–213 | Rewrite equations, reduce control | 4 Method |
| 4.1 Retrieval and retriever ladder | 188 | Keep, remove retriever-strength conclusion | 4.1 |
| 4.2 Lift | 190–201 | Rewrite with clamp, π_s, κ, λ | 4.2 + Eqs. (1)–(3) |
| 4.3 Scopes | 203 | Keep, rename M-notation | 4.3 |
| 4.4 Control and decisions | 205 | Reduce | 4.4 gate as risk control; oracle bound |
| 4.5 Evaluation | 207–213 | Keep metrics; add expert and reply baseline | 4.5 |
| 5 Results I | 215–292 | Reframe | 5 The submitted result and its scope |
| 5.1 Protocol reversal | 217–245 | Keep numbers, add split audit | 5.1 legacy reversal; 5.2 what the split allowed |
| 5.2 Scopes | 247–256 | Move to appendix | Appendix: scope grid |
| 5.3 Retriever dependence | 258–268 | Replace with matched comparison | 7.2 / Table 3B |
| 5.4 Formula ordering | 270 | Move to appendix | Appendix: formula ordering |
| 5.5 Reconciliation with prior work | 272–292 | Rewrite | 5.3 or 2.5 |
| 6 Results II | 294–350 | Split | 6 Where the gain comes from; 7 Controlled repair |
| 6.1 Two-vote algebra | 296 | Rewrite symmetry | 4.2 + 6.1 mechanism |
| 6.2 Pool saturation | 298–312 | Keep, add measured distribution | 6.1 / Figure 4 |
| 6.3 The repair | 314–344 | Replace with ablation | 7.1 / Table 3A |
| 6.4 Running ticket | 346–350 | Move to appendix | Appendix example |
| 7 Results III | 352–416 | Reduce | 7.3 Control as risk control |
| 7.1 What the gate removes | 354–389 | Keep oracle bound; drop recovery claims | 7.3 + appendix table |
| 7.2 Calibration/control substitutes | 391 | Delete general claim | one cautious sentence in 10 |
| 7.3 Expected values | 393–416 | Move to appendix / limitations | Appendix: exploratory EV, non-independent rows |
| 8 Results IV | 418–476 | Rewrite | 8 Expert evaluation; 9 Validation |
| 8.1 Independent metrics/judge | 420–437 | Keep, relabel | 9.1 |
| 8.2 Robustness | 439 | Keep, update | 9.2 seeds, splits, power |
| 8.3 Negative results map | 441 | Move to appendix | Appendix |
| 8.4 Stability summary | 443–476 | Rewrite | 9.3 |
| 9 Discussion | 478–495 | Rewrite | 10 Discussion and limitations |
| 9.1 Practical recipe | 480 | Rewrite as hypotheses | 10.1 |
| 9.2 Implications | 483 | Rewrite | 10.2 |
| 9.3 What we do not claim | 485 | Keep, update | 10.3 |
| 9.4 Threats | 487–495 | Keep, add measurement/expert | 10.4 |
| 10 Conclusion | 496–499 | Rewrite | 11 Conclusion |
| Appendix | 501–515 | Keep/update, add examples | Appendix A–D |

## Table map

| Submitted | Action | Revised |
|---|---|---|
| T1 data/protocols | Merge corpus/protocol facts into text; replace table with data/overlap | **Table 1** data regimes and overlap |
| T2 generated per scope | Replace; keep legacy per-scope numbers in a compact Section 5 table | **Table 2** generated effects by regime and source; legacy table in Section 5 |
| T3 calibration rescue | Replace with one-factor ablation | **Table 3A** dev ablation; **Table 3B** matched retrievers |
| T4 gate policy | Move to appendix; keep oracle bound/blind-only | Appendix table; one row in Section 7.3 |
| T5 validation | Merge into Table 2/3B and Section 9; appendix for the rest | Appendix |
| T6 negative results | Move to appendix | Appendix |
| — | New | **Table 4** expert evaluation |
| — | New | **Table 5** prior-work comparison |
| — | New | **Table 6** cost and latency (appendix) |

The LaTeX source for the new tables is `TABLES.tex`; the numbers come from
`revision_ieee_access/01_data_protocol/tables/*.csv`.

## Figure map

| Submitted | Action | Revised |
|---|---|---|
| F1 judge/protocol | Replace | **F1** data-regime schematic (`fig1_regime.tex`) |
| F2 reversal | Replace | **F2** generated effects and protocol contrast (`fig2_generated_effects.pdf`) |
| F3 scope | Move to appendix | Appendix |
| F4 retriever | Replace | **F5** matched retrievers (`fig5_matched_retrievers.pdf`) |
| F5 mechanism | Replace/relabel | **F4** lift saturation (`fig4_lift_saturation.pdf`) |
| F6 rescue | Move to appendix | Appendix |
| F7 negatives | Move to appendix | Appendix |
| F8 control | Move to appendix or drop | Appendix |
| F9 magnitude | Move to appendix or drop | Appendix |
| F10 validation | Keep in appendix, update | Appendix |
| F11 calibrated null | Keep in appendix, or merge into F2 | Appendix |
| — | New | **F3** expert evaluation (`fig3_expert_evaluation.pdf`) |

## Text edits that must not be forgotten

1. **922 correction.** Replace “88 identical-reply groups cover 922 tickets” with
   “922 tickets share a 300-character reply opening; 238 share an identical full
   reply.” The 922 figure is a prefix group, not a full-reply group.
2. **Disjoint split correction.** The submitted combined hash did not guarantee
   procedure disjointness: 12.8 % of its eval tickets share a full reply and
   53.3 % share an opening with training. Use “structurally disjoint” for the new
   panels; keep “submitted procedure-disjoint split” in quotes when referring to
   the old one.
3. **Conditioned protocol.** Remove “unrealizable oracle”. Use
   “resolution-informed judge”; state that it is deployable for historical pairs
   and that a new ticket has no resolution at inference.
4. **Two-vote algebra.** State the antisymmetry explicitly; pair the negative-cap
   sentence with the positive-cap sentence; attribute the observed asymmetry to
   the score distribution.
5. **Legacy baseline.** Remove “standard popularity lift”; call it the inherited
   configuration of the submitted study. Report the 25-cell sweep and the rank.
6. **Repair attribution.** Do not credit “calibration” for the joint change;
   report the one-factor ablation and the scope/scale contributions.
7. **Retriever strength.** Withdraw the claim; report the matched configuration.
8. **Gate.** Rename Eq. 3 a bookkeeping identity; no “exact decomposition law”;
   no ceiling-recovery claim; gate is risk control only.
9. **Expected values.** Remove from abstract/conclusion; appendix or limitations;
   state the two-rows-per-ticket non-independence.
10. **MDE.** Never say “no effect”; say “not confirmed / below the design's
    sensitivity”.
11. **Human study.** One expert complete, second pending; 105 items, 101 paired;
    2–7 discordant; length-driven familiar-arm preference; no agreement yet.
12. **Wording.** Remove “for free”, “exactly”, “standard”, “oracle” where the
    revision no longer supports them.
13. **Model names.** Name generator and judge at the start of Section 3, with API
    identifiers, temperature, prompt hash and dates.
14. **License/artifacts.** Add the license decision and the three reproduction
    levels; keep confidential data closed.

## Consistency checks before compiling

- Every number in the abstract appears in Table 2 or 4 with the same sign and
  rounding.
- Every claim in the conclusion is supported by a table/figure and is phrased as
  conditional on the corpus.
- “Conditioned” and “blind” are never pooled in a main table.
- The strict panel is labelled primary; the random panel is labelled familiar;
  the duplicate panel is intermediate.
- The 95 % intervals and the test statistic are named in every caption.
- The prior-work table and the disclosure paragraph agree on corpus size, judge,
  pairs and evaluation size.
- The response letter's section/table references match the compiled manuscript.
