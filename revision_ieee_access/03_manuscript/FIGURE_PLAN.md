# Figure plan for the revision

The submitted paper has eleven figures. Some remain valid; several describe results
the revision no longer supports. The rule is: one message per figure, readable at
print size, main results separate from exploratory panels.

## Global style

- Vector PDF, fonts ≥ 8 pt at final column width; no figure below 7 pt.
- Maximum four series per panel; direct labels instead of legends where possible.
- Colour-blind-safe palette; no red/green pair as the only distinction.
- Panel titles short; units and sample sizes in the caption, not in the plot.
- Main figures: data regime, generated effects, expert evaluation. Exploratory
  figures go to the appendix.

## Proposed figures

| New | Content | Source | Replaces |
|---|---|---|---|
| **F1** | Data-regime schematic: corpus → structural groups → group-aware split → allowed history; what the retriever, feedback rows and priors may see | design; no data | submitted F1 (judge schematic), redrawn |
| **F2** | Generated effects: three regimes × three evidence sources, forest plot with 95 % intervals; protocol contrast as a second panel | `tables/generated_deltas.csv`, `tables/generated_protocol_contrast.csv` | submitted F2 (reversal) |
| **F3** | Expert evaluation: (a) correctness difference with intervals, (b) length-matched vs unmatched, (c) expert vs cosine scatter | human notebook / `results/alberto/` | none (new) |
| **F4** | Lift saturation: score distribution and share at the cap for n votes; scope base rates in a small inset | `LIFT_DIAGNOSTICS.md`, `tables/score_distribution.csv`, `vote_saturation.csv` | submitted F5 (mechanism), relabelled |
| **F5** | Matched retrievers: feedback gain under one configuration for dense, BM25, hybrid, absolute vs pool-std | `tables/matched_retrievers.csv` | submitted F4 (retriever) |
| **F6** | Calibrated null (appendix): effect with intervals for legacy and calibrated cells on the strict panel | `tables/generated_deltas.csv` | submitted F11, relabelled |

## What happens to the submitted figures

| Submitted | Decision |
|---|---|
| F1 judge/protocol | Redraw as the data-regime schematic (F1). |
| F2 reversal | Replace with the regime forest (F2); the legacy reversal moves to a table. |
| F3 scope | Keep in the appendix, relabelled; the scope ablation table is the main evidence. |
| F4 retriever | Replace with matched retrievers (F5); the submitted version confounded lift settings. |
| F5 mechanism | Keep, relabel as F4; add the measured score distribution. |
| F6 rescue | Appendix only; one example is not a result. |
| F7 negatives | Appendix; keep the negative-results map. |
| F8 control, F9 magnitude | Appendix or drop; the gate contribution is reduced to the oracle bound and the blind-only table. |
| F10 validation | Keep in the appendix; update numbers. |
| F11 calibrated null | Keep, relabel as F6. |

## Caption rules

- State the regime, the evidence source, the number of tickets, and the interval
  method in the caption.
- For the expert figure, state “one expert; second pending” in the caption, not only
  in the text.
- Do not put “n.s.” without the interval; the interval is the message.

## Generation

F2, F3, F4 and F5 are generated from the committed aggregate tables by
`experiments/36_revision_figures.py`; the vector PDFs are in
`revision_ieee_access/03_manuscript/figures/`. F1 is the TikZ schematic
`fig1_regime.tex` (requires `tikz`, `positioning`, `fit`, `backgrounds`,
`arrows.meta`). F6 remains an appendix panel and can be built from
`generated_deltas.csv` when the appendix is assembled.

The script does not call a model or the API. Set `REVISION_FIGURE_PNG=1` only
for local previews; the PDFs are the submission files. The submitted PGF figures
are in `paper_figures/`; keep the source files and do not reuse the submitted
versions where the claim changed.
