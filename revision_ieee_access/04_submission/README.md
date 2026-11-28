# Submission package — revised IEEE Access manuscript

This folder contains a flat copy of the revised manuscript and its figures:

```
access_revision.tex   clean manuscript source
ieeeaccess.cls        IEEE Access class used by the submitted version
fig1_regime.tex       TikZ data-regime schematic
figures/
  fig2_generated.pdf  generated-answer effects
  fig3_matched.pdf    matched-retriever comparison
  fig4_saturation.pdf lift saturation
  fig5_expert.pdf     expert assessment
```

The submitted `access.tex` is not included and is not modified.

## Compile

Use the official IEEE Access LaTeX template and fonts (the class expects the
Formata fonts and `IEEEtran.cls`). The simplest route is the IEEE Access Overleaf
template: replace its `access.tex` with `access_revision.tex`, upload `figures/`
and `fig1_regime.tex`, and compile with pdfLaTeX. A vanilla TeX Live installation
may fail on the Formata fonts; that is a template-font issue, not a manuscript
error.

The manuscript uses `tikz` with `positioning`, `fit`, `backgrounds` and
`arrows.meta`; these are in a standard TeX installation.

## Before upload

1. Add the three author photos and restore the optional `[{\includegraphics...}]`
   argument in the `IEEEbiography` environments. The biography text is in place.
2. Replace the DOI placeholder `10.1109/ACCESS.2026.XXXXXXX` if the production
   office provides a real DOI.
3. Check the volume/year header (`VOLUME 14, 2026`).
4. Add the second expert's agreement results to Table 6 and Section 8 when they
   are available.
5. If the generated ablation for R1-M2 is run, add the generated cells to
   Section 7.1 and Table 4; the current text states the retrieval-level scope
   explicitly.

## Files outside this package

- `../03_manuscript/RESPONSE_TO_REVIEWERS_DRAFT.md` — response letter.
- `../03_manuscript/COVER_LETTER_DRAFT.md` — cover letter.
- `../03_manuscript/TABLES.tex` — standalone table source.
- `../02_human_study/` — human-evaluation protocol, results and second-rater plan.
- `../../REPRODUCE.md` — revision commands and artifact provenance.
