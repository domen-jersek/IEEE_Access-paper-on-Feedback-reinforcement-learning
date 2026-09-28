"""Generate all manuscript figures (F1-F12).

    python notebooks/paper_figures.py

Outputs vector PDF + 300-dpi PNG into `paper_ieee_access/paper_figures/`.
"""
from __future__ import annotations

from paper_figures_a import (fig1_judge, fig2_reversal, fig3_scope, fig4_retriever,
                             fig5_mechanism)
from paper_figures_b import (fig6_rescue, fig7_negatives, fig8_control, fig9_magnitude,
                             fig10_validation, fig11_calibrated_null)

FIGURES = [
    fig1_judge,
    fig2_reversal,
    fig3_scope,
    fig4_retriever,
    fig5_mechanism,
    fig6_rescue,
    fig7_negatives,
    fig8_control,
    fig9_magnitude,
    fig10_validation,
    fig11_calibrated_null,
]


if __name__ == "__main__":
    for figure in FIGURES:
        figure()
    print("done")