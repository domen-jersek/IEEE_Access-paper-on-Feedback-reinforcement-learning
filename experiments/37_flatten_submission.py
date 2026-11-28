#!/usr/bin/env python3
"""37_flatten_submission.py — build a flat Overleaf/submission package.

Copies the assembled manuscript, the IEEE Access class, the TikZ figure and the
four generated PDFs into revision_ieee_access/04_submission/, rewriting the
figure paths so the folder can be uploaded as-is. No model or API call.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_ieee_access" / "04_submission"
FIGS = OUT / "figures"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    text = (ROOT / "access_revision.tex").read_text(encoding="utf-8")
    text = text.replace(
        "\\graphicspath{{paper_figures/}{./}{figures/}{revision_ieee_access/03_manuscript/figures/}}",
        "\\graphicspath{{figures/}{./}}",
    )
    text = text.replace(
        "\\input{revision_ieee_access/03_manuscript/fig1_regime.tex}",
        "\\input{fig1_regime.tex}",
    )
    mapping = {
        "revision_ieee_access/03_manuscript/figures/fig2_generated_effects.pdf": "figures/fig2_generated.pdf",
        "revision_ieee_access/03_manuscript/figures/fig5_matched_retrievers.pdf": "figures/fig3_matched.pdf",
        "revision_ieee_access/03_manuscript/figures/fig4_lift_saturation.pdf": "figures/fig4_saturation.pdf",
        "revision_ieee_access/03_manuscript/figures/fig3_expert_evaluation.pdf": "figures/fig5_expert.pdf",
    }
    for old, new in mapping.items():
        if old not in text:
            raise SystemExit(f"path not found in manuscript: {old}")
        text = text.replace(old, new)
    (OUT / "access_revision.tex").write_text(text, encoding="utf-8")

    shutil.copy2(ROOT / "ieeeaccess.cls", OUT / "ieeeaccess.cls")
    shutil.copy2(ROOT / "revision_ieee_access" / "03_manuscript" / "fig1_regime.tex",
                 OUT / "fig1_regime.tex")
    for src, dst in {
        "revision_ieee_access/03_manuscript/figures/fig2_generated_effects.pdf": FIGS / "fig2_generated.pdf",
        "revision_ieee_access/03_manuscript/figures/fig5_matched_retrievers.pdf": FIGS / "fig3_matched.pdf",
        "revision_ieee_access/03_manuscript/figures/fig4_lift_saturation.pdf": FIGS / "fig4_saturation.pdf",
        "revision_ieee_access/03_manuscript/figures/fig3_expert_evaluation.pdf": FIGS / "fig5_expert.pdf",
    }.items():
        shutil.copy2(ROOT / src, dst)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
