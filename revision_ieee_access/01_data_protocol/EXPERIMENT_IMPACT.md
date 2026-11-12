# Effect on existing experiments

**No paid calls, new generated answers or changes to legacy artifacts are part of
package A. No final group split or GFT sample is selected here.**

Keep canonical IDs/dataset, legacy scores, indices and submitted result folders.
The new context references pinned inputs but does not migrate or rewrite them.
Revised artifacts go under `data/processed/revision/` and `results/revision/`;
these and `data/cache/` are ignored by Git because they may contain confidential
data. Public exports need a separate explicit review.

The planning audit counted 94,913 nonself train/train pairs per protocol for the
existing random split, and 53,620 for the existing disjoint split. The latter
covers only 490/886 train queries. These are reusable filtered subsets, not newly
completed top-k evidence and not counts for a future group split.

Package B must recompute aggregates and priors from allowed rows and build pools
inside allowed history. Old top-100 postfiltering is not a replacement. Cached
embeddings/individual scores/answers are reusable only when their relevant
content and model/request provenance agree; a warm cache alone is insufficient.
Changed retrieval context generally requires new answers, not editing old JSON.

Before any calls, package C reports retained evidence, pool coverage and exact
cache misses for approved configurations. Author approval is required for a new
judging/generation regime or paid work. The GFT main dataset is exported only
after final query IDs, history, dev-selected methods and generated outputs are
frozen with explicit run IDs. App development with synthetic data can proceed
independently.

## Measured on the revision data layer

History-only, both feedback endpoints, self-pairs removed, priors from the same
rows:

| Panel | History | Eval queries | Retained feedback rows |
|---|---:|---:|---:|
| legacy random | 878 | 398 | 94,913 |
| submitted disjoint | 886 | 390 | 53,620 |
| duplicate panel | 878 | 398 | 52,321 |
| prefix panel | 856 | 427 | 58,826 |

Every panel retrieves only from its history partition (asserted in the runner).
Retrieval-level effects and confidence intervals are in `R1M7_RESPONSE.md`.
Generated-answer effects are not measured yet.

## Generation budget (exact prompt-key counts, no calls made)

`tables/r1m7_generation_budget.csv` lists unique prompt keys and cache misses per
panel and configuration, computed with the frozen prompt template and the existing
generation cache. For the minimal generated panel (baseline + one calibrated
configuration per data regime, conditioned and blind), the exact count is **5,288
new prompts** across four regimes; dropping the superseded submitted-disjoint
regime gives **3,933**. Adding the ablation matrix multiplies this by the number of
configurations. These are counts, not a cost estimate and not an approval; a new
judging regime would add a separate budget.

