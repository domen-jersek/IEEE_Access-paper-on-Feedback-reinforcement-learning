# Lift diagnostics — score distribution, saturation, scope priors

Recomputes the numbers behind the revised equations and the saturation argument on
the allowed history-only evidence of the strict panel. No API calls.

Reproduce:

```powershell
python experiments/34_lift_diagnostics.py --panel disjoint_prefix
```

Outputs: `results/revision/lift_diagnostics/<run>/`; aggregate copies in
`revision_ieee_access/01_data_protocol/tables/`.

## 1. Judge-score distribution (allowed pairs)

| Protocol | Rows | Mean | Median | 10th | 90th | ≤0.4 | ≥0.8 | Neutral | Exactly 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conditioned | 58,826 | 0.305 | 0.2 | 0.0 | 0.8 | **62.9 %** | **12.1 %** | 25.0 % | 20.0 % |
| blind | 58,826 | 0.338 | 0.4 | 0.0 | 0.7 | 57.3 % | 7.9 % | 34.9 % | 17.3 % |

The judge is zero-inflated and negative-skewed. One in five conditioned scores is
exactly 0.5.

## 2. The formula is symmetric; the data are not

Laplace lift with m = 0.80, cap = 0.20, α = β = 1:

| Positives | Negatives | Lift |
|---:|---:|---:|
| 1 | 0 | +0.067 |
| 0 | 1 | −0.067 |
| 2 | 0 | +0.200 |
| 0 | 2 | −0.200 |
| 5 | 0 | +0.200 |
| 0 | 5 | −0.200 |
| 1 | 1 | 0.000 |
| 2 | 2 | 0.000 |

Two positives reach the cap exactly as two negatives do. The observed asymmetry
comes from the vote distribution above: negative evidence is five times more
common than positive evidence.

## 3. How often the cap is actually reached

Resampling n votes with replacement from the empirical score distribution:

| Protocol | n votes | Negative cap | Positive cap | Mean lift |
|---|---:|---:|---:|---:|
| conditioned | 1 | 0.0 % | 0.0 % | −0.026 |
| conditioned | 2 | **13.2 %** | **0.15 %** | −0.077 |
| conditioned | 5 | 15.6 % | 0.0 % | −0.106 |
| conditioned | 10 | 15.4 % | 0.0 % | −0.125 |
| blind | 1 | 0.0 % | 0.0 % | −0.022 |
| blind | 2 | 9.1 % | 0.0 % | −0.064 |
| blind | 5 | 7.0 % | 0.0 % | −0.089 |
| blind | 10 | 6.7 % | 0.0 % | −0.107 |

At realistic evidence counts the negative cap is reached for one in eight
candidates, the positive cap almost never. The “two negative votes reach the cap”
sentence must be paired with the symmetric positive statement and with these
measured frequencies.

## 4. Lift scale relative to retrieval scores (strict panel, intersection)

| Protocol | Configuration | Candidates at cap | Mean lift std / score std |
|---|---|---:|---:|
| conditioned | Laplace legacy | 25.5 % | **1.31** |
| conditioned | Laplace EB κ=2 | 13.7 % | 1.22 |
| conditioned | EB κ=2 + pool-std λ=1 | 13.7 % | **0.27** |
| blind | Laplace legacy | 19.1 % | 0.95 |
| blind | Laplace EB κ=2 | 1.7 % | 1.04 |
| blind | EB κ=2 + pool-std λ=1 | 1.7 % | **0.23** |

On the strict panel the legacy lift standard deviation is 1.31× the retrieval-score
standard deviation, not the 1.9–2.2× reported for the submitted setup; the ratio is
panel-dependent and must be reported per cell. Pool-relative scaling reduces it to
about 0.27, which is the scale argument in R1-M3.

## 5. Scope base rates π_s (allowed evidence)

| Protocol | Scope | Scopes | Mean of means | Min | Max |
|---|---|---:|---:|---:|---:|
| conditioned | global | 1 | 0.305 | 0.305 | 0.305 |
| conditioned | class | 10 | 0.259 | 0.063 | 0.543 |
| conditioned | team | 32 | 0.125 | 0.000 | 0.518 |
| conditioned | intersection | 57 | 0.180 | 0.000 | 0.552 |
| blind | global | 1 | 0.338 | 0.338 | 0.338 |
| blind | class | 10 | 0.241 | 0.087 | 0.580 |
| blind | team | 32 | 0.149 | 0.006 | 0.685 |
| blind | intersection | 57 | 0.183 | 0.003 | 0.685 |

π_s is far from 0.5 in every scope, and team/intersection scopes are strongly
negative. Centring the prior at 0.5 is therefore a modelling error, not a neutral
default; EB’s per-scope π_s is estimated from the same allowed rows.

## 6. What this changes in the manuscript

- Eq. 1 and Eq. 2 must show the clamp explicitly and state that m, cap, κ, λ and
  π_s are separate parameters.
- The two-vote paragraph must state the symmetry first, then the empirical
  negative skew, with the measured cap frequencies.
- “Standard” for the Laplace construction needs a citation or must be dropped.
- Saturation percentages are measured, not exact; the wording in the paper should
  say so.
- The lift-to-score ratio is not a constant: report 1.31 for the legacy cell and
  0.27 for pooled scaling on the strict panel, and explain that the submitted
  1.9–2.2 refers to a different data regime.
