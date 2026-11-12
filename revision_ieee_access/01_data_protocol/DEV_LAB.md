# Dev lab — tuned Laplace, ablations, reply-similarity baseline

Answers R1-M3, R1-M2, R1-M5 and R2-13 on development data only, before any
generated-answer run. Script: `experiments/22_revision_dev_lab.py`. No API call.

## Selection rule (declared before looking at results)

- Primary metric: `proxy_top1` (retrieved-reply cosine to the reference reply),
  tie-break `proxy_mean5`.
- Every candidate cell is evaluated; the published legacy cell (m = 0.80,
  cap = 0.20, absolute, intersection) is included but not privileged.
- Each lift family receives the same 5 × 5 budget per scope.
- Selection uses dev only; selected cells are then evaluated once on eval
  (`dev_to_eval.csv`). This does not remove selection optimism, which is reported.

## 1. The published Laplace setting is nearly the worst cell (R1-M3)

On the strict panel, intersection scope, the legacy cell ranks **23rd of 25**
Laplace configurations by dev `proxy_top1`. Tuning m and cap moves it from
−0.0083 to −0.0023, and pool-relative scaling to −0.0042. All remain at or below
zero. The reviewer's claim that the baseline was untuned is correct, and tuning
does not rescue it on disjoint history.

## 2. The calibrated gain collapses with procedure overlap (R1-M2)

At the frozen finalist (empirical Bayes κ = 2, intersection, absolute):

| Panel | Conditioned dev | Blind dev | Reply-embed dev |
|---|---:|---:|---:|
| random | +0.043 | −0.056 | +0.041 |
| duplicate | +0.050 | −0.011 | +0.048 |
| prefix (strict) | −0.001 | −0.008 | +0.005 |

## 3. One-factor-at-a-time contrasts (conditioned, dev, top-1 proxy)

| Contrast | Duplicate panel | Strict panel |
|---|---:|---:|
| Laplace legacy (0.8 / 0.20) | +0.037 | −0.008 |
| Laplace tuned (dev) | +0.042 | −0.002 |
| Laplace + pool-std λ=1 | +0.027 | −0.004 |
| EB κ=2, absolute | +0.050 | −0.001 |
| EB κ=2 + pool-std λ=1 | +0.024 | +0.000 |
| EB κ=2 + pool-std, team scope | +0.013 | +0.000 |
| EB κ=2 + pool-std, backoff | +0.022 | −0.005 |

Conditioned, 95 % bootstrap CIs and Wilcoxon p-values are in
`tables/dev_ablation.csv`. Scaling reduces the gain on the duplicate panel rather
than rescuing it; the scope change costs about half the effect; backoff does not
recover it. The dominant factor is the data regime, not the prior.

## 4. A free reply-similarity baseline matches the LLM judge (R1-M5)

Same historical pairs, same aggregation and lift; only the score source changes.

| Panel | Conditioned LLM judge | Reply exact | Reply BGE embedding |
|---|---:|---:|---:|
| random | +0.043 | +0.019 | +0.041 |
| duplicate | +0.050 | +0.007 | +0.048 |
| prefix (strict) | −0.001 | +0.001 | +0.005 |

Under its dev-selected cell the BGE baseline reaches +0.020 on strict eval
(CI [+0.011, +0.031]) versus +0.014 for the judge (CI [+0.006, +0.023]). A placebo
check with a random other reference shows that most of the judge's gain is generic
(+0.009 of +0.014), while the reply-embedding source has a larger query-specific
component. Conclusion: the LLM judge does not add much over a cheap reply-similarity
prior, and on disjoint history it adds nothing measurable.

## 5. Frozen finalist and consequences

`configs/revision/finalists_v1.yaml`: `M4_intersection`, empirical Bayes κ = 2,
absolute scaling, applied identically to conditioned, blind and reply-embedding
evidence on three panels (random, duplicate, strict). This is the configuration
that goes into generated answers and then into the human study. It is deliberately
not the dev-selected per-panel cell, to avoid selecting the method on the strict
panel that is supposed to test it.

Expected new generation prompts: about 5,000 (three panels × three sources, with
shared baselines). Exact per-panel counts are in `tables/r1m7_generation_budget.csv`.

## Reproduction

```powershell
.\.venv\Scripts\python.exe experiments/22_revision_dev_lab.py --validate-eval
.\.venv\Scripts\python.exe -m pytest tests/test_reply_similarity.py -q
```
