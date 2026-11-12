# Generated-answer results under the corrected data regimes

Model: `openai/gpt-5.6-luna`, temperature 0.0, system prompt unchanged, regime
`51fd9f4b37e900bb` (identical to the audited submitted run). Only the data
boundary and the evidence source change. Finalist: `M4_intersection`, empirical
Bayes κ = 2, absolute scaling (`configs/revision/finalists_v1.yaml`).

Metric: per-ticket `cosine(generated reply, recorded reference reply)` difference
against the baseline answer generated from the same history-only retrieval.
95 % paired bootstrap intervals (5,000 resamples); two-sided Wilcoxon.
**3,513 new prompts** were generated in total.

## Cosine delta by data regime and evidence source

| Panel | Conditioned (LLM judge) | Blind (LLM judge) | Reply-similarity (BGE) |
|---|---|---|---|
| random | +0.0105 [+0.0005, +0.0212], p=.029 | **−0.0444** [−0.0606, −0.0285], p=.0002 | +0.0143 [+0.0047, +0.0242], p=.009 |
| duplicate | +0.0149 [+0.0059, +0.0246], p=.002 | −0.0214 [−0.0367, −0.0073], p=.66 | +0.0081 [+0.0016, +0.0152], p=.09 |
| **strict prefix** | +0.0093 [+0.0026, +0.0169], p=.066 | +0.0008 [−0.0063, +0.0084], p=.86 | +0.0026 [−0.0039, +0.0097], p=.84 |

Direct protocol contrast (conditioned − blind, same tickets):

| Panel | Difference [95 % CI] | p |
|---|---|---|
| random | +0.0549 [+0.0389, +0.0710] | <1e−4 |
| duplicate | +0.0363 [+0.0217, +0.0517] | .002 |
| strict prefix | +0.0085 [+0.0016, +0.0158] | .032 |

ROUGE-L reproduces the direction but not the strict-panel significance:
conditioned +0.0112, +0.0087, +0.0034 (p = .47 on strict); blind −0.0539,
−0.0205, +0.0020.

## What this changes

1. **The harm is an overlap artefact.** Blind feedback costs −0.044 when evaluation
   tickets share procedures with history and +0.0008 when they do not. The
   protocol contrast shrinks six-fold, from +0.055 to +0.009, but stays positive.
2. **The conditioned benefit is small and survives strict disjointness.** On the
   strict panel it is +0.0093 with a cosine interval above zero, but the rank test
   is marginal (p = .066) and ROUGE-L is not significant. The honest statement is
   “small, metric-dependent, not robustly confirmed”, not “harm removed, no gain”
   and not “the gain disappears”.
3. **A free reply-similarity baseline matches the judge only when procedures
   overlap.** On random data it is as good or better (+0.0143 vs +0.0105); on
   strict data it gives +0.0026 vs +0.0093. The LLM judge therefore adds value
   exactly where procedure overlap does not carry the answer.
4. **The retrieval proxy and the generated answers agree on the size of the
   conditioned effect, not on its certainty.** At the same configuration the
   retrieval-level estimate was +0.010 (seed 42; +0.009 across five splits), and
   the generated effect is +0.0093. What collapses under strictness is the
   *legacy* Laplace benefit (+0.036 → +0.002), not the calibrated one. The proxy
   is useful for ranking configurations but cannot certify significance; the
   generated interval and the independent metric decide that.

## Zero inflation and the mean/median gap

On the strict panel 55 % of conditioned tickets have an unchanged delta, against
23 % on random. On the duplicate panel the blind mean is −0.0214 with an interval
excluding zero, yet Wilcoxon p = .66: the rank test sees no shift because the
large negative and positive changes cancel in ranks while most tickets are
unchanged. This is the explanation requested in R1-M6. The pre-specified statistic
is the mean paired delta with its bootstrap interval; medians and zero shares are
reported alongside it, not used to select a favourable test.

## Cache anomaly and sensitivity

On 2/398 (random) and 5/427 (strict) tickets the identical baseline prompt was
served from the generation cache by a later evidence source with a different
response. Prompt hashes match; the difference is hosted-model nondeterminism when
a cache read misses. Recomputing the headline effects without those tickets moves
every mean by at most 0.0004 (conditioned strict +0.0093 → +0.0097; blind random
−0.0444 → −0.0446). No conclusion depends on them. The affected tickets are
recorded here rather than silently repaired; the study arms use one evidence
source per panel, so the displayed baseline is that run's own answer.

## Freeze status

These runs are the frozen source for the human study (`experiments/24_export_study.py`
refuses mismatched query sets or differing baseline answers). Any change to
retrieval, evidence, model or configuration requires a new study version.
