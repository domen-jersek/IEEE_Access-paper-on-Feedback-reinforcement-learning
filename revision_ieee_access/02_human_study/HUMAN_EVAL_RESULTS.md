# Human evaluation — first rater results

**Status: one evaluator (Alberto), 105 items, descriptive analysis.** The second
rater is pending. This document reports what the first expert saw; it does not
claim inter-rater reliability and it does not confirm either the harm or the
benefit of the automatic results.

- Package: `data/processed/revision/study/v1`, `package_id` matches the ratings.
- Items: 105, 35 per arm, answer block only, A/B blinded and balanced within arm.
- Ratings: `revision_ieee_access/02_human_study/evaluation_alberto/ratings.csv`
  (private; raw ratings are not committed).
- Analysis: `experiments/33_human_eval.py`;
  aggregates in `revision_ieee_access/02_human_study/results/alberto/`.
- Strict correctness = `correct` only; lenient = `correct` or `partially correct`.
  Ties, both-incorrect and cannot-judge are reported separately.

## 1. Paired correctness

| Arm | n | Baseline | Feedback | Δ [95 % CI] | McNemar p | Discordant pairs |
|---|---:|---:|---:|---|---:|---:|
| random / blind | 34 | 0.647 | 0.794 | **+0.147** [0.000, +0.294] | .25 | 7 |
| strict / blind | 33 | 0.545 | 0.515 | −0.030 [−0.121, +0.061] | 1.00 | 3 |
| strict / conditioned | 34 | 0.559 | 0.559 | 0.000 [−0.088, +0.088] | 1.00 | 2 |

Lenient correctness: random 0.676 → 0.794 (+0.118 [0.000, +0.265]); strict blind
0.576 → 0.545 (−0.030 [−0.091, 0.000]); strict conditioned 0.588 → 0.618
(+0.029 [−0.059, +0.147]).

With 2–7 discordant pairs, only an almost perfectly consistent advantage would be
detectable at 80 % power. The study is a validity check, not a confirmatory test;
the confidence intervals are the relevant statement.

## 2. Preferences

| Arm | Feedback better | Baseline better | Equal | Both incorrect | Cannot judge | Net [95 % CI] |
|---|---:|---:|---:|---:|---:|---|
| random / blind | 8 | 5 | 16 | 6 | 0 | +3 [−4, +10] |
| strict / blind | 2 | 4 | 15 | 13 | 1 | −2 [−7, +3] |
| strict / conditioned | 7 | 3 | 11 | 13 | 1 | +4 [−2, +10] |

Across 105 items: 42 equal, 32 both incorrect, 2 cannot judge. A raw A/B preference
would have hidden a third of the sample as “both wrong”; this is why the per-reply
correctness rating was collected.

## 3. Answer length explains the only positive signal

| Arm | Baseline chars | Feedback chars | Difference | Spearman(length, Δ) | Matched (≤100 chars) Δ | Unmatched Δ |
|---|---:|---:|---:|---:|---:|---:|
| random / blind | 523 | 696 | **+173** | **+0.63** (p = .0001) | **0.000** (n = 13) | +0.227 (n = 22) |
| strict / blind | 451 | 450 | −1 | +0.10 (p = .57) | +0.044 (n = 25) | −0.200 (n = 10) |
| strict / conditioned | 424 | 430 | +6 | −0.02 (p = .89) | 0.000 (n = 22) | 0.000 (n = 13) |

On the random arm the feedback replies are one third longer, and the expert's
preference tracks that difference. Among pairs whose lengths differ by at most 100
characters the effect is exactly zero. The apparent benefit on familiar procedures
is therefore a preference for longer, more detailed replies, not a demonstrated
procedural improvement.

## 4. Expert and automatic metric disagree

| Arm | Automatic Δ (same items) | Expert Δ | Sign agreement | Spearman | AUC for preference |
|---|---:|---:|---:|---:|---:|
| random / blind | **−0.088** | **+0.147** | 0.14 (n = 7) | **−0.48** (p = .004) | 0.22 (n = 12) |
| strict / blind | −0.026 | −0.030 | 0.33 (n = 3) | +0.04 | 0.67 (n = 5) |
| strict / conditioned | +0.022 | 0.000 | 0.50 (n = 2) | −0.04 | 0.05 (n = 10) |

On the random arm the two measures point in opposite directions: the cosine metric
penalises the longer replies, the expert prefers them. On the strict arms there is
too little variation to estimate a relationship. This is the concrete evidence for
R1-M8 and R2-16: similarity to one recorded reply is not procedural correctness,
and a human judgement can itself be sensitive to answer length.

## 5. Unsafe and missing steps

| Arm | Baseline unsafe | Feedback unsafe | Baseline missing | Feedback missing |
|---|---:|---:|---:|---:|
| random / blind | 5.7 % | 8.6 % | 0 % | 0 % |
| strict / blind | 2.9 % | 5.7 % | 2.9 % | 0 % |
| strict / conditioned | 8.6 % | 8.6 % | 0 % | 0 % |

Events are rare; these are descriptive counts, not evidence of a safety
difference. The flags are the part of the rubric that a similarity metric cannot
capture at all.

## 6. What this means for the paper

1. **No confirmed benefit or harm on the strict panels.** Both intervals include
   zero; the point estimates are within ±0.03. This matches the automatic effect,
   which was also small there.
2. **The random-arm preference is length-driven.** Raw +0.147 becomes 0.000 when
   lengths are comparable; 61 % of the variance in the expert delta is explained by
   the length difference. It should not be reported as evidence that blind feedback
   improves procedures.
3. **The automatic harm is not reproduced by the expert.** The expert does not
   confirm that blind feedback makes replies procedurally worse on familiar
   procedures; if anything, the expert prefers the longer feedback replies, and the
   cosine metric penalises them.
4. **Neither measure is a clean criterion.** The paper should state this explicitly:
   cosine measures similarity to one recorded reply; the expert measures procedure
   but is sensitive to length. A second rater and a length-controlled design are the
   way to separate these.
5. **The study is underpowered for small effects.** With 2–7 discordant pairs per
   arm, the minimum detectable advantage among discordant pairs is essentially
   100 %. Report intervals, not p-values as confirmation.

## 7. Reviewer mapping

| Comment | What this analysis provides | Caveat |
|---|---|---|
| R1-M8, R2-1 | 105 expert ratings of procedural correctness, per reply, with flags and preferences | one rater, no agreement yet |
| R2-16 | direct comparison of expert delta with cosine delta, including sign disagreement and AUC | strict arms too few non-zero deltas |
| R1-M6, R2-6, R2-7 | paired bootstrap intervals, exact McNemar, MDE, separate ties/both-bad | underpowered; reported as such |
| R2-11 | blind arms reported separately, never pooled with conditioned | — |
| R1-M1, R2-5, R2-20 | no claim of confirmed gain; the random-arm signal is length-confounded | — |
| R1-M11 | concrete failure/neutral cases exist in the comments and ratings | examples must be anonymised in the paper |

## 8. Second rater

The second rater adds what this analysis cannot provide: percent agreement and
Cohen's kappa (with bootstrap interval), weighted kappa for correctness, PABAK/AC1
as prevalence-robust sensitivity, and a combined per-item estimate. The same
pipeline runs unchanged:

```powershell
python experiments/33_human_eval.py --package data/processed/revision/study/v1 `
  --ratings rater1.csv --ratings rater2.csv
```

If the second rater shows the same length sensitivity, that is a property of the
task/rubric, not of one person. If not, the first rater's length effect is
individual and the strict-panel null becomes more credible.

## 9. Reproduction

```powershell
python experiments/33_human_eval.py --package data/processed/revision/study/v1 `
  --ratings revision_ieee_access/02_human_study/evaluation_alberto/ratings.csv --label alberto
python -m pytest tests/test_human_eval.py -q
```

The script verifies the frozen details hashes and the package identity before
analysis. It writes only aggregates; ticket text stays in the private package.
