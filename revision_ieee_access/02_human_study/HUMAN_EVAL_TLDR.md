# Human evaluation — TL;DR

**Scope.** 105 tickets, three arms of 35, one GFT expert (Alberto Marchinelli),
blinded A/B, answer block only, changed pairs only (the feedback answer differs
from the baseline). **Status:** first rater complete; second rater pending.

## Headline

On **new procedures** the expert sees neither benefit nor harm. On **familiar
procedures** the expert preferred the feedback replies, but the preference is
explained by answer length, not by procedure. The automatic cosine metric and the
expert disagree in that arm.

## Numbers

| Arm | Baseline | Feedback | Δ [95 % CI] |
|---|---:|---:|---|
| random / blind | 0.647 | 0.794 | +0.147 [0.000, +0.294] |
| strict / blind | 0.545 | 0.515 | −0.030 [−0.121, +0.061] |
| strict / conditioned | 0.559 | 0.559 | 0.000 [−0.088, +0.088] |

Strict correctness = `correct` only. Intervals are paired bootstrap over tickets.

- **Length.** Feedback replies on the random arm are 173 characters longer; the
  expert delta tracks that difference (Spearman +0.63). Among pairs whose lengths
  differ by at most 100 characters the effect is **exactly zero**.
- **Metric divergence.** On the random arm the automatic cosine delta is −0.088
  while the expert delta is +0.147; Spearman −0.48, AUC 0.22. The metric penalises
  the longer replies that the expert prefers.
- **Preferences.** 42 of 105 items were equal, 32 both incorrect, 2 cannot judge.
  A raw A/B preference would have hidden a third of the sample as “both wrong”.

## What it means

1. The automatic “harm” is **not confirmed** by the expert.
2. The apparent benefit on familiar procedures is **length-driven**, not a
   demonstrated procedural improvement.
3. Neither cosine nor expert preference is a clean measure of procedural
   correctness.
4. With one rater and 2–7 discordant pairs per arm, this is a **descriptive
   validity check**, not a confirmatory test. A second rater is needed for
   agreement.

## One-line version for slides or email

> The expert confirms neither harm nor benefit on new procedures; on familiar
> procedures the apparent preference is explained by answer length; automatic
> similarity and expert judgement diverge.

## Where the details are

- `HUMAN_EVAL_RESULTS.md` — full analysis, reviewer mapping, limitations.
- `results/alberto/` — aggregate tables and figures (no ticket text).
- `STUDY_DESIGN.md` — pre-registered design.
- `../../01_data_protocol/LIFT_DIAGNOSTICS.md` — vote distribution and saturation.
- `../../01_data_protocol/MANUSCRIPT_PATCH.md` (B2, C2) — manuscript text and
  response to reviewers.

Raw ratings are private and not committed.
