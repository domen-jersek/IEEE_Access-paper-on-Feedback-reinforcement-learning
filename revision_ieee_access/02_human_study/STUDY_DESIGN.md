# Human evaluation design (pre-registered)

**Status:** design fixed before sample selection and before any expert rating.
**First rater completed 105 items; second rater pending.** Results:
[`HUMAN_EVAL_RESULTS.md`](HUMAN_EVAL_RESULTS.md). The analysis plan below is fixed;
the second rater will be analysed with the same pipeline and then merged into
agreement and combined estimates.
**Version:** v1 (2026-11-20). Any change after the first rating requires a new
study version and a new expert evaluation.

## Questions

1. **HE-1 (benefit under new procedures).** Does conditioned feedback improve the
   procedural correctness of first replies when evaluation tickets do not share a
   procedure or reply template with history?
2. **HE-2 (harm under overlap).** Does blind feedback harm procedural correctness
   when procedures do overlap?
3. **HE-3 (metric alignment).** Is the automatic cosine difference aligned with the
   experts' correctness difference?

## Arms

Three arms spanning the observed automatic effects, fixed in advance:

| Arm | Panel | Evidence | Automatic Δ (cosine) | Role |
|---|---|---|---:|---|
| A | random | blind | −0.044 | harm expected |
| B | strict prefix | blind | +0.001 | neutral control |
| C | strict prefix | conditioned | +0.009 | small gain expected |

The neutral arm is included deliberately: without it, a difference between A and C
could be attributed to the data regime rather than to the evidence. Every arm
compares baseline against the feedback system with the frozen finalist
(`configs/revision/finalists_v1.yaml`).

## Sampling frame

- Units: evaluation tickets of the frozen runs (`frozen_runs_v1.json`).
- **Changed pairs only.** A ticket is eligible if the feedback answer differs from
  the baseline answer. Unchanged pairs are byte-identical; asking experts to rate
  them would add no information. The changed share of each arm is known from the
  frozen runs and is recorded in the manifest (A 87.7 %, B 49.1 %, C 45.5 %).
- **Excluded: tickets whose baseline answer differs across sources of the same
  panel** (hosted-model nondeterminism on a cache miss). Five strict-panel tickets
  are excluded from the study frame; they remain in the automated analysis, where a
  sensitivity check showed at most a 0.0004 shift.
- **Not selected by automatic delta sign or magnitude.** Eligibility depends only
  on whether the answer changed, never on whether the change was good or bad.
- **Stratified by resolving team**, proportional allocation with a deterministic
  hash order within stratum. If a team has fewer eligible tickets than its quota,
  the remainder is redistributed.
- **Sampled without replacement across arms.** No ticket appears in two arms of
  the main package, and the pilot package is disjoint from the main package.
- Target: **35 tickets per arm, 105 total**. Two independent experts rate the same
  items.
- **A/B is balanced within each arm**, not only globally.
- If only a one-hour session is possible, a separate **20-ticket pilot package**
  is drawn from arm A only and is disjoint from the main sample. Pilot ratings are
  analysed as a separate session and are reported as a pilot, not as the main
  study.
- Two private practice items (`practice.md` in the package) are used in the
  introductory call to agree the rubric. They are not study data.

## Estimands

Two numbers are reported, separately labelled:

1. **Conditional effect:** expert correctness difference between feedback and
   baseline among changed pairs. This is what the raters see.
2. **Population effect:** `changed_share × conditional effect`. Unchanged pairs
   have identical answers, so their correctness difference is zero by definition.
   The changed share comes from the frozen runs, not from the sample.

This avoids the bias of oversampling changed pairs without hiding it. Selecting
changed pairs and then reporting only the conditional effect would overstate the
deployment effect by roughly a factor of two on the strict panel.

## Power

The paired expert comparison is a McNemar test on discordant items. Required
discordant pairs at 80 % power, two-sided α = .05: about 196 for a 60/40 split,
87 for 65/35, 49 for 70/30. With 105 tickets and a plausible discordance of
30–40 %, the study can detect only large differences (roughly 70/30 or stronger).
It is a **validity check with descriptive inference**, not a confirmatory trial.
This is stated in the paper; no claim rests on a p-value below the design's
sensitivity.

## Blocks and rubric

- **Primary block: first replies.** Each ticket shows two replies in random A/B
  order. For each: correct / partially correct / incorrect / cannot judge; flags for
  a wrong or unsafe step and a missing required step; then a preference with
  equal, both incorrect and cannot judge.
- **Optional second block: retrieved tickets**, only if time allows and on a
  subset. It is reported separately and is not part of the primary estimand.
- Blinding: the producing system is hidden; item IDs are opaque; A/B assignment is
  balanced per block.

## Analysis plan

- Per arm: strict correctness rate (correct only) and lenient rate (correct or
  partially correct) for baseline and feedback; paired difference with a
  ticket-level bootstrap interval; McNemar test on discordant items, reported as
  descriptive.
- Preference shares: feedback better, baseline better, equal, both incorrect,
  cannot judge.
- Flags: unsafe-step and missing-step rates per side.
- Agreement: percent agreement and Cohen's kappa for preference and for the
  correctness pair; per-arm and per-block.
- Population extrapolation as defined above, with the changed share treated as
  fixed (it is measured, not estimated from the sample).
- HE-3: descriptive comparison of expert correctness delta with automatic cosine
  delta across arms; no causal claim.

## Freeze and provenance

The sample, A/B assignment, answers, retrieval contexts and manifest are frozen
before the first rating. The manifest records run directories, details hashes,
finalist hash, dataset hash and the assignment seed. The application refuses to
start if the package content changes. Any later change creates a new study version
and discards previous ratings.

The private package lives under `data/processed/revision/study/` (git-ignored).
A sanitized provenance record without ticket IDs or paths is committed under
`revision_ieee_access/02_human_study/packages/`.

## Limitations stated in advance

- Two experts from one organisation; no population inference.
- Changed-pair estimand; population values depend on the measured changed share.
- Answer block measures procedural correctness, not ticket resolution outcomes.
- The neutral arm has a small automatic effect; a null expert result there is
  consistent with both no effect and insufficient power.
