# Combined analysis plan — E1 + E2

**Status:** waiting for Timothy's ratings. The package, sample and answers are
frozen; this plan changes nothing in the study. It fixes how the second rater is
analysed so the result is not chosen after seeing it.

## 1. Export and verify

1. Export Timothy's ratings from the app (admin CSV or SQLite query).
2. Confirm the file has one rater ID (`timothy`) and the same `package_id` as
   `packages/v1.json`.
3. Check counts: E1 105; E2 however many he completed. Do not top up or edit
   items.
4. Archive both raw files privately with the package manifest. Do not commit
   them.

## 2. Run the fixed pipeline

```powershell
python experiments/33_human_eval.py --package data/processed/revision/study/v1 `
  --ratings revision_ieee_access/02_human_study/evaluation_alberto/ratings.csv `
  --ratings <path>/timothy_ratings.csv --label combined
```

The script:

- orients A/B per item from the package key;
- reports per-rater strict and lenient correctness, preferences, flags and
  length;
- matches the common items and reports coverage (`n_common`, `n_rater_1`,
  `n_rater_2`);
- computes percent agreement, Cohen's κ with a bootstrap interval, weighted κ for
  correctness, and PABAK/AC1;
- computes the combined per-item estimate as the average of the two raters'
  differences on the common items, with a ticket-level bootstrap interval.

## 3. What to read, in order

1. **Coverage.** `agreement_agreement.csv`: `n_common`, `n_rater_1`,
   `n_rater_2`. If Timothy rated fewer than ~60 items, the combined estimate is
   dominated by E1's items; label it as such.
2. **Preference agreement.** Percent agreement and Cohen's κ for the five-way
   preference (A, B, equal, both incorrect, cannot judge). Report the raw
   agreement beside κ because the prevalence of "both incorrect" and "equal"
   inflates agreement.
3. **Correctness agreement.** Weighted κ for the ordinal correctness pair
   (correct / partial / incorrect / cannot judge), plus PABAK/AC1 as a
   prevalence-robust sensitivity.
4. **Combined effect.** `combined_effect.csv`: per arm, E1 delta, E2 delta,
   combined delta and 95 % CI on the common items.
5. **Length.** Compare E1 and E2 length correlations. If both show the same
   direction, the length sensitivity is a property of the task/rubric; if only E1
   does, it is individual and the strict-panel null is more credible.

## 4. Decision rules fixed in advance

- **Agreement is reported, not used to select items.** If κ is low, say so; do
  not drop discordant items or re-weight to improve the result.
- **No pooling of arms.** Familiar/blind, strict/blind and strict/conditioned stay
  separate. The combined estimate is within arm, not across arms.
- **No significance claim from the combined sample.** The study remains a
  validity check; intervals are the result. A non-significant combined effect is
  not evidence of absence.
- **If the strict combined intervals include zero and the point estimates are
  within ±0.05,** the text says the expert data are consistent with the small
  automatic effects and do not confirm harm or benefit.
- **If the familiar-arm length effect replicates,** the text says the apparent
  familiar-procedure preference is length-sensitive for both experts; it is not
  reported as a procedural benefit.
- **If E2 disagrees strongly on the strict arms,** report the disagreement and
  keep the automatic result separate; do not average it away.

## 5. Manuscript updates

- **Table 4:** add an agreement column or a second panel with `n_common`,
  preference agreement, κ [CI], weighted κ, PABAK/AC1. Keep E1 and E2 separate;
  the combined estimate goes in a note or a lower panel.
- **Abstract:** add one clause only if the pattern replicates; otherwise keep the
  current sentence and say the second expert is reported in the table.
- **Section 8:** add the coverage and agreement paragraph; state that the sample
  is underpowered and that the second rater may not cover all items.
- **Limitations:** update "one expert" to "two experts, partial overlap" only
  after the analysis.
- **Response letter:** replace the E2 placeholder with the actual agreement and
  combined numbers; keep the wording that the expert study is a validity check.

## 6. What would change the paper's claim

- A large, consistent strict-panel benefit in both experts would require softening
  "no confirmed gain" to "a small benefit that the automatic panels did not
  resolve". It would not turn the study into a confirmatory trial.
- A large, consistent strict-panel harm would require reporting expert-automatic
  disagreement explicitly; it would not rescue the submitted "opposite sign"
  claim.
- A strong length effect in both experts would support the measurement critique
  and should be stated in the abstract; it would not be a nuisance to hide.
