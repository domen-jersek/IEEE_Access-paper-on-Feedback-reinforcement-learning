# R1-M7 — procedure overlap and the disjoint evaluation

Reviewer 1, major comment 7 (verbatim):

> Overlap between training and evaluation tickets in the random split. With 58
> percent of tickets in identical-reply groups and 264 in near-duplicate title
> groups, a random 878/319/398 split lets eval tickets share a procedure with
> training tickets. The procedure-disjoint split (390 eval tickets) is the right
> control, and I appreciate it. But the conditioned gain there is only +0.0152
> (p = .025) for Team-intersect-Class and +0.0103 for Team (BGE p = .058), and the
> calibrated blind result is +0.0051 (p = .16). These should be the primary
> results, not a robustness row, because they match a realistic deployment on new
> procedures.

**Status: retrieval-level evidence and the generated panel are complete; the human
study is pending. Generated runs used the identical model, temperature and system
prompt (regime `51fd9f4b37e900bb`).**

## 1. What the submitted split actually guaranteed

The submitted `combined_group_hash` is a hash of the *conjunction* of reply
prefix, title and subcategory. Requiring simultaneous equality is not the same as
keeping connected near-duplicates together. Measured on the canonical 1,595
tickets (train → eval):

| Split | Equal full reply | Equal reply opening (300 chars) | Equal title + subcategory | `duplicate_template_v1` components |
|---|---:|---:|---:|---:|
| Submitted random (398 eval) | 59 (14.8 %) | 232 (58.3 %) | 61 (15.3 %) | 116 (29.1 %) |
| Submitted “disjoint” (390 eval) | 50 (12.8 %) | 208 (53.3 %) | 38 (9.7 %) | 88 (22.6 %) |
| New duplicate panel (398 eval) | 0 | 193 (48.5 %) | 0 | 0 |
| New prefix panel (427 eval) | 0 | 0 | 0 | 0 |

Two conclusions follow. First, the reviewer’s 58 % figure is reproduced exactly:
it is the *reply-opening* definition on the submitted random split. Second, the
submitted “procedure-disjoint” split was not procedure-disjoint under any
component-wise definition; 88 of its 390 eval tickets share a structural group
with training.

The 922-ticket / 58 % figure therefore describes shared reply openings, not
identical full replies. Identical full replies cover 238 tickets (14.9 %), which
is why the paper must stop calling these “identical-reply groups”.

## 2. Corrected protocol

Two versioned structural panels, both with a train-only history:

- **duplicate panel** (`duplicate_template_v1`): connected components over equal
  normalized full reply OR equal normalized title + subcategory. Removes exact
  duplicates and identical title/subcategory pairs. It does *not* remove shared
  reply templates, and 48.5 % of eval tickets still share a reply opening with
  training. This is an intermediate control, not a procedure-disjoint claim.
- **prefix panel** (`prefix_template_v1`, **primary control**): the rule above
  plus equal reply opening, using both the submitted raw-cut definition and a
  normalized-cut definition. Overlap is zero for every family in the table.

Both panels restrict, identically for every retriever:

- dense candidate index rebuilt from frozen embeddings over the train partition only;
- BM25 IDF fitted on train documents only; hybrid fusion and cross-encoder
  re-ranking operate on that same restricted pool;
- feedback rows with **both** endpoints in train, self-pairs removed;
- scope priors estimated from exactly those retained rows;
- evaluation queries excluded from history by construction.

Routing labels are the stored resolving team and the reply-derived intent class,
held fixed across panels, so that the only manipulated factor is the data
boundary. This is a label-conditional panel, not a claim that those labels are
available at ticket arrival (R1-M5 / R2-2). Ticket IDs, group mappings and local
paths remain in the private local artifacts, not in this document.

Group sizes: the duplicate rule yields 1,218 components (largest 83); the prefix
rule yields 676 components (largest 299). Groups are assigned atomically; split
sizes are 878/319/398 (duplicate) and 856/312/427 (prefix). The prefix split
cannot hit 878/319/398 exactly without cutting the 299-ticket component, which we
refused to do. Parameters were not chosen using any outcome.

## 3. Retrieval-level results

Metrics are retrieval-level reply-similarity proxies (cosine between the retrieved
reply and the reference reply, `multi-qa-MiniLM-L6-cos-v1`), **not** generated
answer quality. Δ is the paired per-ticket difference against baseline retrieval
on the same history. Intervals are 95 % paired bootstrap (5,000 resamples);
p-values are two-sided Wilcoxon. The tables below show split seed 42; the
five-split summary follows them.

Conditioned feedback, top-1 proxy:

| Configuration | Duplicate panel Δ [95 % CI] | Prefix panel Δ [95 % CI] |
|---|---|---|
| M4 ∩, Laplace legacy | +0.036 [+0.021, +0.051], p=.001 | +0.004 [−0.002, +0.012], p=.37 |
| M4 ∩, Laplace EB κ=2 | +0.043 [+0.030, +0.058], p<1e−7 | +0.010 [+0.001, +0.020], p=.19 |
| M4 ∩, EB κ=2 + pool-std | +0.032 [+0.020, +0.044], p<1e−5 | +0.008 [+0.003, +0.014], p=.13 |
| M2 team, Laplace legacy | +0.021 [+0.005, +0.037], p=.002 | −0.031 [−0.046, −0.017], p<1e−3 |
| M2 team, EB κ=2 + pool-std | +0.023 [+0.012, +0.035], p<1e−5 | +0.004 [−0.002, +0.010], p=.95 |
| M5 backoff, EB κ=2 | +0.040 [+0.024, +0.056], p<1e−6 | −0.021 [−0.037, −0.005], p=.003 |

Single-split cells, especially backoff and team, must be read together with the
five-split table below; several of them change sign across splits.

Blind feedback, top-1 proxy:

| Configuration | Duplicate panel Δ [95 % CI] | Prefix panel Δ [95 % CI] |
|---|---|---|
| M4 ∩, Laplace legacy | −0.014 [−0.034, +0.006], p=.66 | −0.010 [−0.019, −0.002], p=.004 |
| M4 ∩, Laplace EB κ=2 | −0.015 [−0.033, +0.004], p=.56 | −0.003 [−0.013, +0.006], p=.16 |
| M4 ∩, EB κ=2 + pool-std | −0.007 [−0.020, +0.005], p=.30 | +0.002 [−0.003, +0.008], p=.99 |
| M2 team, Laplace legacy | +0.005 [−0.013, +0.022], p=.05 | −0.043 [−0.058, −0.029], p<1e−7 |
| M5 backoff, EB κ=2 | −0.013 [−0.031, +0.006], p=.82 | −0.017 [−0.031, −0.004], p=.011 |

The median paired Δ is exactly 0 in every cell (median CI [0, 0]) because
71–88 % of tickets keep an unchanged top-1 context. The per-ticket distribution
is zero-inflated, so the median is uninformative here; the mean paired Δ with its
bootstrap interval, plus the share of changed contexts (28.3 % for M4 legacy,
23.7 % for EB, 12.4 % for EB + pool-std on the prefix panel), is the statistic the
claims rest on. This also explains the mean/median gap noted in R1-M6: it is
zero-inflation, not a heavy tail.

The prefix panel is also *harder* at baseline (top-1 proxy 0.610 versus 0.721 on
the duplicate panel), so its smaller gain is not a ceiling effect: there is more
headroom, yet less benefit. That is consistent with the gain being carried by
retrieved tickets whose procedure or template already exists in history.

### Seed robustness (five group-aware splits)

Because a single disjoint split can be lucky, every revised panel was generated
for seeds 42, 123, 456, 789 and 1024 (`tables/r1m7_seed_robustness.csv`).
Conditioned, top-1 proxy:

| Configuration | Duplicate panel mean (range), sig. | Prefix panel mean (range), sig. |
|---|---|---|
| M4 ∩, Laplace legacy | +0.034 (+0.019, +0.043), 4/5 | **+0.002 (−0.006, +0.006), 0/5** |
| M4 ∩, Laplace EB κ=2 | +0.039 (+0.026, +0.049), 5/5 | +0.009 (+0.006, +0.011), 2/5 |
| M4 ∩, EB κ=2 + pool-std | +0.021 (+0.012, +0.032), 5/5 | +0.006 (+0.003, +0.008), 1/5 |
| M2 team, Laplace legacy | +0.013 (−0.012, +0.023), 4/5 | −0.007 (−0.031, +0.008), 1/5 |
| M5 backoff, EB κ=2 | +0.031 (+0.007, +0.040), 4/5 | +0.008 (−0.021, +0.021), 4/5 |

The legacy-lift benefit on the strict panel is a robust null (0/5 significant).
The empirical-Bayes point estimate is consistently small and positive but
significance is unstable (1–2 of 5 splits). Team-only and backoff change sign
across splits and must not be presented as stable effects.

Blind feedback over the same five splits (top-1 proxy, mean and significant
splits):

| Configuration | Duplicate panel | Prefix panel |
|---|---|---|
| M4 ∩, Laplace legacy | −0.019 (1/5) | −0.008 (4/5) |
| M4 ∩, Laplace EB κ=2 | −0.018 (0/5) | −0.003 (1/5) |
| M4 ∩, EB κ=2 + pool-std | −0.015 (1/5) | −0.0005 (0/5) |
| M2 team, Laplace legacy | −0.012 (0/5) | −0.018 (2/5) |
| M5 backoff, EB κ=2 | −0.020 (0/5) | +0.004 (2/5) |

Blind harm is small and its stability depends on the panel; the calibrated lift
is indistinguishable from zero on the strict panel (0/5). The global scope (M1)
is negative at every seed in all four panel/protocol combinations (conditioned
−0.008, blind −0.007 on the strict panel), so scope is a first-order factor, not
a nuisance.

## 4. Generated answers (frozen finalist, 3,513 new prompts)

The same finalist (empirical Bayes κ = 2, intersection, absolute) was generated
with the identical model, temperature and system prompt as the submitted runs.
Δ is the per-ticket cosine difference of the generated reply against baseline.

| Panel | Conditioned | Blind | Reply similarity |
|---|---|---|---|
| random | +0.0105 [+0.0005, +0.0212], p=.029 | −0.0444 [−0.0606, −0.0285], p=.0002 | +0.0143 [+0.0047, +0.0242], p=.009 |
| duplicate | +0.0149 [+0.0059, +0.0246], p=.002 | −0.0214 [−0.0367, −0.0073], p=.66 | +0.0081 [+0.0016, +0.0152], p=.09 |
| strict prefix | +0.0093 [+0.0026, +0.0169], p=.066 | +0.0008 [−0.0063, +0.0084], p=.86 | +0.0026 [−0.0039, +0.0097], p=.84 |

Direct protocol contrast (conditioned − blind): +0.0549 [+0.0389, +0.0710] on
random, +0.0363 [+0.0217, +0.0517] on duplicate, and +0.0085 [+0.0016, +0.0158]
on strict (p=.032).

Three corrections to the retrieval-only reading:

1. The **harm** is an overlap artefact: it disappears entirely on the strict panel.
2. The **legacy Laplace benefit** collapses, but the **calibrated** conditioned
   effect stays small and positive (+0.0093). Its interval excludes zero while the
   rank test is marginal and ROUGE-L is not significant, so it is a small,
   metric-dependent effect, not a confirmed gain.
3. The reply-similarity baseline matches the judge only when procedures overlap
   (+0.0143 versus +0.0105); on strict history the judge is better
   (+0.0093 versus +0.0026).

## 5. Interpretation

1. The reviewer is correct: the submitted disjoint split was not procedure-disjoint.
   Moving it into the main table without re-deriving it would have repeated the error.
2. Removing exact reply duplicates and identical title/subcategory pairs leaves a
   large conditioned retrieval gain (+0.034 to +0.039 across five splits,
   significant in 4–5 of 5). Removing shared reply openings as well removes almost
   all of the **legacy** lift: +0.002, significant in 0/5 splits. The calibrated
   lift keeps a small positive effect (+0.009 in 2/5 retrieval splits; generated
   answers +0.0093 [0.0026, 0.0169], rank-test p=.066, ROUGE-L not significant).
   The benefit is therefore concentrated in tickets whose procedure or reply
   template is already present in history; what remains outside that regime is
   small and metric-dependent.
3. Blind harm at the intersection scope is small at retrieval level and disappears
   in generated answers: −0.0444 on the random panel, −0.0214 on the duplicate
   panel, +0.0008 on the strict panel (p=.86). The protocol contrast shrinks from
   +0.0549 to +0.0085 but stays positive (p=.032). “Harm removed, conditional
   benefit small” survives the stricter control.
4. The generated answers use the identical model, temperature and system prompt as
   the submitted runs; only the data boundary and the evidence source change. They
   do not establish procedural correctness, which is what the blinded expert study
   is for.

## 6. What remains before this is a paper-ready answer

- Human evaluation with two GFT experts on the frozen outputs
  (`revision_ieee_access/02_human_study/`), then the manuscript rewrite and the
  point-by-point response.

## 7. Reproduction

```powershell
.\.venv\Scripts\python.exe experiments/20_build_revision_data.py
.\.venv\Scripts\python.exe experiments/21_revision_retrieval_panels.py --protocol conditioned
.\.venv\Scripts\python.exe experiments/21_revision_retrieval_panels.py --protocol blind
# Seed robustness: repeat with --split-seed 123|456|789|1024 and
# --panels disjoint_duplicate disjoint_prefix.
.\.venv\Scripts\python.exe -m pytest tests/test_data_protocol.py tests/test_revision_boundaries.py -q
```

Inputs are pinned by SHA-256 in the run manifests; no model or API call is made.
Aggregate tables for the manuscript: `tables/r1m7_split_overlap.csv`,
`tables/r1m7_retrieval_panels.csv`, `tables/r1m7_seed_robustness.csv` and
`tables/r1m7_generation_budget.csv`.
