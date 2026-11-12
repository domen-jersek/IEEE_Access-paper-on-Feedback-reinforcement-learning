# Implementation status and next increments

## A — Shared data contract (implemented)

`DataProtocolConfig`, `ResolvedDataContext`, strict config loading, pinned local
artifacts, immutable query/ID views, all-partition overlap validation, and an
auditable feedback-row selector. References are not loaded into the context.
The selector rejects invalid scores/IDs and removes both target roles for train
LOO. Legacy `EvalConfig` serialization and command behavior are unchanged.

The new preflight imports no encoders, API clients or experiment runners. Its
status is deliberately `data_boundaries_validated_not_experiment_ready`.

## B1 — Revision data layer (implemented)

`src/data/groups_v2.py`, `src/data/split_v2.py`, `src/data/audit.py`,
`src/retrieval/history.py`, `experiments/20_build_revision_data.py` and
`experiments/21_revision_retrieval_panels.py`.

- Structural groups are versioned and keyed by ticket ID; empty signatures never
  link tickets; the prefix rule is a superset of the duplicate rule.
- Group-aware splits assign whole groups, are deterministic per seed, and were
  validated for group atomicity and exact coverage over 5 seeds.
- History-only retrieval rebuilds dense indices from frozen vectors, fits BM25 on
  history documents, and restricts hybrid/CE branches to the same pool.
- `FeedbackBundle` estimates priors from exactly the supplied rows;
  `minus_query_bundle()` removes the target in both roles before recomputing
  priors.
- Cross-encoder cache keys are content-based (stable ticket ID + document hash).

Verified on the real corpus: the submitted random split lets 58.3 % of eval
tickets share a reply opening with train (reproducing the reviewer), and the
submitted "disjoint" split still shares an identical reply for 12.8 % and a reply
opening for 53.3 %. The corrected duplicate panel reaches 0 / 0 for reply and
title signatures; the prefix panel reaches 0 for all reported families.

## B2 — Wire the normal entry points (next; not implemented)

1. `experiments/03_build_feedback.py`, `04_evaluate.py`: explicit
   `--data-protocol`, reject revision execution until adapted, reject scope
   routing when the metadata policy does not expose the required labels, and
   record the resolved context/history in the run manifest.
2. `evaluation/runner.py`, `experiments/utils.py`: one run ID across
   manifest/details/summary; write-once run records; protocol violations stop the
   run instead of becoming silently skipped tickets; enforce cache-only operation.
3. Offline entry points (`10`–`18`): same context, same allowed pools, no reuse of
   revision features as legacy features.
4. Replace the `.gitignore`-ignored local panels with a declared frozen run once
   the generated-answer panel is approved.

B2 must not change lift formulas, tune methods, or launch judging/generation.

## B1.5 — Dev selection (implemented, no API)

`experiments/22_revision_dev_lab.py` evaluates three panels × four evidence
sources × 160 configurations on dev, with matched 25-cell budgets per lift family,
then validates the selected cells once on eval. Findings and the frozen finalist
are in `DEV_LAB.md` and `configs/revision/finalists_v1.yaml`.

## B2 — Generated panel (entry point implemented, blocked on API key)

`experiments/23_revision_evaluate.py` reuses `EvaluationRunner` with the resolved
revision context. It refuses nothing implicitly: the panel, source and frozen lift
are explicit. The full eval panel (three regimes × three sources) requires about
5,000 new prompts; a smoke run must pass before the full run, and the outputs are
frozen for the human study.

## Behaviour changes in shared code (verified, no stored result edited)

| Change | Effect on stored results | Effect on reruns |
|---|---|---|
| `build_pool_tensor(..., per_query_loo=True)` now uses priors from `minus_query_bundle()` | none (artifacts untouched) | train-LOO priors now exclude the target in both roles; previously they included it |
| `CrossEncoderReranker` cache key is content-based | none | the existing CE cache misses and is recomputed with identical deterministic scores |
| `FeedbackBuilder` self-exclusion uses integer positions | none | new builds no longer contain self-pairs (the submitted DBs contain 878) |
| `FeedbackBundle.prior_mean` computed from the supplied rows | none | priors now follow the filtered evidence set instead of the raw DB |
| `ReplySimilarity` accepts v2 group / full-reply overrides | none | legacy callers keep the submitted hashes by default |

The two stale aggregation assertions in `tests/test_feedback_aggregation.py`
were a helper bug (`np.nan or ""` produced the string `"nan"` scope keys); the
helper now matches the documented empty-label exclusion. Aggregation code was not
changed.


## E — Human study (application implemented; data pending)

`revision_ieee_access/02_human_study/` contains a Flask + SQLite application with
two blocks per ticket (first replies, then retrieved tickets), independent
correctness ratings, pairwise preference with ties/both-incorrect/cannot-judge,
opaque item IDs, server-side key, one immutable submission per rater and item, and
a deterministic balanced A/B assignment. `experiments/24_export_study.py` builds
the frozen package, refuses mismatched query sets or differing baseline answers,
and samples proportionally by whether feedback changed the top-1.

The application was exercised locally end to end (login → item → submission →
admin progress → CSV export) and confirmed not to expose method labels. The study
package is built only after the generated panel is final; any change requires a new
`--version`, and the app refuses to mix packages in one database.

## C — Local readiness check (after B)

Prepare versioned group artifacts and splits, without overwriting legacy inputs.
Measure actual history/pool/evidence coverage through normal entry points and all
retrievers. Reproduce aggregate audit tables and compute exact request/cache
budgets. This establishes milestone A (development readiness), not completion of
R1-M7. Milestone B requires dev selection, controlled comparisons, final eval
panels, uncertainty analysis and manuscript changes.

Later revision work reuses these boundaries for reply-similarity feedback,
matched retrievers, tuned Laplace/ablations and gate analysis. GFT export must
reference frozen run IDs; no study outputs are selected from this package-A check.

## E — Human study (application implemented; data pending)

`revision_ieee_access/02_human_study/` contains a Flask + SQLite application with
two blocks per ticket (first replies, then retrieved tickets), independent
correctness ratings, pairwise preference with ties/both-incorrect/cannot-judge,
opaque item IDs, server-side key, one immutable submission per rater and item, and
a deterministic balanced A/B assignment. `experiments/24_export_study.py` builds
the frozen package, refuses mismatched query sets or differing baseline answers,
and samples proportionally by whether feedback changed the top-1.

The application was exercised locally end to end (login → item → submission →
admin progress → CSV export) and confirmed not to expose method labels. The study
package is built only after the generated panel is final; any change requires a new
`--version`, and the app refuses to mix packages in one database.

## Freeze rule

Nothing in retrieval, generation, sampling or A/B assignment may change after the
two experts receive a package. The study manifest records run directories,
details-file hashes, config hash, dataset hash and the assignment seed. The
analysis uses the exported `ratings.csv` with the hidden side mapping.
