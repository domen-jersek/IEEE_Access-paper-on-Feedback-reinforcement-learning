# 01 — Revision data protocol

**Status: data layer, retrieval panels, dev selection and the generated panel are
complete. The human study is the next step; the generated runs are frozen.**

**Manuscript preparation:** `revision_ieee_access/03_manuscript/` contains the paper
spine, the prior-work comparison, draft sections, the provenance table and the
review-response map. `access.tex` is not modified until the rewrite is approved.

This revision extends the existing `src/` and `experiments/` pipeline. It does not
create a second evaluator, modify submitted results, or certify a semantic
procedure split.

| Layer | Implementation |
|---|---|
| Data contract | `DataProtocolConfig`, `resolve_data_context`, pinned SHA-256 inputs, immutable views, train-only history, both-endpoint feedback selection |
| Groups | `src/data/groups_v2.py`: `duplicate_template_v1` and `prefix_template_v1` (connected components, versioned, structural proxy) |
| Splits | `src/data/split_v2.py`: atomic group assignment, deterministic per seed, 5 seeds |
| History retrieval | `src/retrieval/history.py`: dense index rebuilt from frozen vectors, BM25 fitted on history, hybrid/CE on the same pool, content-based CE cache keys |
| Priors | `FeedbackBundle` priors from exactly the supplied rows; `minus_query_bundle()` removes the target in both roles |
| Reply baseline | `src/feedback/reply_similarity.py`: exact and BGE reply similarity on the same pairs |
| Retrieval panels | `experiments/21_revision_retrieval_panels.py`: four regimes × compact grid × five seeds |
| Dev selection | `experiments/22_revision_dev_lab.py`: tuned Laplace, matched-budget ablations, reply baselines, dev→eval validation |
| Generated answers | `experiments/23_revision_evaluate.py`: same `EvaluationRunner`, revision data context, frozen finalist |
| Freeze | `experiments/27_freeze_runs.py`: write-once hash manifest of the nine runs |
| Retriever check | `experiments/28_matched_retrievers.py`: same lift across dense/BM25/hybrid |
| Power | `experiments/29_power_multiplicity.py`: pre-specified Holm families and 80 % MDE |
| Decision/cost | `experiments/30_decision_and_cost.py`: abstention headroom and end-to-end cost |
| Lift diagnostics | `experiments/34_lift_diagnostics.py`: score distribution, cap frequencies, π_s; `LIFT_DIAGNOSTICS.md` |
| Study export | `experiments/24_export_study.py`: blinded A/B, balanced assignment, frozen manifest |
| Study app | `revision_ieee_access/02_human_study/`: Flask + SQLite, two blocks, opaque IDs |
| Human analysis | `experiments/33_human_eval.py`, `src/evaluation/human_eval.py`; first-rater results in `HUMAN_EVAL_RESULTS.md` |
| Frozen finalist | `configs/revision/finalists_v1.yaml` |
| Evidence | `R1M7_RESPONSE.md`, `DEV_LAB.md`, `MANUSCRIPT_PATCH.md`, `tables/*.csv` |

## Commands (no API calls)

```powershell
.\.venv\Scripts\python.exe experiments/20_build_revision_data.py
.\.venv\Scripts\python.exe experiments/21_revision_retrieval_panels.py --protocol conditioned
.\.venv\Scripts\python.exe experiments/21_revision_retrieval_panels.py --protocol blind
.\.venv\Scripts\python.exe experiments/22_revision_dev_lab.py --validate-eval
.\.venv\Scripts\python.exe experiments/27_freeze_runs.py --label v1
.\.venv\Scripts\python.exe experiments/27_freeze_runs.py --verify results/revision/frozen/frozen_runs_v1.json
.\.venv\Scripts\python.exe experiments/28_matched_retrievers.py --source conditioned
.\.venv\Scripts\python.exe experiments/29_power_multiplicity.py
.\.venv\Scripts\python.exe experiments/34_lift_diagnostics.py --panel disjoint_prefix
.\.venv\Scripts\python.exe -m pytest tests/test_data_protocol.py tests/test_revision_boundaries.py tests/test_reply_similarity.py tests/test_human_eval.py -q
```

`23_revision_evaluate.py` is the generated-answer entry point (needs
`OPENROUTER_API_KEY`); it uses the frozen finalist in
`configs/revision/finalists_v1.yaml` and must not be used with other settings after
the human-evaluation freeze.

`20_` writes versioned groups, splits, overlap audit and a manifest under
`data/processed/revision/` (git-ignored, private). `21_` writes a run directory
under `results/revision/` with `panel_metrics.csv`, `saturation.csv`,
`pool_overlap_audit.csv`, `scope_coverage.csv`, `summary.json` and a manifest.

## Not yet implemented

`experiments/03_build_feedback.py` and `experiments/04_evaluate.py` still use the
submitted all-ticket path. Until they accept `--data-protocol`, revision
generation cannot be launched from the normal entry points; this is deliberate,
so a legacy run cannot be mistaken for a revision run. Next increment: wire the
builder, evaluator and offline analyses to the same resolved context, with
negative tests and cache invalidation by `history_id`.

An existing local file is never overwritten silently. `20_` refuses to replace a
different artifact unless `--overwrite` is passed.

