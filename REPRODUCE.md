# REPRODUCE.md — ordered command list (P0–P4)

All commands run from `paper_ieee_access/`. Every script writes an immutable
`manifest_<run_id>.json`, updates `manifest_latest.json`, and appends a row to
`results/registry.csv`. Set `USE_TF=0` once per
shell to keep `transformers` from importing TensorFlow (PowerShell: `$env:USE_TF='0'`).

Install the notebook and test dependencies once with `python -m pip install -e ".[dev]"`.

Already completed before this programme (SIKDD replication, luna generator, dev split, seed 42):
`baseline`, `M1_global` (conditioned/blind/binary), `M2_team`, `M3_class`, `M4_intersection`
→ `results/*_dev_conditioned_*/`. These are the reference runs; nothing below modifies them.

## P0 — sanity

```powershell
python -m pytest -q                                                   # 43 tests
python experiments/04_evaluate.py --method M2_team --limit 3 --tag smoke   # ~$0.05; run twice -> 2nd run = 100% cache hits
```
Check: `results/M2_team_dev_conditioned_continuous_smoke/manifest_latest.json` exists, a row was appended to
`results/registry.csv`, the second run's `*_summary.json` shows `generation_cache.misses = 0`.
Delete the smoke folder afterwards (or keep it; it is tagged).

## P1 — methodological validity (all free except 09)

```powershell
python experiments/07_validate_proxy.py                     # proxy vs generated deltas (minilm + bge reply-sim; bge encodes 1,595 replies once, ~10 min CPU)
python experiments/10_feedback_calibration.py               # judge reliability, scope priors, lift saturation on 319 dev pools
python experiments/14_audit_identical_prompts.py             # correct diagnostic for legacy duplicate-generation deltas
python experiments/08_rescore.py                            # bge cosine + BERTScore on stored answers of all dev runs (CPU, ~20-40 min); add --no-bertscore for a fast first pass
python experiments/09_llm_judge_pairwise.py --judge-model <NON-OPENAI MODEL ID> --runs M2_team_dev_conditioned_continuous M4_intersection_dev_conditioned_continuous M1_global_dev_conditioned_continuous   # ~600 calls, cached
```
Decision gate 1 (`results/proxy_validation/report.json -> decision`):
`adopt_for_config_selection` must be true to use the ladder/blend sweeps for choosing configs.
If `adopt_for_per_ticket_labels` is false, the learned gate is still trained on proxy labels but
must be *reported* on generated labels (05_gate_cv.py --eval-details does this).

## P2 — retriever ladder (free)

```powershell
python experiments/02_build_index.py --embedder bge          # data/processed/faiss_index_bge/ (~5 min CPU)
python experiments/11_retriever_ladder.py --split dev         # ~15-30 min CPU (cross-encoder on 319x100 pairs, cached)
```
Read `results/retriever_ladder/ladder_curve.csv` (retriever strength vs best feedback gain,
coverage confound) and `grid.csv`. Decision gate 2: does team/intersection feedback still
gain on `hybrid_rrf`, `dense_bge`, `ce_rerank`?

## P3 — granularity (free)

```powershell
python experiments/12_learn_blend.py                                     # laplace, absolute; weights learned on train (per-query LOO), evaluated on dev
python experiments/12_learn_blend.py --lift laplace_eb --tag eb          # calibrated lift variant
python experiments/12_learn_blend.py --retriever hybrid_rrf --scale-mode pool_std --tag hybrid   # if P2 says hybrid is the right base
python experiments/13_feedback_volume_curve.py
$d=(Get-ChildItem results/M4_intersection_dev_conditioned_continuous/*_details.json | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
python experiments/05_gate_cv.py --features-parquet results/blend_eb/gate_features_train_M4_intersection.parquet --eval-features-parquet results/blend_eb/gate_features_dev_M4_intersection.parquet --eval-details $d --tag eb_m4
python experiments/11_retriever_ladder.py --split eval --retrievers dense_minilm bm25 hybrid_rrf --tag eval
python experiments/11_retriever_ladder.py --split eval --regime disjoint --retrievers dense_minilm bm25 hybrid_rrf --tag disjoint
```

## P4 — generation confirmation (API cost, finalists only; ~$4 per dev run with luna)

```powershell
# finalists on dev (luna)
python experiments/04_evaluate.py --method M4_intersection --lift laplace_eb
python experiments/04_evaluate.py --method M5_backoff --lift laplace_eb --min-evidence 2
python experiments/04_evaluate.py --method M2_team --lift laplace_eb
python experiments/04_evaluate.py --method M4_intersection --retriever hybrid_rrf --lift laplace_eb --scale-mode pool_std --pool-lambda 2
python experiments/04_evaluate.py --method baseline --retriever hybrid_rrf
# economic parity (generator != feedback judge)
python experiments/04_evaluate.py --method baseline --generator-model openai/gpt-4o-mini
python experiments/04_evaluate.py --method M2_team --lift laplace_eb --generator-model openai/gpt-4o-mini
python experiments/04_evaluate.py --method M4_intersection --lift laplace_eb --generator-model openai/gpt-4o-mini
# robustness
python experiments/04_evaluate.py --method M2_team --seed 123 ; python experiments/04_evaluate.py --method M4_intersection --seed 123
python experiments/04_evaluate.py --method M2_team --seed 456 ; python experiments/04_evaluate.py --method M4_intersection --seed 456
python experiments/04_evaluate.py --method M2_team --split eval ; python experiments/04_evaluate.py --method M4_intersection --split eval
python experiments/04_evaluate.py --method M4_intersection --split eval --regime disjoint
# independent metrics on the new runs
python experiments/08_rescore.py --force
python experiments/09_llm_judge_pairwise.py --judge-model <NON-LUNA MODEL ID> --runs M2_team_dev_conditioned_continuous M4_intersection_dev_conditioned_continuous M1_global_dev_conditioned_continuous
python experiments/06_report.py
```

## Paper notebooks

```powershell
python notebooks/build_ieee_notebooks.py
python notebooks/validate_ieee_notebooks.py --execute
```

Open from the `notebooks/` directory. The suite is analysis + modeling, with an
explanation on every output:

| Notebook | Type | Content |
|----------|------|---------|
| `02_protocol_and_validity.ipynb` | analysis | corpus, splits, feedback protocols, judge calibration, proxy validity |
| `03_when_feedback_helps.ipynb` | analysis | conditional benefit, evidence, scope, volume, ceilings, retriever ladder, negative controls |
| `04_modeling_the_prior.ipynb` | modeling | lift formulas, centering, pool-relative scaling, backoff, blend null, aggression tradeoff |
| `05_modeling_the_control_policy.ipynb` | modeling | pre-generation gating, dev→eval methodology, policy value, ceiling recovery |
| `06_final_results_and_claims.ipynb` | results | locked configuration, dev + eval results, robustness, independent metrics, judge, claim ledger |
| `07_general_gate.ipynb` | modeling | general gate (grouped CV, dev-frozen thresholds), decomposition, magnitude policy, live gated runs |

## P5 — general gate study (fixed methodology)

```powershell
# one gate across all routing signals and both protocols, dev -> eval
# (GroupKFold by ticket, thresholds frozen on dev, clustered CIs, decomposition)
python experiments/17_gate_study.py --tag general

# semantic relevance-filter routings (offline negative result)
python experiments/11_retriever_ladder.py --routings M4_intersection M5_backoff semantic_intersection_tau0.6 semantic_intersection_tau0.7 semantic_backoff_tau0.6 semantic_backoff_tau0.7 --tag semantic
python experiments/11_retriever_ladder.py --feedback-protocol blind --routings M4_intersection M5_backoff semantic_intersection_tau0.6 semantic_intersection_tau0.7 semantic_backoff_tau0.6 semantic_backoff_tau0.7 --tag semantic_blind
```

## P6 — magnitude-aware control policy

```powershell
# expected-delta regression + cost-sensitive policy + action selection (free)
python experiments/18_magnitude_policy.py

# live gated evaluation (dev verification + eval; cache-only after the prompts are warm)
python experiments/04_evaluate.py --method M4_intersection --gating-model results/gate_study_general/gate_model.joblib --gating-threshold 0.3
python experiments/04_evaluate.py --method M4_intersection --split eval --gating-model results/gate_study_general/gate_model.joblib --gating-threshold 0.3
python experiments/04_evaluate.py --method M5_backoff --lift laplace_eb --prior-strength 2 --scale-mode pool_std --pool-lambda 0.5 --min-evidence 2 --feedback-protocol blind --gating-model results/gate_study_general/gate_model.joblib --gating-threshold 0.3
python experiments/04_evaluate.py --method M5_backoff --lift laplace_eb --prior-strength 2 --scale-mode pool_std --pool-lambda 0.5 --min-evidence 2 --feedback-protocol blind --split eval --gating-model results/gate_study_general/gate_model.joblib --gating-threshold 0.3
```

Regeneration note
- Dev runs generated before 2026-09-17 had no response cache and are not reproducible; the affected
  dev configurations were regenerated on 2026-09-22 and now carry a `generation_regime` block
  (regime id, cache hits/misses, `warm` flag). Only warm runs of the same regime are compared.

Notes
- Baseline runs generate once per ticket (feedback == baseline), so a baseline run costs half a method run.
- Any method ticket whose baseline and feedback prompts are identical also generates once, preventing nondeterministic false deltas.
- `--generator-model` changes the folder suffix (`_gen<model>`); the feedback DB / judge are unchanged.
- Runs on seeds != 42 use the seed-42 feedback DB minus the evaluated split (see README "Feedback DB across seeds").
