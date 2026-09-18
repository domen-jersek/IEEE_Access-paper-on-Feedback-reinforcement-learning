# REPRODUCE.md — ordered command list (P0–P4)

All commands run from `paper_ieee_access/`. Every script writes `manifest.json` in its
output folder and appends a row to `results/registry.csv`. Set `USE_TF=0` once per
shell to keep `transformers` from importing TensorFlow (PowerShell: `$env:USE_TF='0'`).

Already completed before this programme (SIKDD replication, luna generator, dev split, seed 42):
`baseline`, `M1_global` (conditioned/blind/binary), `M2_team`, `M3_class`, `M4_intersection`
→ `results/*_dev_conditioned_*/`. These are the reference runs; nothing below modifies them.

## P0 — sanity

```powershell
python -m pytest -q                                                   # 39 tests
python experiments/04_evaluate.py --method M2_team --limit 3 --tag smoke   # ~$0.05; run twice -> 2nd run = 100% cache hits
```
Check: `results/M2_team_dev_conditioned_continuous_smoke/manifest.json` exists, a row was appended to
`results/registry.csv`, the second run's `*_summary.json` shows `generation_cache.misses = 0`.
Delete the smoke folder afterwards (or keep it; it is tagged).

## P1 — methodological validity (all free except 09)

```powershell
python experiments/07_validate_proxy.py                     # proxy vs generated deltas (minilm + bge reply-sim; bge encodes 1,595 replies once, ~10 min CPU)
python experiments/10_feedback_calibration.py               # judge reliability, scope priors, lift saturation on 319 dev pools
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
python experiments/05_gate_cv.py --features-parquet results/blend/gate_features_train.parquet --eval-features-parquet results/blend/gate_features_dev.parquet --eval-details results/M4_intersection_dev_conditioned_continuous/<newest>_details.json
```

## P4 — generation confirmation (API cost, finalists only; ~$4 per dev run with luna)

```powershell
# finalists on dev (luna)
python experiments/04_evaluate.py --method M5_backoff
python experiments/04_evaluate.py --method M6_blend --blend-weights results/blend/learned_weights.json
python experiments/04_evaluate.py --method M4_intersection --lift laplace_eb            # if calibration helps in P2/P3
python experiments/04_evaluate.py --method M4_intersection --retriever hybrid_rrf --scale-mode pool_std   # if P2 says so (+ matching baseline below)
python experiments/04_evaluate.py --method baseline --retriever hybrid_rrf
# economic parity (generator != feedback judge)
python experiments/04_evaluate.py --method baseline --generator-model openai/gpt-4o-mini
python experiments/04_evaluate.py --method M2_team  --generator-model openai/gpt-4o-mini
python experiments/04_evaluate.py --method M4_intersection --generator-model openai/gpt-4o-mini
# robustness
python experiments/04_evaluate.py --method M2_team --seed 123 ; python experiments/04_evaluate.py --method M4_intersection --seed 123
python experiments/04_evaluate.py --method M2_team --seed 456 ; python experiments/04_evaluate.py --method M4_intersection --seed 456
python experiments/04_evaluate.py --method M2_team --split eval ; python experiments/04_evaluate.py --method M4_intersection --split eval
python experiments/04_evaluate.py --method M4_intersection --split eval --regime disjoint
# independent metrics on the new runs
python experiments/08_rescore.py --runs "*gengpt4omini*" "M5_*" "M6_*" "*_eval_*"
python experiments/06_report.py
```

Notes
- Baseline runs generate once per ticket (feedback == baseline), so a baseline run costs half a method run.
- `--generator-model` changes the folder suffix (`_gen<model>`); the feedback DB / judge are unchanged.
- Runs on seeds != 42 use the seed-42 feedback DB minus the evaluated split (see README "Feedback DB across seeds").
