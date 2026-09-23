# IEEE Access paper — outline and claim→evidence matrix

Working title: **When Does Feedback Help Retrieval-Augmented IT Support? A Conditional Study of
LLM-Judge Feedback, Calibration, and Control Policies**

Positioning: a systematic, reproducible empirical study (not a "new SOTA" method paper). The
feedback is LLM-judge-simulated; the conditioned protocol is resolution-informed and is reported as
an upper bound; the blind protocol is the deployable result.

## 1. Claims and evidence

Status: **ready** = artifact exists and is leak-free; **pending** = waiting on B3/B4 runs; **weak** =
supported but small/uncertain; **negative** = documented rejection.

| # | Claim | Evidence (numbers) | Artifact | Status |
|---|---|---|---|---|
| C1 | Feedback benefit is conditional on how the feedback is produced (protocol reversal) | Dev: M2 +0.022 / M4 +0.031 conditioned vs −0.034 / −0.051 blind; eval: +0.0138 (p=.012) / +0.0151 (p=.057) vs −0.038 (p<.05); BGE agrees (+0.012/+0.012 vs −0.017) | `results/<run>/*_summary.json`; `results/rescored/method_comparison_v2.csv`; notebook 06 | ready |
| C2 | Broad pooling harms; fine scopes help | Offline ladder: M1/M3 negative, M2/M4/M5 positive across six retrievers; stable under both protocols | `results/retriever_ladder{,_blind}/grid.csv` | ready |
| C3 | Judge scores are zero-inflated and naive popularity lift saturates | 58.7% exact zeros (conditioned), 48.3% (blind); Laplace 40–50% of lifts at the negative cap; EB removes saturation | `results/feedback_calibration/{report.json,saturation.csv}` | ready |
| C4 | Calibration (EB centering + pool-relative scaling) removes the harm; gains are not individually significant | Blind offline −0.051 → +0.007…+0.010; generated blind dev +0.0096 (p=.42), eval +0.0021 (p=.86), disjoint +0.0051 (p=.16) | `results/rescored/method_comparison_v2.csv`; notebook 04 | ready (weak effect) |
| C5 | Lift-formula ordering is stable: EB > tanh > Laplace > LCB | Conditioned 0.053/0.046/0.044/0.036; blind 0.011/0.007/0.006/0.005 (offline proxy) | `results/retriever_ladder_liftablation{,_blind}/grid.csv` | ready |
| C6 | The gate is risk control, not gain: it recovers 0.54–0.56 of the ceiling where harm is concentrated and is neutral otherwise | Grouped dev OOF AUC 0.699 [0.676, 0.721]; uncalibrated blind policy +0.001…+0.002 (gain +0.039/+0.041, recovery 0.54/0.56); calibrated blind −0.003; conditioned ±0.0005 | `results/gate_study_general/{learned_gate.csv,learned_gate_decomposition.csv}`; notebook 07 | ready |
| C7 | The gate identifies volatility, not harm: decomposition explains the value | Closed sets: uncalibrated blind mean −0.07…−0.10 (56–60% harmful); calibrated blind +0.006 (62% no-ops); conditioned +0.003 (55–90% no-ops) | `results/gate_study_general/learned_gate_decomposition.csv` | ready |
| C8 | Expected-value (magnitude) modelling beats sign-only gating; action selection turns positive | GBR R²=0.47, ρ=0.17; blind policies +0.0074/+0.0024/+0.0035 vs sign +0.0023/+0.0010/−0.0011; action selection +0.0116 [+0.0034, +0.0205] vs sign −0.0079, best-fixed +0.0002, oracle +0.0571 | `results/magnitude_policy/{policy_table.csv,action_selection.csv}`; notebook 07 | ready |
| C9 | Independent metrics and an independent LLM judge agree in direction on the conditioned effect | Gated intersection: cosine +0.0164 (p=.028), BGE +0.0130 (p=.0002), BERTScore +0.0160; judge net win +0.31 (M4) and +0.31 (gated); blind ≈ 0 | `results/rescored/method_comparison_v2.csv`; `results/answer_judge/summary.json` | ready |
| C10 | The calibrated blind gain does not transfer to stronger retrievers | Generated: hybrid RRF −0.006, cross-encoder rerank +0.004 vs dense MiniLM +0.007/+0.010; baseline answer quality nearly identical (0.654–0.658) | `results/M4_intersection_dev_blind_*` runs | ready (confounded lift configs; note) |
| C11 | Conditioned robustness to seeds and unseen procedures | M4 disjoint +0.0152 (BGE +0.0096, p=.0065), M2 disjoint +0.0103 (BGE +0.0072, p=.058); M4 seeds +0.015/+0.013 (BGE p=.009/.12); M2 seeds +0.006/−0.001 (BGE p=.051/.69) | `results/M*_seed*`, `results/M*_disjoint` | ready |
| C12 | Cross-generator robustness (not luna) | Gemini 3.8 Flash: M2 +0.029 (BGE +0.015, p<1e-4), M4 +0.040 (BGE +0.024, p<1e-4), M5 blind +0.007 (BGE +0.006, p=.07) vs luna +0.014/+0.015/+0.002 | `results/*_gengemini38flash` | ready |
| N1 | Semantic relevance filter is worse than base routings | Offline, all retrievers, both protocols | `results/retriever_ladder_semantic{,_blind}/grid.csv` | negative |
| N2 | Learned blend collapses to the best fixed scope (null) | Train-selected weights ≈ intersection | `results/blend{,_eb}/learned_weights.json` | negative |
| N3 | Sign-only multi-action selection loses | −0.0079 vs best-fixed +0.0002 | `results/gate_study_general/multi_action.csv` | negative |

## 2. Proposed structure (~20 pages)

1. **Introduction** (1.5 pp) — repetitive-but-not-identical tickets; similarity ≠ procedure;
   feedback as a retrieval prior; three questions: when does it help, why does it fail, can it be
   controlled.
2. **Related work** (2 pp) — RAG, dense retrieval/reranking, learning-to-rank from implicit
   feedback, LLM-as-judge, calibration of noisy labels, selective prediction/gating.
3. **Data and protocols** (2.5 pp) — corpus (1,595; 878/319/398; disjoint), taxonomy, two
   LLM-judge protocols (conditioned = resolution-informed upper bound; blind = deployable), judge
   calibration (AUC 0.90 vs 0.84; zero inflation), generation regime and provenance.
4. **Method** (3 pp) — retrieval, lift formulas, routing scopes, scaling, gate/control policy,
   magnitude-aware policy; statistical protocol (grouped CV, dev-frozen thresholds, clustered CIs,
   counterfactual decomposition).
5. **When does feedback help?** (3 pp) — C1, C2, C5, C10; difficulty/evidence/scope; retriever
   ladder; lift ablation.
6. **Why does it fail?** (2.5 pp) — C3, C4; saturation mechanism; calibration rescue; negative
   results N1–N3.
7. **Controlling the intervention** (3 pp) — C6, C7, C8; gate construction, decomposition,
   magnitude policy and action selection; live gated runs.
8. **Results and validation** (2 pp) — eval table, independent metrics, judge, robustness (C11/C12
   when available).
9. **Discussion** (1.5 pp) — protocol-dependence, oracle leakage, calibration-vs-control
   substitution, practical recommendations.
10. **Limitations and conclusion** (1 pp) — LLM-judge simulation, single corpus, small effects,
    oracle-conditioned protocols, per-ticket judge/embedding disagreement.

## 3. Figure and table map (notebook → paper)

| Paper item | Source | Notebook |
|---|---|---|
| Fig. 1 corpus/splits | `figures/ieee_02_corpus_composition.png` | 02 |
| Fig. 2 judge reliability | `figures/ieee_02_judge_reliability.png` | 02 |
| Fig. 3 proxy validity | `figures/ieee_02_proxy_validity.png` | 02 |
| Fig. 4 scope ordering by protocol | `figures/ieee_03_scope_ordering_protocol.png` | 03 |
| Fig. 5 retriever strength vs gain | `figures/ieee_03_retriever_strength.png` | 03 |
| Fig. 6 benefit by difficulty / coverage | `figures/ieee_03_benefit_by_*.png` | 03 |
| Fig. 7 lift ablation | `figures/ieee_04_lift_ablation.png` | 04 |
| Fig. 8 saturation / scaling rescue | `figures/ieee_04_*.png` | 04 |
| Fig. 9 gate AUC by protocol | `figures/ieee_07_general_gate_auc.png` | 07 |
| Fig. 10 gate decomposition | `figures/ieee_07_gate_decomposition.png` | 07 |
| Fig. 11 magnitude action selection | `figures/ieee_07_magnitude_action.png` | 07 |
| Fig. 12 protocol reversal (dev) | `figures/ieee_06_dev_protocol_reversal.png` | 06 |
| Fig. 13 independent metrics | `figures/ieee_06_independent_metrics.png` | 06 |
| Table 1 data/splits | notebook 02 tables | 02 |
| Table 2 dev results (canonical) | notebook 06 cell | 06 |
| Table 3 eval results + independent metrics + judge | notebook 06 cells | 06 |
| Table 4 gate policy + decomposition | notebook 07 cells | 07 |
| Table 5 magnitude policy + action selection | notebook 07 cells | 07 |
| Table 6 lift ablation | notebook 04 cell | 04 |

## 4. What we will not claim

- No "human feedback": the signal is LLM-judge-simulated; human validation is future work.
- No deployable improvement over the baseline: the calibrated blind result is null; the gate is
  risk control, not gain.
- No oracle-conditioned gains as achievable: the conditioned protocol sees the historical
  resolution and is an upper bound.
- No per-ticket agreement between the embedding metric and the LLM judge (0.35–0.40); only
  aggregate direction.
- No generalisation beyond the single organizational corpus without the pending robustness runs.

## 5. Open items

- Final rescore pass over the B3/B4 runs, then one final notebook re-execution so every number in
  the paper is frozen from warm-cache artifacts of a single regime (plus the separate gemini
  regime).
- Human annotation study (not currently feasible): the single biggest credibility lever.
- Optional: a generated strong-retriever baseline with identical lift/scaling for a clean
  retriever-transfer claim (currently confounded by differing lift configs).
