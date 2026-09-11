# Feedback as a Learnable Retrieval Prior

**A clean, modular research environment for studying human-feedback-driven retrieval
re-ranking in retrieval-augmented generation (RAG) for IT support ticket first-reply
generation.**

Jožef Stefan Institute, Ljubljana, Slovenia

---

## Table of Contents

1. [What this repository is](#what-this-repository-is)
2. [Research context and motivation](#research-context-and-motivation)
3. [Repository structure](#repository-structure)
4. [The data](#the-data)
5. [Data preparation (`data/`)](#data-preparation)
6. [The pipeline, module by module](#the-pipeline)
7. [Experimental design and anti-overfitting controls](#experimental-design)
8. [How to run the pipeline](#how-to-run-the-pipeline)
9. [Configuration and provenance](#configuration-and-provenance)
10. [Reproducibility](#reproducibility)

---

## What this repository is

This repository is a **from-scratch, standalone reimplementation** of a feedback-driven
retrieval system for IT support tickets. It is intentionally separated from prior
exploratory work (an earlier prototype lived in the `notebooks/` directory at the
repository root, which accumulated 227 ad-hoc evaluation runs and several versioned
feedback databases). This environment exists to satisfy journal-review-grade standards:

- every experiment is a single command-line invocation;
- every configuration is a frozen, versioned object;
- every output artifact carries a provenance hash;
- the train/dev/eval separation is explicit and enforced in code.

The code is organized as a Python package under `src/` driven by thin CLI entry
points under `experiments/`.

## Research context and motivation

IT service desks receive a steady stream of tickets that are *repetitive but not
perfectly repetitive*. A RAG system retrieves similar historical tickets (using
dense embeddings indexed by FAISS) and conditions a large language model on them to
generate a first reply. The central problem this work addresses:

> Text similarity is not the same as procedural correctness. Two tickets can share
> almost identical vocabulary (a "Citrix .ica file won't open" ticket vs. a Citrix
> troubleshooting ticket) yet require entirely different resolutions (a software-install
> form redirect vs. VPN troubleshooting steps).

The idea under study is to **augment retrieval with human feedback**: historical
candidates that past users rated as useful get promoted in the ranking, even if they
are less textually similar to the query. The key open questions are:

1. **When** does feedback help, and when does it hurt?
2. **Can we predict** (before generation) whether feedback will help or hurt a given
   query — i.e., can we build a *gate* that selectively deploys feedback?
3. **How is the feedback signal itself best produced** — from an oracle-conditioned
   judge (that sees the ground-truth reply) or a "blind" judge that simulates a real
   user with no access to ground truth?

This repository provides the data engineering, retrieval, feedback-generation,
evaluation, and learned-gate components needed to answer these questions with
statistical rigor.

## Repository structure

```
paper_ieee_access/
├── data/
│   ├── raw/                          # Immutable source data (see "The data")
│   └── processed/                    # Generated artifacts (gitignored, recreatable)
│       ├── dataset.parquet           #   Canonicalized ticket table (1,595 rows)
│       ├── dataset_manifest.json     #   Provenance: source hash, row counts, schema
│       ├── taxonomy.csv              #   118 raw labels -> 17 intent classes
│       ├── procedure_groups.json     #   Near-duplicate group statistics
│       ├── splits/                   #   train/dev/eval splits (JSON + stratification reports)
│       ├── faiss_index/              #   FAISS index + metadata parquet
│       ├── baseline_difficulty.csv   #   Per-ticket top-1 FAISS similarity
│       └── feedback_*.db             #   Fresh LLM-judge feedback databases
│
├── src/                              # The Python package (imports are `from src...`)
│   ├── config.py                     #   ProjectPaths + frozen dataclasses (single source of truth)
│   ├── data/                         #   canonical.py, taxonomy.py, groups.py, split.py
│   ├── retrieval/                    #   encoder.py, index.py, search.py
│   ├── generation/                   #   prompts.py, client.py
│   ├── feedback/                     #   judge.py, builder.py, lift.py, loader.py
│   ├── gate/                         #   features.py, model.py
│   └── evaluation/                   #   protocol.py, runner.py, metrics.py, reporting.py
│
├── configs/                          # YAML experiment definitions (base + M1..M4 + gate)
├── experiments/                      # CLI entry points 00..06 + shared utils.py
├── tests/                            # Unit tests (pytest)
├── results/                          # All evaluation outputs (gitignored)
├── pyproject.toml                    # Package + dependencies
├── requirements.txt
└── README.md
```

### Why `experiments/` and `src/` are separate

`src/` holds **reusable, testable logic** with no I/O side effects on source data.
`experiments/` are **thin orchestration scripts** that wire together components,
resolve paths via `ProjectPaths`, and write results. This separation means the core
retrieval/lift/gate logic is unit-testable, while the "run" scripts remain short and
auditable.

---

## The data

### Source

The raw data is the file `tickets_large_first_reply_label.csv` in the `notebooks/`
directory of the parent repository. It contains **1,597 rows** of anonymized IT
support tickets with:

| Column | Meaning |
|--------|---------|
| `Ref` | Original enterprise ticket ID (e.g. `R-544314`) |
| `Title_anon` | Anonymized ticket title |
| `Description_anon` | Anonymized ticket description |
| `first_reply` | The actual first reply — the **reference answer** used for evaluation |
| `Team->Name` | Resolving team (37 distinct values) |
| `Service subcategory->Name` | Granular service category (149 distinct values) |
| `label_auto` | Auto-derived intent label (118 distinct values) |

### Canonicalization summary

`experiments/00_canonicalize.py` produces `dataset.parquet`:

- **1,595 finalized tickets**, each keyed by a clean `seq_id` (`R-1`..`R-1595`).
- 1 junk row dropped (missing `Ref`), 6 exact-duplicate rows deduplicated, 1 genuine
  `Ref` collision preserved (two physically distinct tickets sharing an ID).
- A **17-class intent taxonomy** built from the 118 `label_auto` values, documented
  field-by-field in `taxonomy.csv` (see `src/data/taxonomy.py`).
- **Near-duplicate procedure groups** computed via hashing of normalized title +
  subcategory + reply text (`procedure_groups.json`): 50 multi-ticket groups covering
  147 tickets. These are essential for the *disjoint* split (see below).

Key properties of the dataset that shaped the design:

- **Heavy class/team imbalance** — the top 3 teams hold ~39% of tickets; 14 teams have
  fewer than 10 tickets.
- **Repeated procedures** — 82 tickets share one identical software-form reply; many
  resolutions are boilerplate form-redirects.
- **Near-duplicates are common** — 265 rows share a (title, subcategory) group.

---

## Data preparation

### `src/data/canonical.py` — cleaning

- Drops rows with missing `Ref`, trims whitespace, normalizes newlines.
- Handles 7 `Ref` collisions (6 exact duplicates dropped, 1 genuine collision kept with
  a `_b` suffix).
- Assigns clean sequential IDs, adds length/no-op convenience columns.
- Writes the frozen parquet + a `dataset_manifest.json` recording the SHA-256 of the
  source CSV (so reviewers can verify the pipeline was run on the exact source data).

### `src/data/taxonomy.py` — intent classes

Maps 118 `label_auto` values to 17 `intent_class` values (`admin_rights`,
`software_license`, `offboarding`, `onboarding`, `vpn_access`, ... `other`). The full
mapping is committed to `taxonomy.csv`. Unmapped labels and low-frequency classes are
pooled into `other` for stratification and routing.

### `src/data/groups.py` — near-duplicate detection

Computes `reply_hash`, `title_subcat_hash`, and `combined_group_hash` per ticket.
`combined_group_hash` (reply + title + subcategory) defines groups of near-identical
tickets used by the disjoint split.

### `src/data/split.py` — train/dev/eval

Produces stratified 3-way splits, with stratification applied in priority order:

1. **team** (teams with <5 tickets pooled into `TEAM_OTHER`);
2. **intent class** (classes with <10 pooled into `CLASS_OTHER`);
3. (difficulty bins are applied in later stages once the FAISS index is built).

Two regimes are produced:

- **`random`** — 5 seeds (42, 123, 456, 789, 1024). Near-duplicates *may* cross splits,
  reflecting a realistic repetitive-ticket stream.
- **`disjoint`** — group-aware (procedure-level) split keyed on `combined_group_hash`,
  ensuring no near-duplicate procedure appears in both train and eval. Measures true
  generalization to *unseen procedures*.

Each split file (`split_seed{seed}.json`) records train/dev/eval `seq_id` lists plus
ticket counts; a companion `*_report.csv` records the per-stratum composition for audit.

---

## The pipeline

### `src/retrieval/`

| File | Purpose |
|------|---------|
| `encoder.py` | `TicketEncoder` — wraps `SentenceTransformer` (`all-MiniLM-L6-v2`), encodes `title + \n + description` with normalized embeddings. |
| `index.py` | `FAISSIndex` — `IndexFlatIP` inner-product index with `IndexIDMap`; build/search/save/load, plus `compute_ticket_similarity` for baseline difficulty. |
| `search.py` | Two retrieval paths: `retrieve_baseline` (FAISS-only top-k) and `retrieve_feedback` (FAISS + lift, top-k by `enhanced_score`), plus overlap/pool-feature helpers. |

Lift computation lives in `src/feedback/lift.py` (single source of truth) and is invoked
by `search.py` with routing-aware positive/negative counts.

### `src/generation/`

| File | Purpose |
|------|---------|
| `prompts.py` | Prompt templates: `build_generation_prompt` (first-reply generation) and `build_judge_prompt` (conditioned vs blind scoring). |
| `client.py` | `LLMClient` — async `AsyncOpenAI` with temperature 0, retries with backoff, optional SQLite response cache for exact reproducibility. Reads `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_BASE_URL` from the environment. |

### `src/feedback/`

| File | Purpose |
|------|---------|
| `lift.py` | Lift formulas: `laplace`, `tanh`, `bayesian_lcb`, with clamping and `positive_only` support. Works on float `pos`/`neg` — continuous scores flow through naturally. |
| `judge.py` | `FeedbackJudge` — batches candidate pairs through the LLM client and parses 0–1 scores. |
| `loader.py` | `FeedbackDB` (SQLite schema + insert/count) and `aggregate_feedback_scores` (raw scores → nested `{candidate: {scope: {pos, neg}}}` dict). Supports two aggregation modes (see below). |
| `builder.py` | `FeedbackBuilder` — orchestrates per-query FAISS top-75 retrieval -> batched judge calls -> DB writes. Also includes `validate_feedback_pipeline()` for pre-flight diagnostics. |

#### How feedback aggregation works

Raw judges scores (0.0–1.0 per `(query, candidate)` pair) are stored in SQLite. At
**evaluation time**, they are aggregated into per-candidate, per-scope `pos`/`neg`
counts that feed into the lift formula. Two modes, selectable per run:

| Mode | How it works | Behavior |
|------|-------------|----------|
| `continuous` (default) | `pos = sum(scores)`, `neg = n − sum(scores)` | A candidate rated 0.75 by 100 queries gets `pos=75, neg=25` → strong lift. No information discarded. |
| `binary` | `score ≥ 0.80 → pos`, `score ≤ 0.40 → neg`, else ignored | A candidate rated 0.79 by 100 queries gets `pos=0, neg=0` → zero lift. Preserves the neutral band from the earlier SIKDD methodology. |

Because the raw scores are stored, you can switch modes **without rebuilding the
feedback DB** — pass `--agg-mode continuous` or `--agg-mode binary` to
`04_evaluate.py` and compare results side by side in the final report table.

**Lift formula (Laplace), the primary choice:**

```
p     = (pos + α) / (pos + neg + α + β)        # α = β = 1.0 (uniform Beta prior)
lift  = (p − 0.5) · min(1, n/2) · m            # m = 0.80 multiplier
lift  = clamp(lift, −cap, +cap)                # cap = 0.20
enhanced_score = faiss_score + lift
```

### `src/gate/`

| File | Purpose |
|------|---------|
| `features.py` | `extract_gate_features` — builds pre-generation feature vectors (retrieval confidence, margin/spread, lift statistics, evidence density, team/class one-hot, query length). |
| `model.py` | `train_evaluate_gate` — nested cross-validation (outer 5-fold, inner 3-fold GridSearchCV) over XGBoost and logistic regression; feature importance; final model fitting + thresholded prediction. |

Features are restricted to signals available **before generation** — no reference-reply
(oracle) leakage.

### `src/evaluation/`

| File | Purpose |
|------|---------|
| `protocol.py` | `evaluate_one_ticket` — retrieves baseline + feedback candidates, generates both replies in parallel, computes metrics and deltas. |
| `runner.py` | `EvaluationRunner` — async bulk evaluation with semaphore concurrency, interim saves every N tickets, final detail/summary JSON. |
| `metrics.py` | `cosine_similarity` (`multi-qa-MiniLM-L6-cos-v1`, independent of retrieval embedding), `rouge_l_f1`, and delta helpers. |
| `reporting.py` | Statistical tools: bootstrap CI, Wilcoxon signed-rank, Cohen's d, oracle-decile analysis, gate-cutoff sweep, oracle-ceiling analysis. |

---

## Experimental design

### Train / Dev / Eval separation (enforced)

| Split | Size (seed 42) | Role |
|-------|----------------|------|
| **train** | 878 | Builds the feedback memory; trains the learned gate. |
| **dev** | 319 | Method development, feature selection, gate threshold tuning. |
| **eval** | 398 | Touched exactly once for the final paper numbers. |

Hard guarantees:

1. **Feedback is built from train queries only.** Dev/eval query IDs never appear in
   the feedback database (asserted where feedback is loaded).
2. **Self-exclusion** — a query's own ticket is excluded from its FAISS candidate pool
   (and, for train-side gate training, its own feedback rows are leave-one-out removed).
3. **Dev for everything tunable** — thresholds, features, model selection happen on dev.
   Eval results are computed by a version-frozen config and never used to tune.

### Two evaluation regimes

- **Random (in-distribution):** mirrors production on a repetitive ticket stream.
- **Disjoint (procedure-separated):** near-duplicate procedures never cross the
  train/eval boundary, isolating genuine generalization.

Reporting the gap between these two regimes is itself a finding about overfitting.

### Multi-seed stability

Five random seeds produce independent splits. Headline results are reported as
**mean ± std** across seeds, never a single cherry-picked seed.

### Retrieval methods

| Method | Routing | Lift | Gating |
|--------|---------|------|--------|
| Baseline | none (FAISS-only) | none | none |
| M1 global | all feedback pooled | Laplace | none |
| M2 team-only | within team | Laplace | none |
| M3 class-only | within intent class | Laplace | none |
| M4 intersection | team ∩ class | Laplace | none |
| M gate learned | global + learned gate | Laplace | learned (XGBoost / logistic) |

---

## How to run the pipeline

### 0. Setup

```bash
cd paper_ieee_access
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Unix:
source .venv/bin/activate

pip install -r requirements.txt
```

> The source CSV is read from the *parent* repository's `notebooks/` directory
> (`ProjectPaths.source_csv`). Point it at your own data by editing `src/config.py`.

### 1. Data preparation (offline)

```bash
python experiments/00_canonicalize.py   # -> dataset.parquet, taxonomy.csv, groups.json
python experiments/01_split.py          # -> splits/ (5 random + 1 disjoint)
python experiments/02_build_index.py    # -> faiss_index/ + baseline_difficulty.csv
```

The first download of `all-MiniLM-L6-v2` is ~90 MB.

### 2. Feedback generation (requires API)

Set your key:

```bash
$env:OPENROUTER_API_KEY="..."                # PowerShell
# export OPENROUTER_API_KEY="..."            # bash
```

**Recommended: validate first (20 queries, ~$0.50, ~2 minutes).**

```bash
python experiments/03_build_feedback.py --validate
```

This judges the top-75 FAISS candidates on 20 random train queries with both
protocols, then prints a diagnostic report:

```
VALIDATION SUMMARY
  [conditioned]  mean=0.63  positive=35%  negative=12%  neutral=53%
  [blind]        mean=0.58  positive=28%  negative=18%  neutral=54%
  Conditioned-vs-blind Pearson r: 0.72
Proceed with full build on 878 queries? [y/N]
```

Check that:
- Scores are spread across 0–1 (not all clustered near 0.5).
- The conditioned-vs-blind correlation is reasonable (r > 0.60).
- The positive/negative percentages make sense for your data.

If it looks good, answer `y` to continue with the full build. If something is wrong,
investigate without having spent $18 on 131K wasted calls.

**Full build:**

```bash
python experiments/03_build_feedback.py          # -> feedback_conditioned.db + feedback_blind.db
```

| What | Detail |
|------|--------|
| Judge model | `openai/gpt-luna-latest` |
| Candidates per query | top-75 FAISS (chosen because the ±0.20 lift cap means candidates ranked >75 are almost never promoted into the top-5) |
| Queries judged | 878 train queries only |
| Pairs per protocol | 878 × 75 = 65,850 |
| Total pairs | 131,700 (both protocols) |
| Approx. cost | ~$18 |
| Output | `feedback_conditioned.db` + `feedback_blind.db` (raw continuous scores in SQLite, ~15 MB each) |
| Caching | Judge responses are SHA-256 hashed → re-running resumes without re-incurring cost |
| Protocols | `conditioned` (judge sees query + ground-truth reply + candidate) and `blind` (judge sees only query + candidate — simulated real user) |

### 3. Evaluation (requires API)

```bash
# Dev split — iterate here for method development
python experiments/04_evaluate.py --method M1_global --split dev --seed 42
python experiments/04_evaluate.py --method M2_team   --split dev --seed 42
python experiments/04_evaluate.py --method M3_class  --split dev --seed 42
python experiments/04_evaluate.py --method M4_intersection --split dev --seed 42

# Compare continuous vs binary aggregation (same feedback DB, different lifts):
python experiments/04_evaluate.py --method M1_global --split dev --agg-mode binary

# Train split — produces the data for gate training
python experiments/04_evaluate.py --method M1_global --split train --seed 42

# Eval split — run ONCE at the very end
python experiments/04_evaluate.py --method M1_global --split eval --seed 42
```

Each run writes `results/<experiment_id>/<experiment_id>_<ts>_details.json` and a
`_summary.json`. The experiment ID includes the aggregation mode, so continuous and
binary runs get separate, non-overlapping result folders.

Useful flags:
- `--feedback-protocol blind` — use the blind feedback DB instead of conditioned.
- `--agg-mode binary` — use 0.80/0.40 threshold binarization instead of continuous aggregation.
- `--regime disjoint` — use the procedure-disjoint split.
- `--seed 123` — switch to a different random split seed.

### 4. Learned gate (optional direction)

```bash
# First, evaluate the train split (needed for gate training targets)
python experiments/04_evaluate.py --method M1_global --split train --seed 42

# Then train the gate
python experiments/05_gate_cv.py \
    --details-json "results/M1_global_train_conditioned_continuous/*_details.json" \
    --feedback-protocol conditioned
```

Performs nested CV (outer 5-fold, inner 3-fold) over XGBoost + logistic regression and
writes `results/gate/gate_cv_results.json` + `feature_importance.json`.

### 5. Reporting

```bash
python experiments/06_report.py --input-dir results/ --output-dir results/report/
```

Produces `method_comparison.csv`, per-experiment `oracle_deciles_*.csv`, and
`gate_sweep_*.csv`.

### 6. Tests

```bash
pytest tests/ -v
```

---

## Configuration and provenance

- **API keys**: `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`), optional `OPENROUTER_BASE_URL`.
- **Judge model**: pinned in `experiments/03_build_feedback.py` as
  `JUDGE_MODEL = "openai/gpt-luna-latest"`. To change it, edit that variable and re-run.
- **Generator model**: set per-experiment in `configs/*.yaml` or `src/config.py`
  dataclasses — never hardcoded in the runner.
- **Lift, routing, gating parameters**: all live in `src/config.py` (frozen dataclasses)
  or `configs/*.yaml`. Nothing is hardcoded elsewhere.
- **Provenance**: `EvalConfig.config_hash` (SHA-256 of the serialized config) is stamped
  into every result record; data artifacts record the source CSV hash. This lets
  reviewers confirm a run used exactly the claimed configuration.

## Command reference

| Script | Key flags | Output |
|--------|-----------|--------|
| `00_canonicalize.py` | none | `dataset.parquet`, `taxonomy.csv`, `groups.json` |
| `01_split.py` | none | `splits/split_seed{42..1024}.json`, `*_disjoint.json` |
| `02_build_index.py` | none | `faiss_index/`, `baseline_difficulty.csv` |
| `03_build_feedback.py` | `--validate`, `--validate-only`, `--n-validate N`, `--seed`, `--regime` | `feedback_conditioned.db`, `feedback_blind.db` |
| `04_evaluate.py` | `--method`, `--split`, `--seed`, `--regime`, `--feedback-protocol`, `--agg-mode` | `results/*_details.json`, `*_summary.json` |
| `05_gate_cv.py` | `--details-json`, `--feedback-protocol` | `results/gate/gate_cv_results.json` |
| `06_report.py` | `--input-dir`, `--output-dir` | `results/report/method_comparison.csv`, `oracle_deciles_*.csv`, `gate_sweep_*.csv` |

## Reproducibility

- All LLM calls use `temperature=0` and an optional SQLite response cache, so generation
  is deterministic and re-runnable without re-incurring API cost.
- Splits are randomized-but-seeded; every seed's split file and stratification report
  are committed.
- The FAISS index, feedback databases, and all downstream artifacts can be regenerated
  from the source CSV by running the numbered scripts in order.

---

## Contact

Jožef Stefan Institute, Dept. of Artificial Intelligence — HumAIne Horizon Europe project.