# Provenance, models and artifacts (R1-M11, R2-14, R2-15)

Everything a reader needs to see what is reproducible, what is confidential, and
what was run when. Values are taken from the run manifests and the frozen records.

## Models

| Role | Identifier | Settings | Dates |
|---|---|---|---|
| Generator | `openai/gpt-5.6-luna` | temperature 0.0; system prompt SHA-256 `524098066b407de3`; regime `51fd9f4b37e900bb` | generation cache 2026-09-17 to 2026-09-22; revision runs 2026-11-12 |
| Feedback judge | `openai/gpt-5.6-luna` | temperature 0.0; protocol-conditioned and blind prompts; prompt v1 | judge cache 2026-09-12 to 2026-09-13 |
| Dense retriever | `all-MiniLM-L6-v2` | FAISS inner product, normalized embeddings | frozen index |
| Answer metric | `multi-qa-MiniLM-L6-cos-v1` | cosine to the recorded reference reply | frozen cache |
| Alternative metric | `BAAI/bge-base-en-v1.5` | cosine, independent embedding family | frozen cache |
| Alternative retriever | `BAAI/bge-small-en-v1.5` | FAISS inner product | frozen index |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | re-ranks the dense or hybrid pool | frozen cache |

The generator and judge share the model identifier in the submitted setup; they
differ in prompt and in what the judge sees. The SIKDD 2026 study used
`gpt-4o-mini`; this is a different judging regime and is not pooled with the
revision.

## Data and evaluation artifacts

| Artifact | Size / identity |
|---|---|
| Canonical dataset | 1,595 tickets, 37 teams, 17 intent classes; SHA-256 in `dataset_manifest.json` |
| Random split, seed 42 | 878 / 319 / 398 |
| Group-aware duplicate split | 878 / 319 / 398; 1,218 groups |
| Group-aware prefix split | 856 / 312 / 427; 676 groups |
| Feedback DBs | 175,600 rows each; same pair set; conditioned and blind |
| Allowed evidence, random | 94,913 rows per protocol after both-endpoint and self-pair filtering |
| Allowed evidence, duplicate | 52,321 rows per protocol |
| Allowed evidence, prefix | 58,826 rows per protocol |
| Frozen generated runs | 9 runs, 3,513 new prompts, `frozen_id 64c440e6…`, hash manifest `results/revision/frozen/frozen_runs_v1.json` |
| Study package | `v1`, 105 items, 35 per arm, `package_id a55aca29…`; pilot `pilot-v1`, 20 items |

## Parameters to report per cell

| Parameter | Value in the revision |
|---|---|
| Laplace α, β | 1.0, 1.0 |
| Legacy multiplier m, cap | 0.80, 0.20 |
| EB prior strength κ | 2.0 (finalist); 2, 5, 10, 20, 50 swept on dev |
| Pool scaling λ | 1.0 (finalist); 0.25, 0.5, 1, 2, 4 swept |
| Scope | Team ∩ Class (finalist); Team, Class, Global, Backoff in ablation |
| π_s estimation | mean judge score per scope over the allowed rows; global fallback |
| Top-k context | 5 |
| Candidate pool | 100 |
| Feedback pair set | one score per (query, candidate); both endpoints in training; self-pairs removed |

## What a third party can rerun

1. **From the public repository, without confidential data:** tables and figures from
   the frozen aggregate files; the human-evaluation notebook from the aggregate CSVs;
   the unit tests; the data-protocol validation on synthetic data.
2. **With the confidential corpus, on premises:** the full pipeline, retrieval
   panels, generated panels (if an API key is available), and the human study.
3. **With new API calls:** generation and judging can be repeated, but hosted models
   may drift; the revision pins the regime and hashes rather than claiming bit-level
   reproducibility.

## License and availability

The repository is public but has no license file; this is an open item (R1-M11).
Proposal for the authors to confirm:

- **Code:** MIT or Apache-2.0 (permissive, compatible with IEEE Access).
- **Documentation and aggregate tables:** CC BY 4.0.
- **Tickets, feedback pairs and generated answers:** not released; they contain
  confidential enterprise text. The paper should say exactly what is available and
  what requires the data agreement.

## Cost and latency

From the frozen runs (end-to-end, including model loading and metric computation):
611–749 requests per panel and source, 356–617 seconds per run, 0.48–0.87 seconds per
request. Two requests per ticket for baseline and feedback. Feedback-database
preparation is a separate one-off cost and is not included.
