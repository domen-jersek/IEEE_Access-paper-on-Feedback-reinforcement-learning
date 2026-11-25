# Generated ablation — plan and cost (R1-M2, R2-13)

**Status: not run. Requires approval for paid generation calls.**

The reviewers asked for generated-answer confirmation of the factor isolation, not
only the retrieval-level proxy. The revision currently has:

- a one-factor dev ablation at retrieval level (Table 3A), and
- generated answers for the frozen finalist (EB κ=2, intersection, absolute
  scaling) on all three data regimes (Table 2).

That is enough to show which configuration was selected and to confirm the
finalist, but it does not literally provide generated answers for EB + pool
scaling at Team and intersection, and for Laplace at Backoff. This plan closes
that gap on the strict panel, where the answer matters.

## Cells

On the prefix panel, dev split (312 queries), both protocols:

| Cell | Lift | Scope | Scaling | Purpose |
|---|---|---|---|---|
| 1 | EB κ=2 | Team∩Class | pool λ=1 | prior + scale at the finalist scope |
| 2 | EB κ=2 | Team | pool λ=1 | scope effect under the same prior and scale |
| 3 | Laplace | Backoff | absolute | fallback effect without EB |
| 4 | EB κ=2 | Team∩Class | absolute | finalist reference (already generated on eval; dev is new) |

Four cells × two protocols = **8 runs**. The baseline answer is identical across
cells and should be served from the generation cache; only the feedback answer is
new. The retrieval-level ablation already covers the duplicate panel; if the
editor requires generated duplicate-panel cells, repeat the same 8 runs with
`--panel disjoint_duplicate` (16 runs total).

## Commands

The entry point now accepts optional overrides while keeping the frozen defaults
byte-identical (verified: default config hash `a9d2bda65bb1` matches the frozen
`M4_intersection` runs).

```powershell
# Cell 1: EB + pool, intersection
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source conditioned --lift eb --kappa 2 --scope intersection `
  --scale-mode pool_std --pool-lambda 1 --tag ebpool_m4
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source blind --lift eb --kappa 2 --scope intersection `
  --scale-mode pool_std --pool-lambda 1 --tag ebpool_m4

# Cell 2: EB + pool, Team
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source conditioned --lift eb --kappa 2 --scope team `
  --scale-mode pool_std --pool-lambda 1 --tag ebpool_m2
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source blind --lift eb --kappa 2 --scope team `
  --scale-mode pool_std --pool-lambda 1 --tag ebpool_m2

# Cell 3: Laplace, Backoff
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source conditioned --lift laplace --scope backoff --tag lap_backoff
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source blind --lift laplace --scope backoff --tag lap_backoff

# Cell 4: EB absolute, intersection (finalist reference on dev)
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source conditioned --lift eb --kappa 2 --scope intersection `
  --scale-mode absolute --tag ebabs_m4
python experiments/23_revision_evaluate.py --panel disjoint_prefix --split dev `
  --source blind --lift eb --kappa 2 --scope intersection `
  --scale-mode absolute --tag ebabs_m4
```

Run the conditioned and blind commands sequentially or with a small concurrency
limit; each run already uses six concurrent requests.

## Cost

- 312 queries per run × 2 requests (baseline + feedback) = 624 requests per run.
- Baseline requests are shared across the four cells and should hit the cache;
  assume 312 new feedback requests per run plus cache misses for the first
  baseline, i.e. roughly **2,500–3,000 new requests for 8 runs**.
- At 0.48–0.87 s per request, **20–45 minutes of wall-clock time** on the same
  CPU setup, plus model latency.
- API cost depends on the current `openai/gpt-5.6-luna` price and the prompt
  length; it cannot be estimated reliably from the repository. The last nine
  generated panels (3,513 prompts) are the closest reference.

**Decision needed:** approve the 8 strict-panel dev runs (recommended), approve
both panels (16 runs), or keep the retrieval-level ablation and state the
limitation explicitly. If approved, the runs should be executed before the
manuscript rewrite so Table 3 can include generated cells.

## Analysis

The existing `experiments/26_generated_panels.py` reads the fixed
`M4_intersection_eval_{source}_rev_{panel}` folders and does not know the `_abl_`
suffix. Add a `--tag`/`--pattern` option or a small `37_generated_ablation.py`
that:

1. locates the eight `*_abl_*` details files;
2. aligns by `ticket_id` across cells and protocols;
3. computes paired Δ cosine with ticket-level bootstrap intervals;
4. writes one table with cell × protocol, plus the direct conditioned−blind
   contrast within each cell;
5. verifies that the baseline answers are identical across cells (cache
   integrity check).

The generated ablation is a dev selection experiment; the eval finalist stays as
reported in Table 2. State that distinction in the caption.
