# Human evaluation — blinded two-expert study

**Start here:** [`HUMAN_EVAL_TLDR.md`](HUMAN_EVAL_TLDR.md) for a self-contained
summary, then [`../../notebooks/09_human_evaluation.ipynb`](../../notebooks/09_human_evaluation.ipynb)
for the executed analysis and figures. Full tables are in
[`HUMAN_EVAL_RESULTS.md`](HUMAN_EVAL_RESULTS.md).

Two GFT support experts rate the same frozen items independently. The application
never reveals which side is baseline and which is feedback, nor which evidence
source or data regime produced an item.

## Design

- **Two blocks per ticket.** Block 1 shows the two generated first replies only
  (no retrieval, to avoid anchoring). Block 2 shows the two retrieved historical
  ticket sets. A/B order is randomised independently per block and stays fixed.
- **Independent correctness first.** Each side is rated
  `correct / partially correct / incorrect / cannot judge`. Two flags follow each
  side: a wrong or unsafe step, and a missing required step (for retrieval: a
  misleading ticket, and a missing ticket needed for the procedure).
- **Pairwise preference second.** `A better / B better / equal / both incorrect /
  cannot judge`. “Both incorrect” exists so a preference cannot hide two wrong
  answers.
- **Blinding.** Item IDs are opaque hashes. `items.json` contains no method, panel
  or ticket identifier; the mapping is in `key.json`, which stays on the server and
  is joined only in the admin CSV export.
- **Immutability.** One submission per (rater, item); no editing. The package hash
  is stored in the database and the app refuses to start if the package changes.

## Build the study package (after the generated panel is frozen)

```powershell
python experiments/24_export_study.py `
  --arm legacy_random:blind:35 `
  --arm disjoint_prefix:blind:35 `
  --arm disjoint_prefix:conditioned:35 `
  --version v1
```

Output: `<out-root>/<version>/` (default `data/processed/revision/study/`, git-ignored)
with `items.json` (served), `key.json` (private) and `manifest.json` (freeze record).
Never write the package into a tracked directory: it contains ticket text.

The committed, sanitized provenance records are in
`revision_ieee_access/02_human_study/packages/` (`v1.json`, `pilot-v1.json`).

Two practice items are generated separately:

```powershell
python experiments/31_build_practice.py --package data/processed/revision/study/v1
```
The exporter fails if the sources do not cover identical queries or if their
baseline answers differ, i.e. if the runs are not one comparable regime.

Sampling is **pre-registered** in `STUDY_DESIGN.md`: only tickets whose feedback
answer differs from the baseline are eligible; the sample is stratified by
resolving team with proportional allocation and a deterministic hash order. It is
never selected by the sign or size of the automatic delta. Tickets whose baseline
answer differs across sources of the same panel (hosted-model nondeterminism) are
excluded; the manifest records the count and each arm's changed share.

## Main study

Three arms, 35 tickets each (105 tickets, one block per ticket, about 2.5–3 hours
per expert):

| Arm | Panel | Source | Automatic Δ | What it tests |
|---|---|---|---:|---|
| A | `legacy_random` | `blind` | −0.044 | harm under procedure overlap |
| B | `disjoint_prefix` | `blind` | +0.001 | neutral control |
| C | `disjoint_prefix` | `conditioned` | +0.009 | small benefit |

Two numbers are reported: the effect among changed pairs (what the experts see) and
the population effect, obtained by multiplying it by the arm's changed share. This
is stated in the design before any rating; it avoids overstating the deployment
effect by oversampling informative pairs.

A one-hour pilot uses 20 tickets from arm A and is reported as a pilot, not as the
main study.

## Run locally

```powershell
pip install -r revision_ieee_access/02_human_study/app/requirements.txt
$env:STUDY_PACKAGE="revision_ieee_access/02_human_study/<panel>/<version>"
$env:STUDY_RATERS="expert1,expert2"
$env:STUDY_ACCESS_CODE="..."      # give to the two experts
$env:STUDY_ADMIN_CODE="..."       # coordinator only
$env:STUDY_DB="<package>/ratings.sqlite"
$env:SECRET_KEY="..."             # long random string
python revision_ieee_access/02_human_study/app/app.py
```

Open `http://127.0.0.1:8050`. Admin pages:
`/admin/progress?code=...` and `/admin/export.csv?code=...`.

## Deploy on Atena

1. Copy this directory and the frozen package; keep `key.json` outside any web root.
2. Install `requirements.txt` in a virtual environment.
3. Set the environment variables above (systemd unit or container secrets). Set
   `STUDY_HOST=0.0.0.0` behind the internal reverse proxy; terminate TLS there.
4. Run `python revision_ieee_access/02_human_study/app/app.py` under systemd, or
   `waitress-serve --call app:create_app` with `PYTHONPATH` pointing at `app/`.
5. Restrict access to the internal network. Do not index or log ticket text.

## Freeze and analysis

- The package is created **after** the generated runs are final. Any change to
  retrieval, answers, sample or A/B assignment requires a new `--version`; the app
  refuses to mix packages that share a database.
- Both experts must rate the same version. The manifest records run directories,
  details-file hashes, config hash, dataset hash and the assignment seed.
- Export `ratings.csv` after both finish. It contains the hidden side mapping
  (`a_side`, `b_side`), so it enables agreement analysis without trusting the app.

## Analysis

```powershell
python experiments/25_analyze_study.py --ratings path/to/ratings.csv --manifest path/to/manifest.json
```

Legacy summary script; the revision pipeline is `experiments/33_human_eval.py`
(per-rater effects, length control, metric alignment, and two-rater agreement).
Start with [`HUMAN_EVAL_TLDR.md`](HUMAN_EVAL_TLDR.md); full results are in
[`HUMAN_EVAL_RESULTS.md`](HUMAN_EVAL_RESULTS.md); aggregates are in
`results/alberto/`. Raw ratings stay in the private `evaluation_*/` folder
(git-ignored).

Writes `study_summary.csv` (per panel × source × block: strict correctness rates,
paired strict delta with bootstrap CI, preference shares, net feedback-better),
`study_agreement.csv` (percent agreement and Cohen's kappa for preference and for
the strict-correctness pair) and `study_analysis.json`. Ties, both-incorrect and
cannot-judge are reported separately.

## What the study can and cannot show

It measures procedural correctness of first replies and the usefulness of the
retrieved tickets for two experts. It does not validate the simulated LLM feedback
as real user feedback, and two experts do not establish population-level
generalisation. Report per-rater results, agreement, ties and “both incorrect”
rates explicitly.
