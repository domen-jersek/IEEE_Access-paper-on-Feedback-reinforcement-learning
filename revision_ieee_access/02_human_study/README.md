# Human evaluation — blinded two-expert study

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
  --panel disjoint_prefix `
  --sources conditioned blind reply_embed_bge `
  --version v1 `
  --sample 120
```

Output: `revision_ieee_access/02_human_study/<panel>/<version>/` with
`items.json` (served), `key.json` (private) and `manifest.json` (freeze record).
The exporter fails if the sources do not cover identical queries or if their
baseline answers differ, i.e. if the runs are not one comparable regime.

Sampling is stratified by whether the feedback changed the item content
(`--stratum answer` by default, or `retrieval` top-1). The default
`--changed-fraction 0.7` oversamples informative pairs, because many strict-panel
tickets keep an identical answer; the manifest records the full stratum sizes and
the allocation. If the sampled changed-share differs from the natural prevalence,
report it and either weight the estimate or state that the study measures the
effect conditional on a changed pair. Use `--changed-fraction` equal to the natural
share for a prevalence-preserving sample.

## Proposed main study

Two arms, 50 tickets each (100 tickets, 200 items per rater, about 7 hours each):

| Arm | Panel | Source | What it tests |
|---|---|---|---|
| A | `legacy_random` | `blind` | whether the measured blind harm (−0.044 cosine) corresponds to worse procedures |
| B | `disjoint_prefix` | `conditioned` | whether the remaining conditioned effect (+0.009) corresponds to better procedures |

Arm A is the strongest signal for the harm claim; arm B is the strictest test of
any benefit. Both experts rate the same package independently.

## Run locally

```powershell
pip install -r revision_ieee_access/02_human_study/requirements.txt
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
python experiments/25_analyze_study.py --ratings path/to/ratings.csv
```

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
