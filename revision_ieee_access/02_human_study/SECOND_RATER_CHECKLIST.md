# Second rater — setup checklist (Timothy)

**Goal:** add the rater ID `timothy` to the running study without changing the
package, the sample, the answers or Alberto's ratings. The app keys ratings by
rater, so adding an ID is additive. Do **not** create a new package or a new DB.

**Status:** prepared 2026-11-25. Server access is not available from the current
environment; run these steps on Atena (or ask the administrator).

## 1. Before touching anything

```bash
ssh atena
tmux ls                      # expect a session named study
ps -ef | grep '[a]pp.py'     # confirms the running command and the user
```

Record the current package identity from the admin page or the DB:

```bash
sqlite3 ~/study/study.sqlite "select key, value from meta;"
```

The `package_id` must match the sanitized record in
`revision_ieee_access/02_human_study/packages/v1.json`. If it does not, stop.

## 2. Back up the database

```bash
mkdir -p ~/study/backups
cp ~/study/study.sqlite ~/study/backups/study_$(date +%Y%m%d_%H%M%S).sqlite
ls -lh ~/study/backups/ | tail -3
```

Also copy the backup off the server, or at least confirm it is readable:

```bash
sqlite3 ~/study/backups/study_*.sqlite "select rater, count(*) from ratings group by rater;"
```

Expected before the change: `alberto|105`.

## 3. Add the rater ID

Find how the app receives its environment. Use whichever applies.

**A. Environment file (preferred).** If the service uses `/etc/study.env` or
`~/study/.env`:

```bash
grep -n 'STUDY_RATERS' /etc/study.env ~/study/.env 2>/dev/null
sudo nano /etc/study.env        # or the file that exists
# set: STUDY_RATERS=alberto,timothy
```

Restart the service:

```bash
sudo systemctl restart study    # if systemd
# or, in the tmux session: Ctrl-C, then start the same command again
```

**B. tmux only.** If the environment is exported inside the `study` session:

```bash
tmux attach -t study
# Ctrl-C to stop the app
export STUDY_RATERS=alberto,timothy
# re-run the exact command from step 1, e.g.
python ~/study/app/app.py
# detach with Ctrl-B then D
```

Do not change `STUDY_PACKAGE`, `STUDY_DB`, `STUDY_ACCESS_CODE` or `SECRET_KEY`.
Do not edit `items.json`, `key.json` or `manifest.json`.

## 4. Verify

1. The app starts without the "package changed" error.
2. `alberto` still has 105 ratings:

```bash
sqlite3 ~/study/study.sqlite "select rater, count(*) from ratings group by rater;"
```

3. The progress endpoint or the browser shows `timothy` at 0/105.
4. Log in once as `timothy` with the shared access code, open the first item,
   and confirm that the page renders. Do not submit a rating for a real item
   during the check; if you do, delete it explicitly and re-check the count, or
   use a test item outside the package. Better: verify the login redirect and
   the progress page only.
5. Confirm the package ID is unchanged.

## 5. Friday session

- Window: Friday 11:00–13:00 CET.
- The package is answer-block only: 105 items, 35 per arm.
- The app saves after each submission; Timothy can stop and resume.
- Partial coverage is acceptable; agreement will use the common items.
- Do not discuss individual items with Alberto until both are finished.
- Keep the shared access code out of the repository and out of the thread email.

## 6. After the session

```bash
# Export through the admin endpoint, or copy the DB and query it.
sqlite3 -header -csv ~/study/study.sqlite \
  "select * from ratings where rater='timothy' order by created_utc;" \
  > ~/study/timothy_ratings.csv
```

Store the export with the package manifest; do not commit it. Then run:

```powershell
python experiments/33_human_eval.py --package data/processed/revision/study/v1 `
  --ratings revision_ieee_access/02_human_study/evaluation_alberto/ratings.csv `
  --ratings <path>/timothy_ratings.csv --label combined
```

The script supports partial overlap: it reports `n_common`, per-rater coverage,
percent agreement, Cohen's kappa with a bootstrap interval, weighted kappa, and
the combined per-item estimate on the common items. The existing 8 human-eval
tests cover the partial-overlap path.

## 7. If something goes wrong

- **Package changed error:** restore the DB backup and the original package; do
  not edit the package. A new package means a new study version.
- **Rater not recognised:** confirm the exact spelling `timothy` in
  `STUDY_RATERS` and restart; the login compares the string exactly.
- **Alberto's count changed:** stop, restore the backup, and investigate before
  the session. Adding a rater must not touch existing rows.
- **Access code lost:** retrieve it from the environment file; do not reset it
  during the session.
