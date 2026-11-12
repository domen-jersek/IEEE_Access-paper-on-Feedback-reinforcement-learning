# Deployment on an internal server (Atena)

The application is a small Flask service. It reads the frozen study package and
writes ratings to SQLite. Ticket text stays on the server; the browser only
receives the items of the current page.

## 1. Layout

```
/srv/study/
  app/                 # revision_ieee_access/02_human_study/app
  package/v1/          # items.json, key.json, manifest.json
  study.sqlite         # created on first start
```

`key.json` must be readable only by the service user:

```bash
chmod 600 package/v1/key.json
```

Do not put the package under a directory served by the web server.

## 2. Environment

`/etc/study.env` (mode 600, owned by root):

```
STUDY_PACKAGE=/srv/study/package/v1
STUDY_RATERS=expert1,expert2
STUDY_ACCESS_CODE=<shared code for the two experts>
STUDY_ADMIN_CODE=<coordinator code for export>
STUDY_DB=/srv/study/study.sqlite
SECRET_KEY=<long random string>
STUDY_HOST=127.0.0.1
STUDY_PORT=8050
```

## 3. Install and run

```bash
python3 -m venv /srv/study/venv
/srv/study/venv/bin/pip install -r /srv/study/app/requirements.txt
```

`/etc/systemd/system/study.service`:

```ini
[Unit]
Description=Blinded expert study
After=network.target

[Service]
User=study
WorkingDirectory=/srv/study
EnvironmentFile=/etc/study.env
ExecStart=/srv/study/venv/bin/python /srv/study/app/app.py
Restart=on-failure
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now study
```

Put an internal reverse proxy with TLS in front of `127.0.0.1:8050`. Do not expose
the port directly to the internet.

## 4. Access

- Experts open the internal URL and sign in with their rater ID and the shared
  access code.
- The coordinator uses `/admin/progress?code=...` and
  `/admin/export.csv?code=...`.
- The application refuses to start if the package content changes while the
  database already holds ratings from another package.

## 5. Backup and audit

- Back up `study.sqlite` daily while the study runs; it contains only ratings.
- Keep `manifest.json` with the paper submission: it identifies the exact runs,
  hashes and assignment seed.
- After both experts finish, export `ratings.csv`, archive it with the manifest,
  and stop the service.

## 6. A new study version

Create a new package with `--version v2`, use a **new** `STUDY_DB`, and repeat the
access setup. Never edit a package in place or merge ratings from two versions.
