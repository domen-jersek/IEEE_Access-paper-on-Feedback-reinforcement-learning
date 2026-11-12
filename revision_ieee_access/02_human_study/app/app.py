"""Blinded two-expert study application (Flask + SQLite).

Reads an immutable study package (items.json, key.json, manifest.json) and stores
ratings in SQLite. Method identity stays server-side in key.json and is never sent
to the browser. One submission per (rater, item); no editing after submission.

Environment:
    STUDY_PACKAGE     directory with items.json / key.json / manifest.json
    STUDY_RATERS      comma-separated allowed rater IDs (e.g. expert1,expert2)
    STUDY_ACCESS_CODE shared access code
    STUDY_ADMIN_CODE  code for /admin/export.csv and /admin/progress
    STUDY_DB          SQLite path (default: <package>/ratings.sqlite)
    SECRET_KEY        Flask session secret (set in production)
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, abort, redirect, render_template, request, session, url_for

CORRECTNESS = ("correct", "partially_correct", "incorrect", "cannot_judge")
PREFERENCE = ("A", "B", "equal", "both_incorrect", "cannot_judge")
MAX_COMMENT = 2000


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    if not value and name not in {"STUDY_DB", "SECRET_KEY"}:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def _package_id(package: Path) -> str:
    digest = hashlib.sha256()
    for name in ("items.json", "key.json", "manifest.json"):
        digest.update((package / name).read_bytes())
    return digest.hexdigest()


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(32)

    package = Path(_env("STUDY_PACKAGE")).resolve()
    items = json.loads((package / "items.json").read_text(encoding="utf-8"))
    key = json.loads((package / "key.json").read_text(encoding="utf-8"))
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    item_by_id = {item["item_id"]: item for item in items}
    raters = {name.strip() for name in _env("STUDY_RATERS").split(",") if name.strip()}
    access_code = _env("STUDY_ACCESS_CODE")
    admin_code = _env("STUDY_ADMIN_CODE")
    package_id = _package_id(package)

    db_path = Path(os.environ.get("STUDY_DB") or (package / "ratings.sqlite"))
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ratings (
            rater TEXT NOT NULL, item_id TEXT NOT NULL,
            a_correct TEXT NOT NULL, b_correct TEXT NOT NULL,
            preference TEXT NOT NULL, comment TEXT NOT NULL DEFAULT '',
            created_utc TEXT NOT NULL, package_id TEXT NOT NULL,
            PRIMARY KEY (rater, item_id)
        );
        """
    )
    existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(ratings)")}
    for name in ("a_unsafe", "b_unsafe", "a_missing", "b_missing"):
        if name not in existing_columns:
            connection.execute(f"ALTER TABLE ratings ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0")
    connection.commit()
    row = connection.execute("SELECT value FROM meta WHERE key='package_id'").fetchone()
    if row is None:
        connection.execute("INSERT INTO meta(key, value) VALUES ('package_id', ?)", (package_id,))
        connection.commit()
    elif row["value"] != package_id:
        raise RuntimeError("Study package changed after ratings were recorded; refusing to start")

    def order_for(rater: str) -> list[dict]:
        def order(block: str):
            subset = [item for item in items if item["block"] == block]
            return sorted(subset, key=lambda item: hashlib.sha256(
                f"{rater}|{item['item_id']}".encode()).hexdigest())
        return order("answer") + order("retrieval")

    def rated_ids(rater: str) -> set[str]:
        return {row["item_id"] for row in connection.execute(
            "SELECT item_id FROM ratings WHERE rater=?", (rater,))}

    def progress(rater: str) -> dict:
        done = len(rated_ids(rater))
        return {"done": done, "total": len(items), "remaining": len(items) - done}

    def current_rater() -> str | None:
        rater = session.get("rater")
        return rater if rater in raters else None

    def require_rater() -> str:
        rater = current_rater()
        if rater is None:
            return redirect(url_for("home"))
        return rater

    def next_item(rater: str):
        rated = rated_ids(rater)
        for item in order_for(rater):
            if item["item_id"] not in rated:
                return item
        return None

    @app.get("/")
    def home():
        if current_rater():
            return redirect(url_for("study"))
        return render_template("home.html", error=None, manifest=manifest)

    @app.post("/login")
    def login():
        rater = request.form.get("rater", "").strip()
        code = request.form.get("code", "")
        if rater not in raters or not hmac.compare_digest(code, access_code):
            return render_template("home.html", error="Unknown rater or wrong access code.",
                                   manifest=manifest), 401
        session.clear()
        session["rater"] = rater
        return redirect(url_for("study"))

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.get("/study")
    def study():
        rater = require_rater()
        if not isinstance(rater, str):
            return rater
        item = next_item(rater)
        if item is None:
            return render_template("done.html", progress=progress(rater), rater=rater)
        return redirect(url_for("item", item_id=item["item_id"]))

    @app.route("/item/<item_id>", methods=["GET", "POST"])
    def item(item_id: str):
        rater = require_rater()
        if not isinstance(rater, str):
            return rater
        entry = item_by_id.get(item_id)
        if entry is None:
            abort(404)

        existing = connection.execute(
            "SELECT * FROM ratings WHERE rater=? AND item_id=?", (rater, item_id)).fetchone()

        if request.method == "POST":
            if existing is not None:
                return redirect(url_for("item", item_id=item_id))
            a_correct = request.form.get("a_correct", "")
            b_correct = request.form.get("b_correct", "")
            preference = request.form.get("preference", "")
            comment = request.form.get("comment", "").strip()[:MAX_COMMENT]
            if a_correct not in CORRECTNESS or b_correct not in CORRECTNESS \
                    or preference not in PREFERENCE:
                return render_template(
                    "item.html", entry=entry, rater=rater, progress=progress(rater),
                    existing=None, error="Please answer every question before submitting.",
                    correctness=CORRECTNESS, preference=PREFERENCE), 400
            connection.execute(
                "INSERT OR IGNORE INTO ratings "
                "(rater, item_id, a_correct, b_correct, preference, comment, created_utc, package_id, "
                " a_unsafe, b_unsafe, a_missing, b_missing) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rater, item_id, a_correct, b_correct, preference, comment,
                 datetime.now(timezone.utc).isoformat(), package_id,
                 1 if request.form.get("a_unsafe") else 0,
                 1 if request.form.get("b_unsafe") else 0,
                 1 if request.form.get("a_missing") else 0,
                 1 if request.form.get("b_missing") else 0),
            )
            connection.commit()
            return redirect(url_for("study"))

        return render_template(
            "item.html", entry=entry, rater=rater, progress=progress(rater),
            existing=existing, error=None, correctness=CORRECTNESS, preference=PREFERENCE)

    @app.get("/admin/progress")
    def admin_progress():
        if not hmac.compare_digest(request.args.get("code", ""), admin_code):
            abort(403)
        rows = []
        for rater in sorted(raters):
            stats = progress(rater)
            stats["rater"] = rater
            rows.append(stats)
        return render_template("admin.html", rows=rows, manifest=manifest)

    @app.get("/admin/export.csv")
    def admin_export():
        if not hmac.compare_digest(request.args.get("code", ""), admin_code):
            abort(403)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["rater", "item_id", "panel", "ticket_id", "source", "block",
                         "a_correct", "b_correct", "preference", "a_unsafe", "b_unsafe",
                         "a_missing", "b_missing", "comment",
                         "created_utc", "package_id", "a_side", "b_side"])
        query = connection.execute("SELECT * FROM ratings ORDER BY rater, created_utc")
        for row in query:
            hidden = key.get(row["item_id"], {})
            writer.writerow([row["rater"], row["item_id"], hidden.get("panel"),
                             hidden.get("ticket_id"), hidden.get("source"), hidden.get("block"),
                             row["a_correct"], row["b_correct"], row["preference"],
                             row["a_unsafe"], row["b_unsafe"], row["a_missing"], row["b_missing"],
                             row["comment"], row["created_utc"], row["package_id"],
                             hidden.get("A"), hidden.get("B")])
        return Response(buffer.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=ratings.csv"})

    return app


if __name__ == "__main__":
    host = os.environ.get("STUDY_HOST", "127.0.0.1")
    port = int(os.environ.get("STUDY_PORT", "8050"))
    application = create_app()
    try:
        from waitress import serve
    except ImportError:  # pragma: no cover
        application.run(host=host, port=port, debug=False)
    else:
        serve(application, host=host, port=port, threads=8)
