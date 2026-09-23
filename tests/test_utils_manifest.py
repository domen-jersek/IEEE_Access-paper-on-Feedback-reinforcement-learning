"""Manifest writing must survive OS-level introspection failures (WMI timeouts)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments.utils as utils


def _boom(*args, **kwargs):
    raise OSError("WMI timeout")


def test_write_run_manifest_survives_platform_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.platform, "platform", _boom)
    path = utils.write_run_manifest(tmp_path, script="tests/fake.py", extra={"x": 1}, run_id="testrun")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(manifest["platform"], str) and manifest["platform"]
    assert manifest["extra"] == {"x": 1}


def test_write_run_manifest_survives_hostname_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.socket, "gethostname", _boom)
    path = utils.write_run_manifest(tmp_path, script="tests/fake.py", run_id="testrun2")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["hostname"] == ""
