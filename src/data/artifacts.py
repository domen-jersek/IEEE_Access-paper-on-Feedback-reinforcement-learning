"""Content identities and write-once JSON artifacts (no model/API imports)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def content_id(value: object) -> str:
    """Full SHA-256 of deterministic JSON; callers canonicalize unordered lists."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactIdentity:
    path: str
    sha256: str
    size_bytes: int

    @classmethod
    def verify(cls, path: Path, expected_sha256: str) -> ArtifactIdentity:
        path = Path(path).resolve()
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError(f"Input hash mismatch: {path.name}; expected {expected_sha256}, got {actual}")
        return cls(str(path), actual, path.stat().st_size)

    def to_dict(self) -> dict:
        return asdict(self)


def write_once_json(path: Path, payload: object) -> None:
    """Never overwrite an existing artifact, even when the payload is identical."""
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
