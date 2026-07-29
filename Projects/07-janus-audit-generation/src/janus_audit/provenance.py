from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_asset_provenance(
    path: str | Path,
    expected_sha256: str,
    license_id: str,
    policy_version: str,
) -> None:
    if not license_id.strip():
        raise ValueError("license_id is required")
    if not policy_version.strip():
        raise ValueError("policy_version is required")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"asset hash mismatch: expected={expected_sha256}, actual={actual}")

