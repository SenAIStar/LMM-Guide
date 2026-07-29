from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(group_id: str, seed: int = 42, buckets: int = 10_000) -> int:
    payload = f"{seed}:{group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % buckets


def grouped_split(group_id: str, seed: int = 42) -> str:
    bucket = stable_bucket(group_id, seed=seed)
    if bucket < 8_000:
        return "train"
    if bucket < 9_000:
        return "validation"
    return "test"

