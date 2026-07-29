import hashlib
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_media_files(record: dict[str, Any], data_root: Path) -> list[str]:
    errors: list[str] = []
    root = data_root.resolve()
    images = record.get("images", [])
    hashes = record.get("media_sha256", [])
    for index, relative in enumerate(images):
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root):
            errors.append(f"images[{index}] escapes data_root")
            continue
        if not path.is_file():
            errors.append(f"images[{index}] does not exist: {relative}")
            continue
        if index >= len(hashes):
            continue
        actual = sha256_file(path)
        if actual != hashes[index]:
            errors.append(f"images[{index}] SHA-256 mismatch")
    return errors
