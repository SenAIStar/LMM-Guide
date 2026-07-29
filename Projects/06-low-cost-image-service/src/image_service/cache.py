from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    image_bytes: bytes
    metadata: dict[str, Any]


class ContentAddressedCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _paths(self, tenant_id: str, cache_key: str) -> tuple[Path, Path]:
        if not tenant_id or any(ch in tenant_id for ch in "/\\"):
            raise ValueError("tenant_id must be a non-empty path-safe segment")
        if len(cache_key) != 64 or any(ch not in "0123456789abcdef" for ch in cache_key):
            raise ValueError("cache_key must be a lowercase SHA-256 digest")
        directory = self.root / tenant_id / cache_key[:2]
        return directory / f"{cache_key}.bin", directory / f"{cache_key}.json"

    def get(self, tenant_id: str, cache_key: str) -> CacheEntry | None:
        binary_path, metadata_path = self._paths(tenant_id, cache_key)
        if not binary_path.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return CacheEntry(binary_path.read_bytes(), metadata)

    def put(self, tenant_id: str, cache_key: str, image_bytes: bytes, metadata: dict[str, Any]) -> CacheEntry:
        binary_path, metadata_path = self._paths(tenant_id, cache_key)
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(binary_path, image_bytes)
        self._atomic_write(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
        )
        return CacheEntry(image_bytes, metadata)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

