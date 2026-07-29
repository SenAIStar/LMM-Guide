from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SPLITS = {"train", "validation", "test"}
ALLOWED_CONDITIONS = {"canny", "depth", "pose", "segmentation"}


class ManifestError(ValueError):
    """Raised when a manifest does not satisfy the project data contract."""


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    image: str
    conditioning_image: str
    text: str
    subject_id: str
    capture_group: str
    license_id: str
    split: str
    condition_type: str
    sha256_image: str
    sha256_conditioning: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ManifestRecord":
        missing = [name for name in cls.__dataclass_fields__ if name not in value]
        if missing:
            raise ManifestError(f"missing fields: {', '.join(missing)}")
        record = cls(**{name: str(value[name]).strip() for name in cls.__dataclass_fields__})
        empty = [name for name, item in record.__dict__.items() if not item]
        if empty:
            raise ManifestError(f"empty fields: {', '.join(empty)}")
        if record.split not in ALLOWED_SPLITS:
            raise ManifestError(f"unsupported split: {record.split}")
        if record.condition_type not in ALLOWED_CONDITIONS:
            raise ManifestError(f"unsupported condition_type: {record.condition_type}")
        for name in ("sha256_image", "sha256_conditioning"):
            digest = getattr(record, name)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
                raise ManifestError(f"{name} must be a lowercase SHA-256 digest")
        return record


def _resolve_under(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ManifestError(f"path escapes dataset root: {relative}")
    return candidate


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pnm_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        if magic not in {b"P2", b"P3", b"P5", b"P6"}:
            return None
        tokens: list[bytes] = []
        while len(tokens) < 2:
            line = stream.readline()
            if not line:
                break
            line = line.split(b"#", 1)[0]
            tokens.extend(line.split())
        if len(tokens) < 2:
            raise ManifestError(f"invalid PNM header: {path}")
        return int(tokens[0]), int(tokens[1])


def image_size(path: Path) -> tuple[int, int]:
    pnm = _pnm_size(path)
    if pnm is not None:
        return pnm
    try:
        from PIL import Image
    except ImportError as exc:
        raise ManifestError(f"Pillow is required to inspect {path.suffix} files") from exc
    with Image.open(path) as image:
        return image.size


def load_manifest(path: Path) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(ManifestRecord.from_mapping(value))
            except (json.JSONDecodeError, ManifestError, TypeError) as exc:
                raise ManifestError(f"line {line_number}: {exc}") from exc
    if not records:
        raise ManifestError("manifest contains no records")
    return records


def validate_manifest(
    manifest_path: Path,
    dataset_root: Path,
    *,
    verify_hashes: bool = True,
) -> list[ManifestRecord]:
    records = load_manifest(manifest_path)
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = {}

    for record in records:
        if record.sample_id in sample_ids:
            raise ManifestError(f"duplicate sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        group_splits.setdefault(record.capture_group, set()).add(record.split)

        target = _resolve_under(dataset_root, record.image)
        condition = _resolve_under(dataset_root, record.conditioning_image)
        for label, path in (("image", target), ("conditioning_image", condition)):
            if not path.is_file():
                raise ManifestError(f"{record.sample_id}: missing {label}: {path}")

        if image_size(target) != image_size(condition):
            raise ManifestError(f"{record.sample_id}: target and condition dimensions differ")
        if verify_hashes:
            actual_target = file_sha256(target)
            actual_condition = file_sha256(condition)
            if actual_target != record.sha256_image:
                raise ManifestError(f"{record.sample_id}: target SHA-256 mismatch")
            if actual_condition != record.sha256_conditioning:
                raise ManifestError(f"{record.sample_id}: condition SHA-256 mismatch")

    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        details = ", ".join(f"{group}={sorted(splits)}" for group, splits in sorted(leaked.items()))
        raise ManifestError(f"capture_group leakage across splits: {details}")
    return records


def select_split(records: Iterable[ManifestRecord], split: str) -> list[ManifestRecord]:
    if split not in ALLOWED_SPLITS:
        raise ManifestError(f"unsupported split: {split}")
    return [record for record in records if record.split == split]
