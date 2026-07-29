import hashlib
from typing import Any, Iterable


def stable_group_split(
    group_id: str,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    salt: str = "vlm-product-v1",
) -> str:
    if not group_id:
        raise ValueError("group_id must be non-empty")
    if not 0 < train_ratio < 1 or not 0 <= eval_ratio < 1 or train_ratio + eval_ratio >= 1:
        raise ValueError("invalid split ratios")
    payload = f"{salt}:{group_id}".encode("utf-8")
    bucket = int(hashlib.sha256(payload).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + eval_ratio:
        return "eval"
    return "test"


def localized_values(values: Any, language: str = "en_US") -> list[str]:
    if not isinstance(values, list):
        return []
    preferred = [item.get("value") for item in values if isinstance(item, dict) and item.get("language_tag") == language]
    fallback = [item.get("value") for item in values if isinstance(item, dict) and item.get("value")]
    chosen = preferred or fallback
    return [str(value).strip() for value in chosen if str(value).strip()]


def abo_candidate(
    listing: dict[str, Any],
    image_paths: Iterable[str],
    snapshot_id: str,
    license_id: str,
    language: str = "en_US",
) -> dict[str, Any]:
    """Build a human-review candidate; listing metadata is not visual ground truth."""
    item_id = str(listing.get("item_id", "")).strip()
    paths = [str(path) for path in image_paths if str(path)]
    if not item_id or not paths:
        raise ValueError("listing needs item_id and at least one image")
    if not snapshot_id or not license_id:
        raise ValueError("snapshot_id and license_id are required")
    product_types = [
        str(item.get("value"))
        for item in listing.get("product_type", [])
        if isinstance(item, dict) and item.get("value")
    ]
    return {
        "sample_id": f"abo_{item_id}",
        "group_id": item_id,
        "images": paths,
        "candidate_labels": {
            "product_type": product_types[0] if product_types else "unknown",
            "color": localized_values(listing.get("color"), language),
            "material": localized_values(listing.get("material"), language),
        },
        "source": {
            "dataset": "amazon_berkeley_objects",
            "snapshot_id": snapshot_id,
            "license_id": license_id,
            "source_uri": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
        },
        "label_source": "abo_listing_metadata_candidate_only",
        "review_required": True,
        "split": stable_group_split(item_id),
    }


def assert_no_group_leakage(records: Iterable[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        group_id = str(record.get("group_id", ""))
        split = str(record.get("split", ""))
        previous = seen.setdefault(group_id, split)
        if previous != split:
            raise ValueError(f"group {group_id} appears in both {previous} and {split}")


def assert_no_media_leakage(records: Iterable[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        split = str(record.get("split", ""))
        for media_hash in record.get("media_sha256", []):
            previous = seen.setdefault(str(media_hash), split)
            if previous != split:
                raise ValueError(f"media {media_hash} appears in both {previous} and {split}")
