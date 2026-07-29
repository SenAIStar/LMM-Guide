from collections import Counter
from typing import Any, Iterable

from .contracts import validate_prediction


def normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    if isinstance(value, list):
        return sorted(normalize_value(item) for item in value)
    return value


def get_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_known(value: Any) -> bool:
    return normalize_value(value) not in (None, "unknown", [], ["unknown"])


def evaluate_records(
    gold_records: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = {str(item["sample_id"]): item for item in gold_records}
    pred_by_id = {str(item["sample_id"]): item["prediction"] for item in predictions}
    counts = Counter()
    field_hits = Counter()
    field_totals = Counter()
    unsupported = 0
    predicted_attributes = 0
    evidence_hits = 0
    evidence_required = 0

    for sample_id, gold in gold_by_id.items():
        counts["samples"] += 1
        prediction = pred_by_id.get(sample_id)
        if prediction is None:
            counts["missing_predictions"] += 1
            continue
        allowed_media = set(gold.get("allowed_evidence_media", []))
        media_count = max(allowed_media, default=0) + 1
        if not validate_prediction(prediction, media_count):
            counts["schema_valid"] += 1
        counts[f"decision_{prediction.get('decision', 'missing')}"] += 1

        observable = gold.get("observable_fields", {})
        for field, expected in observable.items():
            actual = get_path(prediction, field)
            field_totals[field] += 1
            if normalize_value(actual) == normalize_value(expected):
                field_hits[field] += 1
            if field.startswith("attributes.") and _is_known(actual):
                predicted_attributes += 1
                if not _is_known(expected):
                    unsupported += 1

        evidence_by_field: dict[str, set[int]] = {}
        for item in prediction.get("evidence", []):
            evidence_by_field.setdefault(str(item.get("field")), set()).add(item.get("media_index"))
        for field, actual in ((field, get_path(prediction, field)) for field in observable):
            if not _is_known(actual):
                continue
            evidence_required += 1
            if evidence_by_field.get(field, set()) & allowed_media:
                evidence_hits += 1

    samples = counts["samples"]
    predicted = samples - counts["missing_predictions"]
    return {
        "sample_count": samples,
        "missing_prediction_count": counts["missing_predictions"],
        "schema_valid_rate": counts["schema_valid"] / samples if samples else 0.0,
        "field_exact_match": {
            field: field_hits[field] / total for field, total in sorted(field_totals.items())
        },
        "unsupported_attribute_rate": unsupported / predicted_attributes if predicted_attributes else 0.0,
        "evidence_coverage": evidence_hits / evidence_required if evidence_required else 0.0,
        "manual_review_rate": counts["decision_review"] / predicted if predicted else 0.0,
    }
