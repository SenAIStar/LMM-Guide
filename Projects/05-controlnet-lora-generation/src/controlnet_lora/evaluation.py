from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable


Point = tuple[int, int]


@dataclass(frozen=True)
class EdgeMetrics:
    precision: float
    recall: float
    f1: float


def _within(point: Point, candidates: set[Point], tolerance: float) -> bool:
    return any(hypot(point[0] - other[0], point[1] - other[1]) <= tolerance for other in candidates)


def edge_metrics(predicted: Iterable[Point], reference: Iterable[Point], tolerance: float = 1.5) -> EdgeMetrics:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set and not reference_set:
        return EdgeMetrics(1.0, 1.0, 1.0)
    precision = (
        sum(_within(point, reference_set, tolerance) for point in predicted_set) / len(predicted_set)
        if predicted_set
        else 0.0
    )
    recall = (
        sum(_within(point, predicted_set, tolerance) for point in reference_set) / len(reference_set)
        if reference_set
        else 0.0
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return EdgeMetrics(precision, recall, f1)


def edge_points(path: str, threshold: int = 127) -> set[Point]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for file-based edge evaluation") from exc
    with Image.open(path).convert("L") as image:
        return {
            (x, y)
            for y in range(image.height)
            for x in range(image.width)
            if image.getpixel((x, y)) > threshold
        }
