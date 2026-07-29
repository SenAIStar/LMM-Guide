from __future__ import annotations

from pathlib import Path


def build_canny(source: Path, destination: Path, *, low: int = 100, high: int = 200) -> None:
    if not 0 <= low < high <= 255:
        raise ValueError("expected 0 <= low < high <= 255")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to build Canny conditions") from exc

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {source}")
    edges = cv2.Canny(image, low, high)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), edges):
        raise OSError(f"unable to write condition image: {destination}")
