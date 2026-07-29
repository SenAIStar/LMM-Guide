from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.evaluation import edge_metrics, edge_points  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Canny adherence with pixel tolerance.")
    parser.add_argument("--prediction-edge", required=True)
    parser.add_argument("--reference-edge", required=True)
    parser.add_argument("--tolerance", type=float, default=1.5)
    args = parser.parse_args()
    metrics = edge_metrics(
        edge_points(args.prediction_edge),
        edge_points(args.reference_edge),
        tolerance=args.tolerance,
    )
    print(f"precision={metrics.precision:.6f} recall={metrics.recall:.6f} f1={metrics.f1:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
