from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_service.capacity import estimate_capacity  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a measured backend report into a capacity estimate.")
    parser.add_argument("report", type=Path)
    parser.add_argument("--utilization-target", type=float, default=0.70)
    parser.add_argument("--gpu-hour-price", type=float)
    parser.add_argument("--allow-simulated", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("result_status") != "measured" and not args.allow_simulated:
        raise SystemExit("refusing to publish capacity from a non-measured report; use --allow-simulated only for debugging")
    estimate = estimate_capacity(
        accepted_images=int(report["accepted_output_count"]),
        aggregate_gpu_seconds=float(report["gpu_seconds_all_attempts"]),
        p95_end_to_end_ms=float(report["end_to_end_p95_ms"]),
        utilization_target=args.utilization_target,
        gpu_hour_price=args.gpu_hour_price,
    )
    output = asdict(estimate)
    output["source_result_status"] = report.get("result_status", "unknown")
    output["utilization_target"] = args.utilization_target
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

