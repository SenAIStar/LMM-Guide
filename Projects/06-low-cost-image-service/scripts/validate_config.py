from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_service.config import load_json, validate_service_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the service reference configuration.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "service.json")
    args = parser.parse_args()
    config = load_json(args.config)
    validate_service_config(config)
    print(f"valid: {args.config}")
    print(f"config_revision: {config['config_revision']}")
    print(f"result_status: {config['result_status']}")


if __name__ == "__main__":
    main()

