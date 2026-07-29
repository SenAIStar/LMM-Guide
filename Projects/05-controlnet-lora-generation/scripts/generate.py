from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.inference import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SDXL + ControlNet + LoRA with a recorded config fingerprint.")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.json"))
    parser.add_argument("--control-image", type=Path, required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = generate(
        config_path=args.config,
        control_image_path=args.control_image,
        lora_path=args.lora,
        output_path=args.output,
    )
    print(f"OK output={args.output} config_hash={record['config_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
