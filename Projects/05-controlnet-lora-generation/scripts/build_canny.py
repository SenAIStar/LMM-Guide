from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.conditions import build_canny  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic Canny control images.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low", type=int, default=100)
    parser.add_argument("--high", type=int, default=200)
    args = parser.parse_args()
    sources = sorted(
        path for path in args.input_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not sources:
        raise SystemExit(f"no supported images under {args.input_dir}")
    for source in sources:
        relative = source.relative_to(args.input_dir).with_suffix(".png")
        build_canny(source, args.output_dir / relative, low=args.low, high=args.high)
    print(f"OK generated={len(sources)} low={args.low} high={args.high}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
