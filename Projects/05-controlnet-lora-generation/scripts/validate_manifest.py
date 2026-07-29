from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.contracts import validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired ControlNet data and split isolation.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    records = validate_manifest(args.manifest, args.root, verify_hashes=not args.skip_hash)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
    print(f"OK records={len(records)} splits={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
