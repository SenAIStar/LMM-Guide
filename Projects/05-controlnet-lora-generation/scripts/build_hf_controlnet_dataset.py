from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.contracts import validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a typed paired-image dataset for official ControlNet training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-id", help="Optional private Hugging Face dataset repository id.")
    args = parser.parse_args()
    try:
        from datasets import Dataset, DatasetDict, Image
    except ImportError as exc:
        raise SystemExit("install requirements-ml.txt before building the Hugging Face dataset") from exc

    records = validate_manifest(args.manifest, args.root)
    split_rows: dict[str, list[dict[str, str]]] = {}
    for record in records:
        split_rows.setdefault(record.split, []).append(
            {
                "sample_id": record.sample_id,
                "image": str((args.root / record.image).resolve()),
                "conditioning_image": str((args.root / record.conditioning_image).resolve()),
                "text": record.text,
                "capture_group": record.capture_group,
                "license_id": record.license_id,
            }
        )
    dataset = DatasetDict()
    for split, rows in split_rows.items():
        item = Dataset.from_list(rows)
        item = item.cast_column("image", Image()).cast_column("conditioning_image", Image())
        dataset[split] = item
    dataset.save_to_disk(str(args.output))
    if args.hub_id:
        dataset.push_to_hub(args.hub_id, private=True)
    print(f"OK splits={list(dataset)} output={args.output} hub_id={args.hub_id or 'not_pushed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
