from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controlnet_lora.contracts import select_split, validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a Diffusers ImageFolder for LoRA training.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = select_split(validate_manifest(args.manifest, args.root), "train")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata:
        for record in records:
            suffix = Path(record.image).suffix.lower()
            file_name = f"{record.sample_id}{suffix}"
            shutil.copy2(args.root / record.image, args.output / file_name)
            metadata.write(json.dumps({"file_name": file_name, "text": record.text}, ensure_ascii=False) + "\n")
    print(f"OK lora_train_records={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
