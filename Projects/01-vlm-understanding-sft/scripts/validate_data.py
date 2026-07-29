import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_product.contracts import validate_training_record  # noqa: E402
from vlm_product.data_pipeline import assert_no_group_leakage, assert_no_media_leakage  # noqa: E402
from vlm_product.provenance import validate_media_files  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root used for image paths; defaults to the parent of the JSONL split directory.",
    )
    args = parser.parse_args()
    records = read_jsonl(args.path)
    data_root = args.data_root or args.path.resolve().parents[1]
    failures = 0
    for line_number, record in enumerate(records, 1):
        errors = validate_training_record(record)
        errors.extend(validate_media_files(record, data_root))
        if errors:
            failures += 1
            print(f"line {line_number}: {'; '.join(errors)}")
    for check in (assert_no_group_leakage, assert_no_media_leakage):
        try:
            check(records)
        except ValueError as exc:
            failures += 1
            print(f"dataset: {exc}")
    print(f"records={len(records)} validation_failures={failures}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
