from __future__ import annotations

import argparse
import json
from pathlib import Path

from product_audit.datasets import assert_no_group_leakage, assign_splits, read_jsonl, validate_record
from product_audit.policy import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    records = read_jsonl(args.input)
    for record in records:
        validate_record(record, args.root)
        if record["policy_version"] != policy["policy_version"]:
            raise ValueError(f"policy version mismatch: {record['sample_id']}")
    splits = assign_splits(records, seed=args.seed)
    assert_no_group_leakage(records, splits)
    counts = {split: list(splits.values()).count(split) for split in sorted(set(splits.values()))}
    print(json.dumps({"valid_records": len(records), "split_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()

