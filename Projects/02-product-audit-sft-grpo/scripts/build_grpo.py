from __future__ import annotations

import argparse
import json

from product_audit.datasets import build_grpo_row, read_jsonl, write_jsonl
from product_audit.policy import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    rows = [build_grpo_row(record, policy) for record in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()

