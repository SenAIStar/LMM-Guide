from __future__ import annotations

import argparse
import json

from product_audit.datasets import read_jsonl
from product_audit.policy import load_policy
from product_audit.rewards import score_completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    scored = []
    for row in read_jsonl(args.input):
        result = score_completion(row["completion"], row["ground_truth"], policy, row["context"])
        scored.append({"sample_id": row["sample_id"], **result.__dict__})
    print(json.dumps(scored, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

