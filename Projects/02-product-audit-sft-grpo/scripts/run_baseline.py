from __future__ import annotations

import argparse
import json

from product_audit.datasets import read_jsonl
from product_audit.policy import evaluate_hard_policy, load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    policy = load_policy(args.policy)
    rows = []
    for record in read_jsonl(args.input):
        context = {key: record[key] for key in ("title", "category", "attributes", "ocr_tokens")}
        result = evaluate_hard_policy(context, policy)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "decision": result.decision,
                "rule_ids": [hit.rule_id for hit in result.hits],
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

