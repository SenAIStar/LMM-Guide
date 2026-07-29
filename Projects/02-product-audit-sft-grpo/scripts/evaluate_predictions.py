from __future__ import annotations

import argparse
import json

from product_audit.datasets import read_jsonl
from product_audit.evaluation import evaluate
from product_audit.policy import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--high-risk", nargs="*", default=["BANNED_CATEGORY", "CONTACT_DIVERSION"])
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    predictions = {row["sample_id"]: row for row in read_jsonl(args.predictions)}
    metrics = evaluate(gold, predictions, load_policy(args.policy), set(args.high_risk))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

