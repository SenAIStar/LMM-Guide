from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.contracts import AuditResult
from janus_audit.datasets import read_jsonl
from janus_audit.evaluation import evaluate
from janus_audit.policy import PolicyEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/sample/audit_examples.jsonl")
    parser.add_argument("--predictions", default="data/sample/predictions.jsonl")
    args = parser.parse_args()
    gold = [AuditResult.from_dict(row["target"]) for row in read_jsonl(ROOT / args.dataset)]
    predictions = read_jsonl(ROOT / args.predictions)
    for row in predictions:
        row.pop("_line", None)
    summary = evaluate(gold, predictions, PolicyEngine.from_path(ROOT / "configs/policy.json"))
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

