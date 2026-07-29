from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.contracts import AuditResult
from janus_audit.datasets import read_jsonl
from janus_audit.grpo import group_relative_advantages
from janus_audit.policy import PolicyEngine
from janus_audit.rewards import score_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/sample/audit_examples.jsonl")
    parser.add_argument("--rollouts", default="data/sample/rollouts.jsonl")
    parser.add_argument("--output", default="artifacts/grpo_scored.jsonl")
    args = parser.parse_args()
    gold = {
        row["asset_id"]: AuditResult.from_dict(row["target"])
        for row in read_jsonl(ROOT / args.dataset)
    }
    policy = PolicyEngine.from_path(ROOT / "configs/policy.json")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(ROOT / args.rollouts):
        prompt_id = str(row["prompt_id"])
        breakdown = score_candidate(row["candidate"], gold[prompt_id], policy)
        row["reward"] = breakdown.to_dict()
        groups[prompt_id].append(row)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for prompt_id, rows in groups.items():
            batch = group_relative_advantages([row["reward"]["total"] for row in rows])
            for row, advantage in zip(rows, batch.advantages):
                row["advantage"] = advantage
                row["group_zero_variance"] = batch.zero_variance
                row.pop("_line", None)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"groups={len(groups)} rows={sum(map(len, groups.values()))} path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

