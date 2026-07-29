from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.datasets import read_jsonl, to_sft_conversation, validate_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/audit_examples.jsonl")
    parser.add_argument("--output", default="artifacts/sft_train.jsonl")
    args = parser.parse_args()
    records = read_jsonl(ROOT / args.input)
    issues = validate_records(records, ROOT)
    if issues:
        raise SystemExit("dataset validation failed; run scripts/validate_data.py")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record.pop("_line", None)
            handle.write(json.dumps(to_sft_conversation(record), ensure_ascii=False) + "\n")
    print(f"wrote={len(records)} path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

