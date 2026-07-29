from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janus_audit.datasets import read_jsonl, validate_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample/audit_examples.jsonl")
    args = parser.parse_args()
    records = read_jsonl(ROOT / args.input)
    issues = validate_records(records, ROOT)
    if issues:
        for issue in issues:
            print(f"line={issue.line} code={issue.code} detail={issue.detail}")
        return 1
    print(f"validated_records={len(records)} status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

