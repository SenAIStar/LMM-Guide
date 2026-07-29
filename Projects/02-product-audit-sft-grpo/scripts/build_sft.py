from __future__ import annotations

import argparse
import json

from product_audit.datasets import build_sft_row, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [build_sft_row(record) for record in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()

