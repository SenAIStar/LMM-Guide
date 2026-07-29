from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from image_service import GenerationRequest  # noqa: E402


def validate_jsonl(path: Path) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    seen_request_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                request = GenerationRequest.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if request.request_id in seen_request_ids:
                raise ValueError(f"{path}:{line_number}: duplicate request_id {request.request_id}")
            seen_request_ids.add(request.request_id)
            requests.append(request)
    if not requests:
        raise ValueError(f"{path}: no requests found")
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generation request JSONL before a load test.")
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "data" / "sample" / "requests.jsonl")
    args = parser.parse_args()
    requests = validate_jsonl(args.path)
    print(f"valid_requests: {len(requests)}")
    print(f"distinct_generation_keys: {len({request.cache_key() for request in requests})}")


if __name__ == "__main__":
    main()

