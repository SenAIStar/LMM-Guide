from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import validate_audit_output
from .provenance import grouped_split, sha256_file


REQUIRED_RECORD_KEYS = {
    "sample_id",
    "product_id",
    "title",
    "category",
    "attributes",
    "ocr_tokens",
    "media",
    "policy_version",
    "ground_truth",
    "source",
    "license",
    "collected_at",
    "adjudication",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate_record(record: dict[str, Any], root: str | Path) -> None:
    if set(record) != REQUIRED_RECORD_KEYS:
        raise ValueError(f"record keys mismatch for {record.get('sample_id', '<unknown>')}")
    media = record["media"]
    if not isinstance(media, list) or not media:
        raise ValueError("media must contain at least one image")
    for item in media:
        if set(item) != {"path", "sha256"}:
            raise ValueError("media item keys mismatch")
        image_path = Path(root) / item["path"]
        if not image_path.is_file():
            raise ValueError(f"missing media file: {image_path}")
        if sha256_file(image_path) != item["sha256"]:
            raise ValueError(f"media hash mismatch: {image_path}")
    validate_audit_output(record["ground_truth"], image_count=len(media))
    adjudication = record["adjudication"]
    if not isinstance(adjudication, dict) or adjudication.get("status") != "resolved":
        raise ValueError("sample must have resolved adjudication")


def assign_splits(records: Iterable[dict[str, Any]], seed: int = 42) -> dict[str, str]:
    return {record["sample_id"]: grouped_split(record["product_id"], seed=seed) for record in records}


def assert_no_group_leakage(records: Iterable[dict[str, Any]], splits: dict[str, str]) -> None:
    group_to_split: dict[str, str] = {}
    for record in records:
        split = splits[record["sample_id"]]
        previous = group_to_split.setdefault(record["product_id"], split)
        if previous != split:
            raise ValueError(f"product group leaked across splits: {record['product_id']}")


def build_prompt(record: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for item in record["media"]:
        content.append({"type": "image", "image": item["path"]})
    request = {
        "title": record["title"],
        "category": record["category"],
        "attributes": record["attributes"],
        "ocr_tokens": record["ocr_tokens"],
        "policy_version": record["policy_version"],
    }
    content.append({"type": "text", "text": json.dumps(request, ensure_ascii=False)})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "审核商品并只输出 audit-output.v1 JSON。证据必须绑定图片序号与策略规则。",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def build_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    messages = build_prompt(record)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": json.dumps(record["ground_truth"], ensure_ascii=False)}
            ],
        }
    )
    return {
        "sample_id": record["sample_id"],
        "product_id": record["product_id"],
        "messages": messages,
        "images": [item["path"] for item in record["media"]],
    }


def build_grpo_row(record: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "prompt": build_prompt(record),
        "images": [item["path"] for item in record["media"]],
        "ground_truth": json.dumps(record["ground_truth"], ensure_ascii=False),
        "policy_json": json.dumps(policy, ensure_ascii=False),
        "context_json": json.dumps(
            {
                "title": record["title"],
                "category": record["category"],
                "attributes": record["attributes"],
                "ocr_tokens": record["ocr_tokens"],
                "image_count": len(record["media"]),
            },
            ensure_ascii=False,
        ),
    }

