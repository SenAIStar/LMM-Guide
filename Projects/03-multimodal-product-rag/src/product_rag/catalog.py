from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from .contracts import validate_answer
from .provenance import sha256_file


CHUNK_KEYS = {
    "chunk_id",
    "product_id",
    "variant_id",
    "tenant_id",
    "source_type",
    "source_revision",
    "effective_at",
    "expires_at",
    "acl",
    "title",
    "text",
    "facts",
    "media",
    "deleted",
}
QUERY_KEYS = {
    "query_id",
    "query_group",
    "text",
    "images",
    "tenant_id",
    "principals",
    "as_of",
    "product_scope",
    "filters",
    "relevant_chunk_ids",
    "answerable",
    "gold_output",
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


def parse_time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_chunk(chunk: dict[str, Any], root: str | Path) -> None:
    if set(chunk) != CHUNK_KEYS:
        raise ValueError(f"chunk keys mismatch: {chunk.get('chunk_id', '<unknown>')}")
    for name in (
        "chunk_id",
        "product_id",
        "variant_id",
        "tenant_id",
        "source_type",
        "source_revision",
        "title",
        "text",
    ):
        if not isinstance(chunk[name], str) or not chunk[name]:
            raise ValueError(f"{name} must be a non-empty string")
    parse_time(chunk["effective_at"])
    if chunk["expires_at"] is not None:
        parse_time(chunk["expires_at"])
    if not isinstance(chunk["acl"], list) or not all(isinstance(x, str) and x for x in chunk["acl"]):
        raise ValueError("acl must be a list of principals")
    if not isinstance(chunk["facts"], dict) or not isinstance(chunk["deleted"], bool):
        raise ValueError("facts or deleted has an invalid type")
    if not isinstance(chunk["media"], list):
        raise ValueError("media must be a list")
    for item in chunk["media"]:
        if set(item) != {"path", "sha256"}:
            raise ValueError("media item keys mismatch")
        path = Path(root) / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"media provenance check failed: {path}")


def validate_query(query: dict[str, Any], chunks: list[dict[str, Any]], root: str | Path) -> None:
    if set(query) != QUERY_KEYS:
        raise ValueError(f"query keys mismatch: {query.get('query_id', '<unknown>')}")
    for name in ("query_id", "query_group", "text", "tenant_id", "as_of"):
        if not isinstance(query[name], str) or not query[name]:
            raise ValueError(f"query {name} must be a non-empty string")
    parse_time(query["as_of"])
    if not isinstance(query["images"], list) or not isinstance(query["principals"], list):
        raise ValueError("query images and principals must be lists")
    for image in query["images"]:
        if not (Path(root) / image).is_file():
            raise ValueError(f"query image not found: {image}")
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    for chunk_id in query["relevant_chunk_ids"]:
        if chunk_id not in chunk_map:
            raise ValueError(f"unknown relevant chunk: {chunk_id}")
    validate_answer(query["gold_output"], [chunk_map[x] for x in query["relevant_chunk_ids"]])
    if bool(query["answerable"]) != (query["gold_output"]["answer_state"] != "not_found"):
        raise ValueError("answerable conflicts with gold_output")


def is_eligible(chunk: dict[str, Any], query: dict[str, Any]) -> bool:
    if chunk["deleted"] or chunk["tenant_id"] != query["tenant_id"]:
        return False
    principals = set(query["principals"]) | {"public"}
    if not principals.intersection(chunk["acl"]):
        return False
    as_of = parse_time(query["as_of"])
    if parse_time(chunk["effective_at"]) > as_of:
        return False
    if chunk["expires_at"] is not None and parse_time(chunk["expires_at"]) <= as_of:
        return False
    if query.get("product_scope") and chunk["product_id"] != query["product_scope"]:
        return False
    filters = query.get("filters") or {}
    return all(chunk["facts"].get(name) == value for name, value in filters.items())


def searchable_text(chunk: dict[str, Any]) -> str:
    facts = " ".join(f"{key} {value}" for key, value in sorted(chunk["facts"].items()))
    return f"{chunk['title']} {chunk['text']} {facts}".strip()
