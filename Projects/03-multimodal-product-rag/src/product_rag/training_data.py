from __future__ import annotations

import json
from typing import Any

from .generator import build_messages


def contexts_for_query(query: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
    return [chunk_map[chunk_id] for chunk_id in query["relevant_chunk_ids"]]


def build_sft_row(query: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = contexts_for_query(query, chunks)
    messages = build_messages(query, contexts)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": json.dumps(query["gold_output"], ensure_ascii=False)}
            ],
        }
    )
    images = list(query["images"])
    images.extend(media["path"] for context in contexts for media in context["media"])
    return {
        "query_id": query["query_id"],
        "query_group": query["query_group"],
        "messages": messages,
        "images": images,
    }

def build_grpo_row(query: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = contexts_for_query(query, chunks)
    images = list(query["images"])
    images.extend(media["path"] for context in contexts for media in context["media"])
    return {
        "query_id": query["query_id"],
        "prompt": build_messages(query, contexts),
        "images": images,
        "gold_json": json.dumps(query["gold_output"], ensure_ascii=False),
        "context_json": json.dumps(contexts, ensure_ascii=False),
    }
