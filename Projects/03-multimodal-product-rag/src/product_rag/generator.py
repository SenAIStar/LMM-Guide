from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_messages(query: dict[str, Any], contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in query["images"]:
        content.append({"type": "image", "image": image})
    context_payload = []
    for context in contexts:
        media_indices: list[int] = []
        for media in context["media"]:
            content.append({"type": "image", "image": media["path"]})
            media_indices.append(len(content) - 1)
        context_payload.append(
            {
                "chunk_id": context["chunk_id"],
                "product_id": context["product_id"],
                "source_revision": context["source_revision"],
                "title": context["title"],
                "text": context["text"],
                "facts": context["facts"],
                "prompt_image_positions": media_indices,
            }
        )
    request = {"question": query["text"], "contexts": context_payload}
    content.append({"type": "text", "text": json.dumps(request, ensure_ascii=False)})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "只依据给定证据回答，并只输出 product-qa.v1 JSON。每条事实必须引用 chunk_id。",
                }
            ],
        },
        {"role": "user", "content": content},
    ]


class Qwen3VLGenerator:
    def __init__(self, model_id: str, revision: str, **model_kwargs: Any) -> None:
        if not revision or revision == "REQUIRED_COMMIT_SHA":
            raise ValueError("pin model revision to an immutable commit SHA")
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install requirements-ml.txt") from exc
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, device_map="auto", dtype="auto", **model_kwargs
        )

    def generate(self, query: dict[str, Any], contexts: list[dict[str, Any]], max_new_tokens: int = 512) -> str:
        messages = build_messages(query, contexts)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
