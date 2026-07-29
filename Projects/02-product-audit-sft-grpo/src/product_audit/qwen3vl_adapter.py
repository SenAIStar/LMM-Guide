from __future__ import annotations

from pathlib import Path
from typing import Any


class Qwen3VLAuditor:
    """Thin lazy-loading adapter matching Qwen3-VL's official Transformers API."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        if not revision or revision == "REQUIRED_COMMIT_SHA":
            raise ValueError("pin model revision to an immutable commit SHA")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install requirements-ml.txt before loading Qwen3-VL") from exc
        dtype: Any = torch_dtype
        if torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            device_map=device_map,
            torch_dtype=dtype,
        )

    def generate(
        self,
        image_paths: list[str | Path],
        request_text: str,
        max_new_tokens: int = 512,
    ) -> str:
        from PIL import Image

        images = [Image.open(path).convert("RGB") for path in image_paths]
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": request_text})
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "审核商品并只输出 audit-output.v1 JSON。",
                    }
                ],
            },
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )[0]

