from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = ""
    max_new_tokens: int = 512


class Qwen3VLAdapter:
    """Lazy Transformers adapter following the official Qwen3-VL chat path."""

    def __init__(self, config: InferenceConfig) -> None:
        if not config.revision or config.revision.startswith("REPLACE_WITH"):
            raise ValueError("an immutable reviewed model revision is required")
        self.config = config
        self._model: Any = None
        self._processor: Any = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            dtype="auto",
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
        )

    def generate(self, image_paths: Sequence[Path], prompt: str) -> str:
        if not image_paths:
            raise ValueError("at least one image is required")
        if self._model is None or self._processor is None:
            self.load()
        content = [
            {"type": "image", "image": path.resolve().as_uri()}
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
        )
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
