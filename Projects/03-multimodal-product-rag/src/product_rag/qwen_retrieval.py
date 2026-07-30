from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from .catalog import searchable_text


INSTRUCTION = "Retrieve product evidence that directly answers the user's query."


class QwenMultimodalEmbedder:
    """Adapter around the official Qwen3-VL-Embedding repository."""

    def __init__(self, official_repo: str | Path, model_path: str | Path, **model_kwargs: Any) -> None:
        repo = Path(official_repo).resolve()
        if not (repo / "src" / "models" / "qwen3_vl_embedding.py").is_file():
            raise ValueError("official_repo must be a pinned Qwen3-VL-Embedding checkout")
        sys.path.insert(0, str(repo))
        module = importlib.import_module("src.models.qwen3_vl_embedding")
        self.model = module.Qwen3VLEmbedder(str(Path(model_path).resolve()), **model_kwargs)

    def embed_query(self, query: dict[str, Any]) -> list[float]:
        item: dict[str, Any] = {"text": query["text"], "instruction": INSTRUCTION}
        if query["images"]:
            item["image"] = query["images"]
        tensor = self.model.process([item])[0]
        return tensor.detach().float().cpu().tolist()

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        items: list[dict[str, Any]] = []
        for chunk in chunks:
            item: dict[str, Any] = {"text": searchable_text(chunk)}
            if chunk["media"]:
                item["image"] = [media["path"] for media in chunk["media"]]
            items.append(item)
        tensors = self.model.process(items)
        return tensors.detach().float().cpu().tolist()


class QwenMultimodalReranker:
    def __init__(self, official_repo: str | Path, model_path: str | Path, **model_kwargs: Any) -> None:
        repo = Path(official_repo).resolve()
        if not (repo / "src" / "models" / "qwen3_vl_reranker.py").is_file():
            raise ValueError("official_repo must be a pinned Qwen3-VL-Embedding checkout")
        sys.path.insert(0, str(repo))
        module = importlib.import_module("src.models.qwen3_vl_reranker")
        self.model = module.Qwen3VLReranker(str(Path(model_path).resolve()), **model_kwargs)

    def score(self, query: dict[str, Any], chunks: list[dict[str, Any]]) -> list[float]:
        query_input: dict[str, Any] = {"text": query["text"]}
        if query["images"]:
            query_input["image"] = query["images"]
        documents: list[dict[str, Any]] = []
        for chunk in chunks:
            document: dict[str, Any] = {"text": searchable_text(chunk)}
            if chunk["media"]:
                document["image"] = [item["path"] for item in chunk["media"]]
            documents.append(document)
        return [
            float(score)
            for score in self.model.process(
                {"instruction": INSTRUCTION, "query": query_input, "documents": documents}
            )
        ]
