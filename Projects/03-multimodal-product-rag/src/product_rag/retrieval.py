from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Protocol

from .catalog import is_eligible, searchable_text
from .fusion import reciprocal_rank_fusion
from .sparse import BM25, tokenize


class Embedder(Protocol):
    def embed_query(self, query: dict[str, Any]) -> list[float]: ...

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: dict[str, Any], chunks: list[dict[str, Any]]) -> list[float]: ...


@dataclass(frozen=True)
class SearchHit:
    chunk: dict[str, Any]
    fusion_score: float
    rerank_score: float | None


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return sum(x * y for x, y in zip(_normalize(left), _normalize(right)))


class HashingEmbedder:
    """Deterministic hashing embedder used by retrieval examples."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def _encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def embed_query(self, query: dict[str, Any]) -> list[float]:
        image_names = " ".join(str(path).replace("\\", "/").split("/")[-1] for path in query["images"])
        return self._encode(f"{query['text']} {image_names}")

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        return [self._encode(searchable_text(chunk)) for chunk in chunks]


class TokenOverlapReranker:
    """Token-overlap reranker used by retrieval examples."""

    def score(self, query: dict[str, Any], chunks: list[dict[str, Any]]) -> list[float]:
        query_terms = set(tokenize(query["text"]))
        values: list[float] = []
        for chunk in chunks:
            terms = set(tokenize(searchable_text(chunk)))
            values.append(len(query_terms & terms) / len(query_terms) if query_terms else 0.0)
        return values


class HybridRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embedder: Embedder,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k

    @staticmethod
    def _rank(scores: list[float], chunks: list[dict[str, Any]], limit: int) -> list[str]:
        ordered = sorted(
            zip(chunks, scores),
            key=lambda pair: (-pair[1], pair[0]["chunk_id"]),
        )
        return [chunk["chunk_id"] for chunk, _ in ordered[:limit]]

    def search(
        self,
        query: dict[str, Any],
        dense_limit: int = 40,
        sparse_limit: int = 40,
        rerank_limit: int = 12,
        final_limit: int = 6,
        max_chunks_per_variant: int = 3,
    ) -> list[SearchHit]:
        # Security and lifecycle filters are applied before either retrieval leg.
        eligible = [chunk for chunk in self.chunks if is_eligible(chunk, query)]
        if not eligible:
            return []
        query_vector = self.embedder.embed_query(query)
        chunk_vectors = self.embedder.embed_chunks(eligible)
        dense_scores = [cosine(query_vector, vector) for vector in chunk_vectors]
        sparse_scores = BM25(searchable_text(chunk) for chunk in eligible).scores(query["text"])
        dense_ranking = self._rank(dense_scores, eligible, dense_limit)
        sparse_ranking = self._rank(sparse_scores, eligible, sparse_limit)
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=self.rrf_k)
        chunk_map = {chunk["chunk_id"]: chunk for chunk in eligible}
        candidates = [(chunk_map[chunk_id], score) for chunk_id, score in fused[:rerank_limit]]

        if self.reranker and candidates:
            rerank_scores = self.reranker.score(query, [chunk for chunk, _ in candidates])
            ranked = sorted(
                zip(candidates, rerank_scores),
                key=lambda item: (-item[1], -item[0][1], item[0][0]["chunk_id"]),
            )
            raw_hits = [SearchHit(chunk, fusion_score, rerank_score) for ((chunk, fusion_score), rerank_score) in ranked]
        else:
            raw_hits = [SearchHit(chunk, fusion_score, None) for chunk, fusion_score in candidates]

        selected: list[SearchHit] = []
        counts: dict[tuple[str, str], int] = {}
        for hit in raw_hits:
            key = (hit.chunk["product_id"], hit.chunk["variant_id"])
            if counts.get(key, 0) >= max_chunks_per_variant:
                continue
            counts[key] = counts.get(key, 0) + 1
            selected.append(hit)
            if len(selected) >= final_limit:
                break
        return selected
