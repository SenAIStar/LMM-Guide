from collections.abc import Iterable, Sequence


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references) or not references:
        raise ValueError("predictions and references must have the same non-zero length")
    return sum(_normalize(p) == _normalize(r) for p, r in zip(predictions, references)) / len(references)


def recall_at_k(ranked_ids: Sequence[Sequence[str]], relevant_ids: Sequence[set[str]], k: int) -> float:
    if len(ranked_ids) != len(relevant_ids) or not relevant_ids:
        raise ValueError("ranked_ids and relevant_ids must have the same non-zero length")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    for ranking, relevant in zip(ranked_ids, relevant_ids):
        if not relevant:
            raise ValueError("each query needs at least one relevant id")
        recalls.append(len(set(ranking[:k]) & relevant) / len(relevant))
    return sum(recalls) / len(recalls)


def grounded_rate(answer_citations: Iterable[set[str]], allowed_sources: Iterable[set[str]]) -> float:
    pairs = list(zip(answer_citations, allowed_sources))
    if not pairs:
        raise ValueError("at least one answer is required")
    return sum(bool(citations) and citations <= allowed for citations, allowed in pairs) / len(pairs)

