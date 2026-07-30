from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[Hashable]], k: int = 60
) -> list[tuple[Hashable, float]]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[Hashable, float] = defaultdict(float)
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
