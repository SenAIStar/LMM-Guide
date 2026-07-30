from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.+\-]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


class BM25:
    def __init__(self, documents: Iterable[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = [tokenize(document) for document in documents]
        self.k1 = k1
        self.b = b
        self.average_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for document in self.documents:
            self.document_frequency.update(set(document))

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        count = len(self.documents)
        scores: list[float] = []
        for document in self.documents:
            frequencies = Counter(document)
            score = 0.0
            for term in terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                idf = math.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
                length_norm = 1.0 - self.b
                if self.average_length:
                    length_norm += self.b * len(document) / self.average_length
                score += idf * frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
            scores.append(score)
        return scores
