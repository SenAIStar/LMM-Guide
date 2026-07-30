from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable


def group_relative_advantages(
    rewards: Iterable[float], group_ids: Iterable[Hashable], epsilon: float = 1e-6
) -> list[float]:
    values = [float(value) for value in rewards]
    groups = list(group_ids)
    if len(values) != len(groups):
        raise ValueError("rewards and group_ids must have equal lengths")
    indices_by_group: dict[Hashable, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        indices_by_group[group].append(index)
    advantages = [0.0] * len(values)
    for indices in indices_by_group.values():
        group_values = [values[index] for index in indices]
        mean = sum(group_values) / len(group_values)
        variance = sum((value - mean) ** 2 for value in group_values) / len(group_values)
        standard_deviation = math.sqrt(variance)
        if standard_deviation <= epsilon:
            continue
        for index in indices:
            advantages[index] = (values[index] - mean) / standard_deviation
    return advantages
