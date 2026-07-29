from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Iterable


def group_relative_advantages(
    rewards: Iterable[float], group_ids: Iterable[Hashable], epsilon: float = 1e-6
) -> list[float]:
    reward_list = [float(value) for value in rewards]
    group_list = list(group_ids)
    if len(reward_list) != len(group_list):
        raise ValueError("rewards and group_ids must have equal lengths")
    by_group: dict[Hashable, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_list):
        by_group[group_id].append(index)
    advantages = [0.0] * len(reward_list)
    for indices in by_group.values():
        values = [reward_list[index] for index in indices]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        if std <= epsilon:
            continue
        for index in indices:
            advantages[index] = (reward_list[index] - mean) / std
    return advantages

