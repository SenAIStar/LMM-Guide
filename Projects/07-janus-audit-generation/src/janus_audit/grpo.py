from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Sequence


@dataclass(frozen=True)
class AdvantageBatch:
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean: float
    std: float
    zero_variance: bool


def group_relative_advantages(rewards: Sequence[float], epsilon: float = 1e-6) -> AdvantageBatch:
    if len(rewards) < 2:
        raise ValueError("GRPO needs at least two candidates per prompt")
    values = tuple(float(value) for value in rewards)
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    std = math.sqrt(variance)
    if std < epsilon:
        advantages = tuple(0.0 for _ in values)
        return AdvantageBatch(values, advantages, mean, std, True)
    advantages = tuple((value - mean) / (std + epsilon) for value in values)
    return AdvantageBatch(values, advantages, mean, std, False)

