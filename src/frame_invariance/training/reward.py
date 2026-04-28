"""Reward functions for frame-invariant forecasting RL."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_PROB_PATTERNS = [
    re.compile(r"probability\s*[:=]\s*([01](?:\.\d+)?|\.\d+|\d{1,3}\s*%)", re.I),
    re.compile(r"final\s*(?:answer|forecast)?\s*[:=]\s*([01](?:\.\d+)?|\.\d+|\d{1,3}\s*%)", re.I),
    re.compile(r"(?<!\d)([01](?:\.\d+)?|\.\d+)(?!\d)"),
    re.compile(r"\b(\d{1,3})\s*%"),
]


@dataclass(frozen=True)
class RewardConfig:
    lambda_invariance: float = 1.0
    parse_fail_reward: float = -1.0
    clamp_probabilities: bool = True


def parse_probability(text: Any) -> float | None:
    """Extract a probability in [0, 1] from model output text.

    Supports decimals like ``0.37`` and percentages like ``37%``. Returns None
    when no valid probability is found.
    """

    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    for pattern in _PROB_PATTERNS:
        match = pattern.search(raw)
        if not match:
            continue
        token = match.group(1).replace(" ", "")
        try:
            if token.endswith("%"):
                value = float(token[:-1]) / 100.0
            else:
                value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
        if 1.0 < value <= 100.0 and token.endswith("%"):
            return value / 100.0
    return None


def brier_reward(probability: float, outcome: int) -> float:
    return -((probability - float(outcome)) ** 2)


def grouped_frame_invariance_rewards(
    completions: Iterable[Any],
    outcomes: Iterable[int],
    group_ids: Iterable[Any],
    config: RewardConfig | None = None,
) -> list[float]:
    """Compute per-completion Brier reward plus paraphrase variance penalty.

    This implements the proposal reward:

        R_i = -(p_i - Y)^2 - lambda * (p_i - p_bar)^2

    where ``p_bar`` is computed over parsed probabilities from the same question
    group. Parse failures receive ``parse_fail_reward``.
    """

    cfg = config or RewardConfig()
    texts = list(completions)
    y_values = [int(y) for y in outcomes]
    gids = [str(g) for g in group_ids]
    if not (len(texts) == len(y_values) == len(gids)):
        raise ValueError("completions, outcomes, and group_ids must have equal length")

    probabilities = [parse_probability(text) for text in texts]
    parsed_by_group: dict[str, list[float]] = defaultdict(list)
    for gid, probability in zip(gids, probabilities):
        if probability is not None:
            parsed_by_group[gid].append(probability)

    means = {
        gid: sum(values) / len(values)
        for gid, values in parsed_by_group.items()
        if values
    }

    rewards: list[float] = []
    for gid, probability, outcome in zip(gids, probabilities, y_values):
        if probability is None or gid not in means:
            rewards.append(cfg.parse_fail_reward)
            continue
        p = min(1.0, max(0.0, probability)) if cfg.clamp_probabilities else probability
        if math.isnan(p):
            rewards.append(cfg.parse_fail_reward)
            continue
        consensus = means[gid]
        reward = brier_reward(p, outcome) - cfg.lambda_invariance * ((p - consensus) ** 2)
        rewards.append(float(reward))
    return rewards


def make_trl_reward_func(lambda_invariance: float, parse_fail_reward: float = -1.0):
    """Build a TRL-compatible reward function.

    TRL passes dataset columns as keyword arguments. The paraphrased dataset must
    include ``outcome`` and either ``id`` or ``source_question_index``.
    """

    config = RewardConfig(lambda_invariance=lambda_invariance, parse_fail_reward=parse_fail_reward)

    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        outcomes = kwargs.get("outcome")
        group_ids = kwargs.get("id") or kwargs.get("source_question_index")
        if outcomes is None or group_ids is None:
            raise KeyError("reward function requires outcome and id/source_question_index columns")
        return grouped_frame_invariance_rewards(completions, outcomes, group_ids, config)

    return reward_func
