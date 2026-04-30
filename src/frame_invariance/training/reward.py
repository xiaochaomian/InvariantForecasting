"""Reward functions for frame-invariant forecasting RL."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_EXPLICIT_PROBABILITY_PATTERN = re.compile(
    r"(?:probability|final\s*(?:answer|forecast)?)\s*[:=]\s*"
    r"(\d{1,3}\s*%|(?:0(?:\.\d+)?|1(?:\.0+)?|\.\d+))(?![\d.%])",
    re.I,
)


@dataclass(frozen=True)
class RewardConfig:
    lambda_invariance: float = 1.0
    parse_fail_reward: float = -1.0
    clamp_probabilities: bool = True


def parse_probability(text: Any) -> float | None:
    """Extract a probability in [0, 1] from model output text.

    Supports explicit forecasts like ``Probability: 0.37`` and
    ``Final answer: 37%``. Returns None when no valid probability is found.

    We intentionally avoid parsing bare numbers from rationales, because model
    outputs often contain list markers, dates, rankings, or prompt fragments
    before the final forecast.
    """

    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    matches = list(_EXPLICIT_PROBABILITY_PATTERN.finditer(raw))
    for match in reversed(matches):
        token = match.group(1).replace(" ", "")
        try:
            if token.endswith("%"):
                percent = float(token[:-1])
                if 0.0 <= percent <= 100.0:
                    return percent / 100.0
                continue
            else:
                value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
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


def grouped_reward_metrics(
    completions: Iterable[Any],
    outcomes: Iterable[int],
    group_ids: Iterable[Any],
) -> dict[str, float]:
    texts = list(completions)
    y_values = [int(y) for y in outcomes]
    gids = [str(g) for g in group_ids]
    probabilities = [parse_probability(text) for text in texts]

    parsed_by_group: dict[str, list[float]] = defaultdict(list)
    outcome_by_group: dict[str, int] = {}
    for gid, probability, outcome in zip(gids, probabilities, y_values):
        outcome_by_group[gid] = outcome
        if probability is not None:
            parsed_by_group[gid].append(probability)

    consensus_briers: list[float] = []
    variances: list[float] = []
    parsed_probabilities: list[float] = []
    for gid, values in parsed_by_group.items():
        if not values:
            continue
        consensus = sum(values) / len(values)
        outcome = outcome_by_group[gid]
        consensus_briers.append((consensus - outcome) ** 2)
        variances.append(sum((value - consensus) ** 2 for value in values) / len(values))
        parsed_probabilities.extend(values)

    return {
        "parse_rate": sum(probability is not None for probability in probabilities) / len(probabilities)
        if probabilities else 0.0,
        "consensus_brier": sum(consensus_briers) / len(consensus_briers)
        if consensus_briers else math.nan,
        "paraphrase_variance": sum(variances) / len(variances) if variances else math.nan,
        "mean_probability": sum(parsed_probabilities) / len(parsed_probabilities)
        if parsed_probabilities else math.nan,
    }


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
        rewards = grouped_frame_invariance_rewards(completions, outcomes, group_ids, config)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric):
            for name, value in grouped_reward_metrics(completions, outcomes, group_ids).items():
                if not math.isnan(value):
                    log_metric(f"reward_{name}", value)
        return rewards

    return reward_func
