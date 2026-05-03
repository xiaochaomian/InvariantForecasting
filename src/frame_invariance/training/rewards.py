"""Reward helpers for grouped paraphrase RL training."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from frame_invariance.eval.metrics import brier, parse_probability, population_std


PUNCT_LOOP_RE = re.compile(r"([!?.;,:\-_=*#])\1{15,}")


@dataclass(frozen=True)
class RewardConfig:
    lambda_invariance: float = 0.0
    parse_fail_reward: float = -2.0
    punctuation_loop_reward: float = -2.0
    allow_loose_parse: bool = False


@dataclass(frozen=True)
class RewardResult:
    reward: float
    prediction: float | None
    parseable: bool
    punctuation_loop: bool
    brier: float | None
    invariance_penalty: float | None


def has_punctuation_loop(text: str) -> bool:
    return bool(PUNCT_LOOP_RE.search(text or ""))


def compute_group_rewards(
    completions: list[str],
    *,
    outcome: int,
    config: RewardConfig,
) -> list[RewardResult]:
    parsed: list[float | None] = []
    loops: list[bool] = []
    for completion in completions:
        loops.append(has_punctuation_loop(completion))
        parsed.append(parse_probability(completion, allow_loose=config.allow_loose_parse))

    valid_predictions = [p for p, loop in zip(parsed, loops) if p is not None and not loop]
    mean_prediction = sum(valid_predictions) / len(valid_predictions) if valid_predictions else math.nan

    results: list[RewardResult] = []
    for prediction, loop in zip(parsed, loops):
        if loop:
            results.append(
                RewardResult(
                    reward=config.punctuation_loop_reward,
                    prediction=prediction,
                    parseable=False,
                    punctuation_loop=True,
                    brier=None,
                    invariance_penalty=None,
                )
            )
            continue
        if prediction is None:
            results.append(
                RewardResult(
                    reward=config.parse_fail_reward,
                    prediction=None,
                    parseable=False,
                    punctuation_loop=False,
                    brier=None,
                    invariance_penalty=None,
                )
            )
            continue
        score = brier(prediction, outcome)
        inv = (prediction - mean_prediction) ** 2 if math.isfinite(mean_prediction) else 0.0
        results.append(
            RewardResult(
                reward=-(score + config.lambda_invariance * inv),
                prediction=prediction,
                parseable=True,
                punctuation_loop=False,
                brier=score,
                invariance_penalty=inv,
            )
        )
    return results


def normalize_advantages(
    rewards: list[float],
    *,
    eps: float = 1e-6,
    clip: float | None = 5.0,
) -> tuple[list[float], bool]:
    if not rewards:
        return [], True
    mean = sum(rewards) / len(rewards)
    std = population_std(rewards)
    if not math.isfinite(std) or std < eps:
        return [0.0 for _ in rewards], True
    advantages = [(r - mean) / (std + eps) for r in rewards]
    if clip is not None:
        advantages = [min(max(a, -clip), clip) for a in advantages]
    return advantages, False


def summarize_reward_results(groups: list[list[RewardResult]], zero_std_flags: list[bool]) -> dict[str, Any]:
    flat = [item for group in groups for item in group]
    rewards = [r.reward for r in flat]
    parseable = [r for r in flat if r.parseable]
    predictions = [float(r.prediction) for r in parseable if r.prediction is not None]
    briers = [float(r.brier) for r in parseable if r.brier is not None]
    inv = [float(r.invariance_penalty) for r in parseable if r.invariance_penalty is not None]
    return {
        "reward_mean": _mean(rewards),
        "reward_std": population_std(rewards) if rewards else None,
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_parse_rate": len(parseable) / len(flat) if flat else 0.0,
        "reward_punctuation_loop_rate": sum(1 for r in flat if r.punctuation_loop) / len(flat)
        if flat
        else 0.0,
        "reward_mean_probability": _mean(predictions),
        "reward_consensus_brier": _mean(briers),
        "reward_paraphrase_variance": _mean(inv),
        "frac_reward_zero_std": sum(1 for flag in zero_std_flags if flag) / len(zero_std_flags)
        if zero_std_flags
        else 0.0,
    }


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)
