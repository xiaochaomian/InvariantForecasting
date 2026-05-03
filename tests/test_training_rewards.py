"""Offline tests for grouped GRPO reward helpers."""

from __future__ import annotations

from frame_invariance.training.rewards import (
    RewardConfig,
    compute_group_rewards,
    has_punctuation_loop,
    normalize_advantages,
)


def test_compute_group_rewards_lambda0_brier_only():
    results = compute_group_rewards(
        ["Probability: 0.2", "Probability: 0.4"],
        outcome=1,
        config=RewardConfig(lambda_invariance=0.0),
    )
    assert [r.parseable for r in results] == [True, True]
    assert abs(results[0].reward - -0.64) < 1e-9
    assert abs(results[1].reward - -0.36) < 1e-9


def test_compute_group_rewards_lambda1_penalizes_paraphrase_variance():
    results = compute_group_rewards(
        ["Probability: 0.2", "Probability: 0.4"],
        outcome=1,
        config=RewardConfig(lambda_invariance=1.0),
    )
    assert abs(results[0].invariance_penalty - 0.01) < 1e-9
    assert abs(results[1].invariance_penalty - 0.01) < 1e-9
    assert abs(results[0].reward - -0.65) < 1e-9
    assert abs(results[1].reward - -0.37) < 1e-9


def test_parse_fail_and_punctuation_loop_rewards():
    results = compute_group_rewards(
        ["no numeric forecast", "!!!!!!!!!!!!!!!!!!!!!!!!!!"],
        outcome=0,
        config=RewardConfig(parse_fail_reward=-2.0, punctuation_loop_reward=-3.0),
    )
    assert results[0].reward == -2.0
    assert results[0].parseable is False
    assert results[1].reward == -3.0
    assert results[1].punctuation_loop is True
    assert has_punctuation_loop("!!!!!!!!!!!!!!!!")


def test_normalize_advantages_zero_std():
    adv, zero_std = normalize_advantages([1.0, 1.0, 1.0])
    assert zero_std is True
    assert adv == [0.0, 0.0, 0.0]


def test_normalize_advantages_nonzero_std():
    adv, zero_std = normalize_advantages([0.0, 1.0])
    assert zero_std is False
    assert adv[0] < 0
    assert adv[1] > 0
