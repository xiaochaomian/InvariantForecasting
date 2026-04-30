import unittest

from frame_invariance.training.reward import (
    RewardConfig,
    grouped_frame_invariance_rewards,
    parse_probability,
)


class RewardTests(unittest.TestCase):
    def test_parse_probability_decimal_and_percent(self):
        self.assertEqual(parse_probability("Probability: 0.37"), 0.37)
        self.assertEqual(parse_probability("Final answer: 37%"), 0.37)
        self.assertIsNone(parse_probability("I think .8"))
        self.assertIsNone(parse_probability("1> Rationale: list marker without forecast"))
        self.assertEqual(parse_probability("Probability: 0.2 then later Probability: 0.7"), 0.7)
        self.assertEqual(parse_probability("Final forecast = .42"), 0.42)
        self.assertEqual(parse_probability("Probability: 100%"), 1.0)
        self.assertIsNone(parse_probability("Probability: 150%"))
        self.assertIsNone(parse_probability("Probability: 1.2"))
        self.assertIsNone(parse_probability("no numeric forecast"))

    def test_grouped_frame_invariance_rewards_lambda_zero_is_brier(self):
        rewards = grouped_frame_invariance_rewards(
            ["Probability: 0.7", "Probability: 0.6", "Probability: 0.8"],
            [1, 1, 1],
            ["q1", "q1", "q1"],
            RewardConfig(lambda_invariance=0.0),
        )
        self.assertAlmostEqual(rewards[0], -0.09)
        self.assertAlmostEqual(rewards[1], -0.16)
        self.assertAlmostEqual(rewards[2], -0.04)

    def test_grouped_frame_invariance_rewards_penalizes_variance(self):
        brier_only = grouped_frame_invariance_rewards(
            ["Probability: 0.7", "Probability: 0.6", "Probability: 0.8"],
            [1, 1, 1],
            ["q1", "q1", "q1"],
            RewardConfig(lambda_invariance=0.0),
        )
        invariant = grouped_frame_invariance_rewards(
            ["Probability: 0.7", "Probability: 0.6", "Probability: 0.8"],
            [1, 1, 1],
            ["q1", "q1", "q1"],
            RewardConfig(lambda_invariance=1.0),
        )
        self.assertEqual(invariant[0], brier_only[0])
        self.assertLess(invariant[1], brier_only[1])
        self.assertLess(invariant[2], brier_only[2])

    def test_parse_failure_gets_penalty(self):
        rewards = grouped_frame_invariance_rewards(
            ["Probability: 0.7", "no parse"],
            [1, 1],
            ["q1", "q1"],
            RewardConfig(lambda_invariance=1.0, parse_fail_reward=-3.0),
        )
        self.assertEqual(rewards[1], -3.0)


if __name__ == "__main__":
    unittest.main()
