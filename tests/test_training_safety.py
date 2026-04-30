import unittest

from frame_invariance.training.train_grpo import (
    TrainingSafetyError,
    build_training_safety_callback,
    load_config,
    validate_training_safety_config,
)


class _CallbackBase:
    pass


class _State:
    global_step = 3


class _Control:
    should_training_stop = False


class TrainingSafetyTests(unittest.TestCase):
    def test_safety_callback_allows_healthy_metrics(self):
        callback = build_training_safety_callback(
            _CallbackBase,
            {"safety": {"enabled": True, "min_reward_parse_rate": 0.95}},
        )
        control = _Control()
        result = callback.on_log(
            None,
            _State(),
            control,
            {
                "reward_parse_rate": "1",
                "reward_punctuation_loop_rate": "0",
                "frac_reward_zero_std": "0",
                "grad_norm": "0.25",
                "kl": "0.001",
            },
        )
        self.assertIs(result, control)
        self.assertFalse(control.should_training_stop)

    def test_safety_callback_stops_on_parse_collapse(self):
        callback = build_training_safety_callback(
            _CallbackBase,
            {"safety": {"enabled": True, "min_reward_parse_rate": 0.95}},
        )
        control = _Control()
        with self.assertRaises(TrainingSafetyError):
            callback.on_log(None, _State(), control, {"reward_parse_rate": "0.2"})
        self.assertTrue(control.should_training_stop)

    def test_safety_callback_stops_on_punctuation_loop(self):
        callback = build_training_safety_callback(
            _CallbackBase,
            {"safety": {"enabled": True, "max_punctuation_loop_rate": 0.0}},
        )
        with self.assertRaises(TrainingSafetyError):
            callback.on_log(None, _State(), _Control(), {"reward_punctuation_loop_rate": "0.04"})

    def test_safety_callback_stops_on_nonfinite_grad_norm(self):
        callback = build_training_safety_callback(_CallbackBase, {"safety": {"enabled": True}})
        with self.assertRaises(TrainingSafetyError):
            callback.on_log(None, _State(), _Control(), {"grad_norm": "nan"})

    def test_mantic_configs_are_guarded(self):
        for path in (
            "configs/training/grpo_forecastbench_mantic_lambda0.yaml",
            "configs/training/grpo_forecastbench_mantic_lambda1.yaml",
        ):
            with self.subTest(path=path):
                cfg = load_config(path)
                train_cfg = cfg["training"]
                validate_training_safety_config(train_cfg)
                self.assertEqual(train_cfg["scale_rewards"], "none")
                self.assertEqual(train_cfg["save_steps"], 5)
                self.assertLessEqual(train_cfg["learning_rate"], 2e-7)
                self.assertTrue(train_cfg["safety"]["enabled"])


if __name__ == "__main__":
    unittest.main()
