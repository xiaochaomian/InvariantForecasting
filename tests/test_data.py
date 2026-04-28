import unittest

from frame_invariance.training.data import SplitConfig, group_paraphrases, load_grouped_prompt_splits


class DataTests(unittest.TestCase):
    def test_group_paraphrases_validates_group_size(self):
        rows = [
            {"id": "a", "variant_index": i, "outcome": 1, "question": f"q{i}"}
            for i in range(5)
        ]
        groups = group_paraphrases(rows, expected_group_size=5)
        self.assertEqual(list(groups), ["a"])
        self.assertEqual([row["variant_index"] for row in groups["a"]], list(range(5)))

    def test_load_grouped_prompt_splits_on_repo_data(self):
        splits = load_grouped_prompt_splits(
            "data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl",
            SplitConfig(train_frac=0.8, val_frac=0.1, seed=17),
        )
        self.assertEqual(len(splits["train"]), 2390)
        self.assertEqual(len(splits["validation"]), 300)
        self.assertEqual(len(splits["test"]), 300)
        self.assertTrue(all("Probability:" in row["prompt"] for row in splits["train"][:5]))


if __name__ == "__main__":
    unittest.main()
