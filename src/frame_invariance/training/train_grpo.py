"""TRL GRPO entry point for frame-invariant forecasting post-training.

This script is meant for a GPU environment. It keeps paraphrase rows grouped and
uses a reward that combines per-output Brier score with an explicit variance
penalty across semantically equivalent frames.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any


from .data import SplitConfig, load_grouped_prompt_splits
from .reward import make_trl_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO post-training for frame-invariant forecasting.")
    parser.add_argument("--config", default="configs/training/grpo_forecastbench.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config without importing GPU training libraries.")
    return parser.parse_args()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    # Fallback parser for this repo's simple config shape when PyYAML is not installed.
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    pending_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            current_section = {}
            root[line[:-1]] = current_section
            pending_list_key = None
        elif indent == 2 and current_section is not None and line.endswith(":"):
            pending_list_key = line[:-1]
            current_section[pending_list_key] = []
        elif indent == 2 and current_section is not None and ":" in line:
            key, value = line.split(":", 1)
            current_section[key.strip()] = _parse_scalar(value)
            pending_list_key = None
        elif indent == 4 and current_section is not None and pending_list_key and line.startswith("-"):
            current_section[pending_list_key].append(_parse_scalar(line[1:]))
        else:
            raise ValueError(f"Unsupported config line in {path}: {raw_line!r}")
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml  # type: ignore
    except ImportError:
        loaded = _load_simple_yaml(config_path)
    else:
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return loaded


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    splits = load_grouped_prompt_splits(
        data_cfg.get("paraphrase_path", "data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl"),
        SplitConfig(
            train_frac=float(data_cfg.get("train_frac", 0.8)),
            val_frac=float(data_cfg.get("val_frac", 0.1)),
            seed=int(data_cfg.get("seed", 17)),
        ),
        expected_group_size=int(data_cfg.get("group_size", 5)),
    )
    print(
        "split rows:",
        {name: len(rows) for name, rows in splits.items()},
        "split groups:",
        {name: len({row["id"] for row in rows}) for name, rows in splits.items()},
    )

    if args.dry_run:
        sample = splits["train"][:5]
        reward_func = make_trl_reward_func(float(train_cfg.get("lambda_invariance", 1.0)))
        demo_rewards = reward_func(
            ["Probability: 0.50"] * len(sample),
            outcome=[row["outcome"] for row in sample],
            id=[row["id"] for row in sample],
        )
        print("dry-run reward sample:", demo_rewards)
        return 0

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install with `pip install -e .` in a GPU environment."
        ) from exc

    group_size = int(data_cfg.get("group_size", 5))
    train_batch_size = int(train_cfg.get("per_device_train_batch_size", group_size))
    if train_batch_size % group_size != 0:
        raise ValueError(
            "per_device_train_batch_size must be a multiple of data.group_size so "
            "paraphrase groups stay together inside reward batches"
        )
    if bool(train_cfg.get("shuffle_dataset", False)):
        raise ValueError("shuffle_dataset must stay false for grouped paraphrase rewards")

    train_dataset = Dataset.from_list(splits["train"])
    eval_dataset = Dataset.from_list(splits["validation"])
    reward_func = make_trl_reward_func(
        lambda_invariance=float(train_cfg.get("lambda_invariance", 1.0)),
        parse_fail_reward=float(train_cfg.get("parse_fail_reward", -1.0)),
    )

    peft_cfg = LoraConfig(
        r=int(model_cfg.get("lora_r", 32)),
        lora_alpha=int(model_cfg.get("lora_alpha", 64)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    num_generations = int(train_cfg.get("num_generations", 5))
    eval_strategy = train_cfg.get("eval_strategy", "steps")
    if eval_strategy is False or eval_strategy is None:
        eval_strategy = "no"
    elif eval_strategy is True:
        eval_strategy = "steps"
    else:
        eval_strategy = str(eval_strategy)

    grpo_kwargs = {
        "output_dir": train_cfg.get("output_dir", "outputs/grpo_forecastbench"),
        "learning_rate": float(train_cfg.get("learning_rate", 5e-6)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 5)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", num_generations)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 1)),
        "num_generations": num_generations,
        "beta": float(train_cfg.get("beta", 0.04)),
        "max_prompt_length": int(train_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(train_cfg.get("max_completion_length", 96)),
        "logging_steps": int(train_cfg.get("logging_steps", 5)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        "eval_strategy": eval_strategy,
        "eval_steps": int(train_cfg.get("eval_steps", 50)),
        "dataloader_drop_last": True,
        "shuffle_dataset": bool(train_cfg.get("shuffle_dataset", False)),
        "scale_rewards": train_cfg.get("scale_rewards", "batch"),
        "loss_type": train_cfg.get("loss_type", "dapo"),
        "remove_unused_columns": False,
    }
    supported_args = set(inspect.signature(GRPOConfig).parameters)
    dropped_args = sorted(set(grpo_kwargs) - supported_args)
    if dropped_args:
        print(f"Dropping unsupported GRPOConfig args for installed TRL: {dropped_args}")
    grpo_args = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in supported_args})

    trainer = GRPOTrainer(
        model=model_cfg.get("name", "Qwen/Qwen2.5-7B-Instruct"),
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
