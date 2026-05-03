"""Mantic-style grouped GRPO training on Tinker.

This trainer is intentionally conservative:

  - one rollout per paraphrase variant in a question group
  - rewards are grouped and normalized within the paraphrase group
  - lambda=0 optimizes Brier only; lambda>0 adds a paraphrase-variance penalty
  - parse failures and punctuation loops receive explicit negative rewards
  - safety stops abort before a degenerate run burns a full budget

It uses Tinker's built-in PPO loss on sampled tokens. The sampled logprobs from
the rollout policy become the PPO reference logprobs; grouped normalized rewards
become per-token advantages over the generated completion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from frame_invariance.eval.baseline import decode_tokens, render_chat_prompt
from frame_invariance.training.rewards import (
    RewardConfig,
    RewardResult,
    compute_group_rewards,
    normalize_advantages,
    summarize_reward_results,
)


DEFAULT_CONFIG = "configs/training/tinker_grpo_lambda0.yaml"


@dataclass(frozen=True)
class TrainConfig:
    input_path: Path
    output_dir: Path
    results_dir: Path
    run_name: str
    base_model: str
    split: str
    seed: int
    max_steps: int
    groups_per_step: int
    save_every: int
    eval_every: int
    lora_rank: int
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    learning_rate: float
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip_norm: float
    max_tokens: int
    temperature: float
    top_p: float
    lambda_invariance: float
    parse_fail_reward: float
    punctuation_loop_reward: float
    advantage_clip: float
    ppo_clip_low: float
    ppo_clip_high: float
    safety_min_parse_rate: float
    safety_max_punctuation_loop_rate: float
    safety_max_zero_std_frac: float
    safety_start_step: int
    tinker_api_key_env: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_rows(rows: list[dict[str, Any]], *, split: str) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        grouped.setdefault(str(row["id"]), []).append(row)
    groups = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: int(r.get("variant_index", 0)))
        if len(ordered) < 2:
            continue
        outcomes = {int(r["outcome"]) for r in ordered}
        if len(outcomes) != 1:
            raise ValueError(f"group {ordered[0]['id']} has inconsistent outcomes")
        groups.append(ordered)
    groups.sort(key=lambda g: str(g[0]["id"]))
    return groups


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        out.append({"role": role, "content": str(msg.get("content", ""))})
    if not out:
        raise ValueError("row has no usable messages")
    return out


class TinkerTrainer:
    def __init__(self, config: TrainConfig) -> None:
        api_key = os.environ.get(config.tinker_api_key_env)
        if not api_key:
            raise RuntimeError(f"${config.tinker_api_key_env} is not set")
        os.environ.setdefault("TINKER_API_KEY", api_key)
        try:
            import tinker  # type: ignore
            from tinker import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install Tinker with `pip install -e '.[tinker]'`") from exc

        self.config = config
        self.tinker = tinker
        self.types = types
        self.service_client = tinker.ServiceClient()
        self.training_client = self.service_client.create_lora_training_client(
            base_model=config.base_model,
            rank=config.lora_rank,
            seed=config.seed,
            train_mlp=config.train_mlp,
            train_attn=config.train_attn,
            train_unembed=config.train_unembed,
            user_metadata={"run_name": config.run_name},
        )
        self.tokenizer = self.training_client.get_tokenizer()
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()

    def sample_one(self, row: dict[str, Any], *, seed: int) -> dict[str, Any]:
        prompt_text = render_chat_prompt(self.tokenizer, normalize_messages(row.get("messages") or []))
        prompt_tokens = list(self.tokenizer.encode(prompt_text))
        prompt = self.types.ModelInput.from_ints(prompt_tokens)
        params = self.types.SamplingParams(
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=seed,
        )
        result = self.sampling_client.sample(prompt, 1, params).result()
        sequence = result.sequences[0]
        completion_tokens = list(sequence.tokens)
        if sequence.logprobs is None:
            raise RuntimeError("Tinker sampling did not return completion logprobs required for PPO")
        completion_logprobs = list(sequence.logprobs)
        if len(completion_logprobs) != len(completion_tokens):
            raise RuntimeError("Tinker sampling returned misaligned tokens/logprobs")
        completion = decode_tokens(self.tokenizer, completion_tokens)
        return {
            "row": row,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "completion_logprobs": completion_logprobs,
            "completion": completion,
            "stop_reason": str(sequence.stop_reason),
        }

    def make_datum(self, sample: dict[str, Any], advantage: float) -> Any:
        prompt_tokens = sample["prompt_tokens"]
        completion_tokens = sample["completion_tokens"]
        completion_logprobs = sample["completion_logprobs"]
        if not completion_tokens:
            fallback_token = self.tokenizer.eos_token_id
            if fallback_token is None:
                fallback_token = self.tokenizer.encode("\n")[0]
            completion_tokens = [int(fallback_token)]
            completion_logprobs = [0.0]

        # Match Tinker's RL examples: next-token targets across prompt+completion,
        # with zero advantages on prompt positions and rollout advantages only on
        # sampled completion tokens.
        full_tokens = prompt_tokens + completion_tokens
        model_input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        n_prompt_targets = max(0, len(prompt_tokens) - 1)
        n_completion = len(completion_tokens)
        old_logprobs = [0.0] * n_prompt_targets + [float(x) for x in completion_logprobs]
        per_token_adv = advantage / max(1, n_completion)
        advantages = [0.0] * n_prompt_targets + [float(per_token_adv)] * n_completion
        if len(target_tokens) != len(old_logprobs) or len(target_tokens) != len(advantages):
            raise ValueError("token alignment bug while building PPO datum")
        if len(model_input_tokens) != len(target_tokens):
            raise ValueError("model input / target alignment bug while building PPO datum")
        return self.types.Datum(
            model_input=self.types.ModelInput.from_ints(model_input_tokens),
            loss_fn_inputs={
                "target_tokens": self.types.TensorData(
                    data=[int(x) for x in target_tokens],
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "logprobs": self.types.TensorData(
                    data=old_logprobs,
                    dtype="float32",
                    shape=[len(old_logprobs)],
                ),
                "advantages": self.types.TensorData(
                    data=advantages,
                    dtype="float32",
                    shape=[len(advantages)],
                ),
            },
        )

    def train_step(self, step: int, batch_groups: list[list[dict[str, Any]]]) -> dict[str, Any]:
        all_group_rewards: list[list[RewardResult]] = []
        zero_std_flags: list[bool] = []
        datums = []
        prediction_rows = []

        reward_config = RewardConfig(
            lambda_invariance=self.config.lambda_invariance,
            parse_fail_reward=self.config.parse_fail_reward,
            punctuation_loop_reward=self.config.punctuation_loop_reward,
        )
        for group_index, group in enumerate(batch_groups):
            samples = [
                self.sample_one(row, seed=self.config.seed + step * 1000 + group_index * 10 + i)
                for i, row in enumerate(group)
            ]
            outcome = int(group[0]["outcome"])
            reward_results = compute_group_rewards(
                [str(sample["completion"]) for sample in samples],
                outcome=outcome,
                config=reward_config,
            )
            advantages, zero_std = normalize_advantages(
                [r.reward for r in reward_results],
                clip=self.config.advantage_clip,
            )
            all_group_rewards.append(reward_results)
            zero_std_flags.append(zero_std)
            for sample, result, advantage in zip(samples, reward_results, advantages):
                datums.append(self.make_datum(sample, advantage))
                row = sample["row"]
                prediction_rows.append(
                    {
                        "id": row["id"],
                        "variant_index": row.get("variant_index", 0),
                        "outcome": row["outcome"],
                        "prediction": result.prediction,
                        "parseable": result.parseable,
                        "reward": result.reward,
                        "advantage": advantage,
                        "stop_reason": sample["stop_reason"],
                        "completion": sample["completion"],
                    }
                )

        fwdbwd = self.training_client.forward_backward(
            datums,
            "ppo",
            {
                "clip_low_threshold": self.config.ppo_clip_low,
                "clip_high_threshold": self.config.ppo_clip_high,
            },
        )
        optim = self.training_client.optim_step(
            self.types.AdamParams(
                learning_rate=self.config.learning_rate,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
                weight_decay=self.config.weight_decay,
                grad_clip_norm=self.config.grad_clip_norm,
            )
        )
        fwdbwd_result = fwdbwd.result()
        optim_result = optim.result()
        self.sampling_client = self.training_client.save_weights_and_get_sampling_client()

        metrics = summarize_reward_results(all_group_rewards, zero_std_flags)
        metrics.update(
            {
                "step": step,
                "n_groups": len(batch_groups),
                "n_variants": len(prediction_rows),
                "loss_metrics": fwdbwd_result.metrics,
                "optim_response": str(optim_result),
            }
        )
        return {"metrics": metrics, "prediction_rows": prediction_rows}

    def save_checkpoint(self, step: int) -> str:
        name = f"{self.config.run_name}_step{step}"
        response = self.training_client.save_weights_for_sampler(name).result()
        return str(response.path)


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    t = raw.get("training", raw)
    return TrainConfig(
        input_path=Path(t.get("input_path", "data/processed/training.jsonl")),
        output_dir=Path(t.get("output_dir", f"outputs/{t.get('run_name', 'tinker_grpo')}")),
        results_dir=Path(t.get("results_dir", "results")),
        run_name=str(t.get("run_name", "tinker_grpo")),
        base_model=str(t.get("base_model", "openai/gpt-oss-120b")),
        split=str(t.get("split", "train")),
        seed=int(t.get("seed", 17)),
        max_steps=int(t.get("max_steps", 5)),
        groups_per_step=int(t.get("groups_per_step", 1)),
        save_every=int(t.get("save_every", 5)),
        eval_every=int(t.get("eval_every", 0)),
        lora_rank=int(t.get("lora_rank", 16)),
        train_mlp=bool(t.get("train_mlp", True)),
        train_attn=bool(t.get("train_attn", True)),
        train_unembed=bool(t.get("train_unembed", False)),
        learning_rate=float(t.get("learning_rate", 2e-7)),
        beta1=float(t.get("beta1", 0.9)),
        beta2=float(t.get("beta2", 0.95)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        grad_clip_norm=float(t.get("grad_clip_norm", 0.05)),
        max_tokens=int(t.get("max_tokens", 512)),
        temperature=float(t.get("temperature", 0.7)),
        top_p=float(t.get("top_p", 0.95)),
        lambda_invariance=float(t.get("lambda_invariance", 0.0)),
        parse_fail_reward=float(t.get("parse_fail_reward", -2.0)),
        punctuation_loop_reward=float(t.get("punctuation_loop_reward", -2.0)),
        advantage_clip=float(t.get("advantage_clip", 5.0)),
        ppo_clip_low=float(t.get("ppo_clip_low", 0.2)),
        ppo_clip_high=float(t.get("ppo_clip_high", 0.2)),
        safety_min_parse_rate=float(t.get("safety_min_parse_rate", 0.8)),
        safety_max_punctuation_loop_rate=float(t.get("safety_max_punctuation_loop_rate", 0.05)),
        safety_max_zero_std_frac=float(t.get("safety_max_zero_std_frac", 0.95)),
        safety_start_step=int(t.get("safety_start_step", 2)),
        tinker_api_key_env=str(t.get("tinker_api_key_env", "TINKER_API_KEY")),
    )


def preflight(config: TrainConfig) -> None:
    rows = read_jsonl(config.input_path)
    groups = group_rows(rows, split=config.split)
    if not groups:
        raise SystemExit(f"no training groups found for split={config.split!r}")
    sizes = sorted({len(g) for g in groups})
    outcomes = sum(int(g[0]["outcome"]) for g in groups) / len(groups)
    print(
        json.dumps(
            {
                "input": str(config.input_path),
                "split": config.split,
                "groups": len(groups),
                "group_sizes": sizes,
                "yes_rate": outcomes,
                "groups_per_step": config.groups_per_step,
                "max_steps": config.max_steps,
                "lambda_invariance": config.lambda_invariance,
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def select_batch(groups: list[list[dict[str, Any]]], *, step: int, config: TrainConfig) -> list[list[dict[str, Any]]]:
    rng = random.Random(config.seed + step)
    return rng.sample(groups, k=min(config.groups_per_step, len(groups)))


def maybe_stop_for_safety(step: int, metrics: dict[str, Any], config: TrainConfig) -> str | None:
    if step < config.safety_start_step:
        return None
    if float(metrics.get("reward_parse_rate", 0.0)) < config.safety_min_parse_rate:
        return f"parse rate {metrics.get('reward_parse_rate')} below {config.safety_min_parse_rate}"
    if float(metrics.get("reward_punctuation_loop_rate", 0.0)) > config.safety_max_punctuation_loop_rate:
        return (
            f"punctuation loop rate {metrics.get('reward_punctuation_loop_rate')} above "
            f"{config.safety_max_punctuation_loop_rate}"
        )
    if float(metrics.get("frac_reward_zero_std", 0.0)) > config.safety_max_zero_std_frac:
        return (
            f"zero-std reward fraction {metrics.get('frac_reward_zero_std')} above "
            f"{config.safety_max_zero_std_frac}"
        )
    return None


def train(config: TrainConfig) -> None:
    rows = read_jsonl(config.input_path)
    groups = group_rows(rows, split=config.split)
    if not groups:
        raise RuntimeError("no groups to train on")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = config.results_dir / config.run_name / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(config.__dict__, default=str, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    trainer = TinkerTrainer(config)
    checkpoint_rows: list[dict[str, Any]] = []
    started = time.time()
    for step in range(1, config.max_steps + 1):
        batch = select_batch(groups, step=step, config=config)
        result = trainer.train_step(step, batch)
        metrics = result["metrics"]
        metrics["elapsed_s"] = round(time.time() - started, 3)
        write_jsonl(run_dir / "metrics.jsonl", [metrics])
        write_csv(run_dir / "rollouts.csv", result["prediction_rows"])
        print(json.dumps(metrics, sort_keys=True), flush=True)

        reason = maybe_stop_for_safety(step, metrics, config)
        if reason:
            print(f"Tinker GRPO safety stop at step {step}: {reason}", file=sys.stderr)
            break

        if config.save_every > 0 and step % config.save_every == 0:
            path = trainer.save_checkpoint(step)
            checkpoint = {"step": step, "path": path, "elapsed_s": round(time.time() - started, 3)}
            checkpoint_rows.append(checkpoint)
            write_jsonl(run_dir / "checkpoints.jsonl", [checkpoint])
            print(f"checkpoint step {step}: {path}", file=sys.stderr)

    if checkpoint_rows:
        (config.output_dir / "latest_checkpoint.json").write_text(
            json.dumps(checkpoint_rows[-1], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train grouped GRPO on Tinker.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    preflight(config)
    if args.preflight:
        print("preflight ok")
        return 0
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
