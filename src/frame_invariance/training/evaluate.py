"""Evaluate base or post-trained forecasting models on paraphrase groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .data import SplitConfig, load_grouped_prompt_splits
from .reward import parse_probability
from .train_grpo import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecast probabilities on paraphrased test groups.")
    parser.add_argument("--config", default="configs/training/grpo_forecastbench.yaml")
    parser.add_argument("--model", required=True, help="HF model name or local checkpoint/adapter path.")
    parser.add_argument("--base-model", default=None, help="Base model for loading a local PEFT adapter.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit-groups", type=int, default=0, help="Debug cap on number of question groups.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock probabilities, no model load.")
    return parser.parse_args()


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_model_and_tokenizer(model_name: str, base_model: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_name)
    adapter_config = model_path / "adapter_config.json"
    tokenizer_name = base_model or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if adapter_config.exists():
        if base_model is None:
            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("Local adapter checkpoint requires --base-model or base_model_name_or_path.")
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def generate_outputs(
    rows: list[dict[str, Any]],
    model_name: str,
    base_model: str | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    import torch

    model, tokenizer = load_model_and_tokenizer(model_name, base_model)
    outputs: list[str] = []
    do_sample = temperature > 0.0

    for batch in batched(rows, batch_size):
        prompts = [row["prompt"] for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_length = encoded["input_ids"].shape[1]
        for sequence in generated:
            completion_tokens = sequence[input_length:]
            outputs.append(tokenizer.decode(completion_tokens, skip_special_tokens=True).strip())
    return outputs


def mock_outputs(rows: list[dict[str, Any]]) -> list[str]:
    outputs = []
    for row in rows:
        # Stable but nontrivial fake probabilities for pipeline validation.
        base = 0.65 if int(row["outcome"]) == 1 else 0.35
        offset = (int(row["variant_index"]) - 2) * 0.02
        outputs.append(f"Probability: {base + offset:.2f}")
    return outputs


def write_variant_predictions(path: Path, rows: list[dict[str, Any]], completions: list[str]) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    for row, completion in zip(rows, completions):
        probability = parse_probability(completion)
        predictions.append(
            {
                "id": row["id"],
                "variant_id": row["variant_id"],
                "variant_index": row["variant_index"],
                "question": row["question"],
                "source": row.get("source"),
                "outcome": row["outcome"],
                "resolved_at": row.get("resolved_at"),
                "probability": probability,
                "parseable": probability is not None,
                "completion": completion,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(predictions[0]) if predictions else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    return predictions


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        groups[str(prediction["id"])].append(prediction)

    group_rows: list[dict[str, Any]] = []
    for gid, items in groups.items():
        items.sort(key=lambda item: int(item["variant_index"]))
        probs = [item["probability"] for item in items if item["probability"] is not None]
        outcome = int(items[0]["outcome"])
        coverage = len(probs) / len(items)
        if probs:
            consensus = mean(probs)
            brier = (consensus - outcome) ** 2
            parasd = pstdev(probs) if len(probs) > 1 else 0.0
        else:
            consensus = None
            brier = None
            parasd = None
        group_rows.append(
            {
                "id": gid,
                "outcome": outcome,
                "n_variants": len(items),
                "n_parseable": len(probs),
                "coverage": coverage,
                "consensus_probability": consensus,
                "brier": brier,
                "parasd": parasd,
            }
        )

    parseable_groups = [row for row in group_rows if row["brier"] is not None]
    return {
        "n_groups": len(group_rows),
        "n_variant_predictions": len(predictions),
        "variant_coverage": mean(float(item["parseable"]) for item in predictions) if predictions else 0.0,
        "group_full_coverage": mean(row["coverage"] == 1.0 for row in group_rows) if group_rows else 0.0,
        "mean_brier": mean(row["brier"] for row in parseable_groups) if parseable_groups else math.nan,
        "mean_parasd": mean(row["parasd"] for row in parseable_groups) if parseable_groups else math.nan,
        "groups": group_rows,
    }


def write_group_summary(path: Path, group_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(group_rows[0]) if group_rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    splits = load_grouped_prompt_splits(
        data_cfg.get("paraphrase_path", "data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl"),
        SplitConfig(
            train_frac=float(data_cfg.get("train_frac", 0.8)),
            val_frac=float(data_cfg.get("val_frac", 0.1)),
            seed=int(data_cfg.get("seed", 17)),
        ),
        expected_group_size=int(data_cfg.get("group_size", 5)),
    )
    rows = splits[args.split]
    if args.limit_groups > 0:
        allowed = sorted({row["id"] for row in rows})[: args.limit_groups]
        rows = [row for row in rows if row["id"] in set(allowed)]

    run_name = args.run_name or Path(args.model.rstrip("/")).name.replace("/", "_")
    output_dir = Path(args.output_dir) / run_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    completions = mock_outputs(rows) if args.dry_run else generate_outputs(
        rows,
        args.model,
        args.base_model,
        args.batch_size,
        args.max_new_tokens,
        args.temperature,
    )
    predictions = write_variant_predictions(output_dir / "variant_predictions.csv", rows, completions)
    summary = summarize(predictions)
    write_group_summary(output_dir / "group_metrics.csv", summary.pop("groups"))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"variant predictions: {output_dir / 'variant_predictions.csv'}")
    print(f"group metrics:       {output_dir / 'group_metrics.csv'}")
    print(f"summary:             {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
