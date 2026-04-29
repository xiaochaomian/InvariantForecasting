"""Dataset loading and deterministic group splits for paraphrased ForecastBench data."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prompts import build_messages, build_prompt

PLACEHOLDER_PATTERN = re.compile(r"\{(?:resolution_date|forecast_due_date)\}")
CONTEXT_FIELDS = (
    "background",
    "resolution_criteria",
    "url",
    "forecastbench_question_set",
    "forecastbench_resolution_snapshot",
    "freeze_datetime",
    "market_info_open_datetime",
    "market_info_close_datetime",
    "market_info_resolution_criteria",
)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 17

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_paraphrases(rows: list[dict[str, Any]], expected_group_size: int = 5) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["id"])].append(row)

    bad_sizes = {gid: len(items) for gid, items in groups.items() if len(items) != expected_group_size}
    if bad_sizes:
        preview = dict(list(bad_sizes.items())[:10])
        raise ValueError(f"Expected {expected_group_size} variants per group; bad groups: {preview}")

    for gid, items in groups.items():
        items.sort(key=lambda item: int(item["variant_index"]))
        indexes = [int(item["variant_index"]) for item in items]
        if indexes != list(range(expected_group_size)):
            raise ValueError(f"Group {gid} has variant indexes {indexes}")
        outcomes = {int(item["outcome"]) for item in items}
        if len(outcomes) != 1:
            raise ValueError(f"Group {gid} has inconsistent outcomes: {outcomes}")
    return dict(groups)


def split_group_ids(group_ids: list[str], config: SplitConfig) -> dict[str, set[str]]:
    if not (0.0 < config.train_frac < 1.0 and 0.0 <= config.val_frac < 1.0):
        raise ValueError("invalid split fractions")
    if config.test_frac < 0.0:
        raise ValueError("train_frac + val_frac must be <= 1")

    shuffled = list(group_ids)
    random.Random(config.seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * config.train_frac))
    n_val = int(round(n * config.val_frac))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": set(train), "validation": set(val), "test": set(test)}


def make_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    if PLACEHOLDER_PATTERN.search(question):
        raise ValueError(
            f"Question {row.get('id')!r} still contains an unresolved date placeholder: {question!r}"
        )
    if not row.get("freeze_datetime"):
        raise ValueError(f"Question {row.get('id')!r} is missing freeze_datetime/forecast date")
    if not row.get("background") and not row.get("resolution_criteria"):
        raise ValueError(f"Question {row.get('id')!r} is missing ForecastBench context")

    prompt_source = dict(row)
    prompt_source["question"] = question
    output = {
        "prompt": build_prompt(prompt_source),
        "messages": build_messages(prompt_source),
        "question": row["question"],
        "id": str(row["id"]),
        "source_question_index": int(row["source_question_index"]),
        "variant_index": int(row["variant_index"]),
        "variant_id": row["variant_id"],
        "is_original": bool(row["is_original"]),
        "source": row.get("source"),
        "outcome": int(row["outcome"]),
        "resolved_at": row.get("resolved_at"),
        "paraphrase_method": row.get("paraphrase_method"),
    }
    for field in CONTEXT_FIELDS:
        output[field] = row.get(field)
    return output


def load_grouped_prompt_splits(
    paraphrase_path: str | Path,
    split_config: SplitConfig | None = None,
    expected_group_size: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    cfg = split_config or SplitConfig()
    groups = group_paraphrases(read_jsonl(paraphrase_path), expected_group_size=expected_group_size)
    split_ids = split_group_ids(sorted(groups), cfg)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split_name, ids in split_ids.items():
        for gid in sorted(ids):
            splits[split_name].extend(make_prompt_row(row) for row in groups[gid])
    return splits
