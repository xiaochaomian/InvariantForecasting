"""Dataset loading and deterministic group splits for paraphrased ForecastBench data."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
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
FORBIDDEN_PROMPT_MARKERS = (
    "resolved_to",
    "forecastbench_resolution_snapshot",
    "_resolution_set.json",
)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 17
    stratify_by: str = "resolved_month"

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


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


def _allocate_counts(bucket_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    total = sum(bucket_sizes.values())
    if total == 0:
        return {key: 0 for key in bucket_sizes}

    raw_counts = {
        key: size * target_total / total
        for key, size in bucket_sizes.items()
    }
    counts = {key: int(raw_counts[key]) for key in bucket_sizes}
    remaining = target_total - sum(counts.values())
    remainders = sorted(
        bucket_sizes,
        key=lambda key: (raw_counts[key] - counts[key], bucket_sizes[key], key),
        reverse=True,
    )
    for key in remainders:
        if remaining <= 0:
            break
        if counts[key] < bucket_sizes[key]:
            counts[key] += 1
            remaining -= 1
    return counts


def _stratification_key(items: list[dict[str, Any]], stratify_by: str) -> str:
    first = items[0]
    if stratify_by == "resolved_month":
        resolved_at = parse_datetime(first.get("resolved_at"))
        if resolved_at is None:
            raise ValueError(f"Group {first.get('id')!r} has invalid resolved_at for stratification")
        return resolved_at.strftime("%Y-%m")
    if stratify_by == "resolved_date":
        resolved_at = parse_datetime(first.get("resolved_at"))
        if resolved_at is None:
            raise ValueError(f"Group {first.get('id')!r} has invalid resolved_at for stratification")
        return resolved_at.strftime("%Y-%m-%d")
    if stratify_by == "none":
        return "all"
    raise ValueError(f"Unsupported split stratification: {stratify_by!r}")


def split_group_ids(
    group_ids: list[str],
    config: SplitConfig,
    groups: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, set[str]]:
    if not (0.0 < config.train_frac < 1.0 and 0.0 <= config.val_frac < 1.0):
        raise ValueError("invalid split fractions")
    if config.test_frac < 0.0:
        raise ValueError("train_frac + val_frac must be <= 1")

    n = len(group_ids)
    n_train = int(round(n * config.train_frac))
    n_val = int(round(n * config.val_frac))
    n_test = n - n_train - n_val

    if groups is None or config.stratify_by == "none":
        shuffled = list(group_ids)
        random.Random(config.seed).shuffle(shuffled)
        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        return {"train": set(train), "validation": set(val), "test": set(test)}

    buckets: dict[str, list[str]] = defaultdict(list)
    for gid in group_ids:
        buckets[_stratification_key(groups[gid], config.stratify_by)].append(gid)
    for key, ids in buckets.items():
        ids.sort()
        random.Random(f"{config.seed}:{key}").shuffle(ids)

    bucket_sizes = {key: len(ids) for key, ids in buckets.items()}
    train_counts = _allocate_counts(bucket_sizes, n_train)
    remaining_sizes = {key: bucket_sizes[key] - train_counts[key] for key in buckets}
    val_counts = _allocate_counts(remaining_sizes, n_val)

    split_ids: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for key, ids in buckets.items():
        train_end = train_counts[key]
        val_end = train_end + val_counts[key]
        split_ids["train"].update(ids[:train_end])
        split_ids["validation"].update(ids[train_end:val_end])
        split_ids["test"].update(ids[val_end:])

    actual = {name: len(ids) for name, ids in split_ids.items()}
    expected = {"train": n_train, "validation": n_val, "test": n_test}
    if actual != expected:
        raise RuntimeError(f"Stratified split produced {actual}, expected {expected}")
    return split_ids


def make_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"])
    if PLACEHOLDER_PATTERN.search(question):
        raise ValueError(
            f"Question {row.get('id')!r} still contains an unresolved date placeholder: {question!r}"
        )
    if not row.get("freeze_datetime"):
        raise ValueError(f"Question {row.get('id')!r} is missing freeze_datetime/forecast date")
    freeze_datetime = parse_datetime(row.get("freeze_datetime"))
    resolved_at = parse_datetime(row.get("resolved_at"))
    if freeze_datetime is None or resolved_at is None:
        raise ValueError(f"Question {row.get('id')!r} has invalid forecast/resolution dates")
    if freeze_datetime >= resolved_at:
        raise ValueError(
            f"Question {row.get('id')!r} has forecast date {row.get('freeze_datetime')!r} "
            f"not before resolution date {row.get('resolved_at')!r}"
        )
    market_close = parse_datetime(row.get("market_info_close_datetime"))
    if market_close is not None and market_close < freeze_datetime:
        raise ValueError(
            f"Question {row.get('id')!r} has market close {row.get('market_info_close_datetime')!r} "
            f"before forecast date {row.get('freeze_datetime')!r}"
        )
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
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker in output["prompt"]:
            raise ValueError(f"Question {row.get('id')!r} prompt contains forbidden marker {marker!r}")
    return output


def load_grouped_prompt_splits(
    paraphrase_path: str | Path,
    split_config: SplitConfig | None = None,
    expected_group_size: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    cfg = split_config or SplitConfig()
    groups = group_paraphrases(read_jsonl(paraphrase_path), expected_group_size=expected_group_size)
    split_ids = split_group_ids(sorted(groups), cfg, groups)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split_name, ids in split_ids.items():
        for gid in sorted(ids):
            splits[split_name].extend(make_prompt_row(row) for row in groups[gid])
    return splits
