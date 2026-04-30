#!/usr/bin/env python3
"""Audit ForecastBench paraphrase data before expensive model runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from frame_invariance.training.data import (
    SplitConfig,
    group_paraphrases,
    load_grouped_prompt_splits,
    parse_datetime,
)
from frame_invariance.training.train_grpo import load_config


KNOWN_BAD_PATTERNS = (
    r"\{(?:resolution_date|forecast_due_date)\}",
    r"matches or exceeds their ranking",
    r"stand as high or higher",
    r"as high or higher than their ranking",
    r"come out on top in second place",
    r"victory in second place",
    r"final the",
    r"achieve a 0-week",
    r"Federal Reverse Banks",
    r"\bEOY\b",
    r"\bEOY2025\b",
    r"\bthis year\b",
    r"\b(?:October 31|November 30|December 31)(?!,?\s*20\d{2})",
)
BACKGROUND_UPDATE_DATE = re.compile(r"\bUpdate\s+(20\d{2}-\d{2}-\d{2})\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ForecastBench data/config readiness.")
    parser.add_argument("--config", default="configs/training/grpo_forecastbench.yaml")
    parser.add_argument(
        "--model-cutoff",
        default="2024-06-30",
        help="Conservative Qwen cutoff guard; all forecast/resolution dates must be after it.",
    )
    parser.add_argument("--human-file", default="human/questions_and_paraphrases.txt")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv_questions(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["question"] for row in csv.DictReader(handle)]


def fail(failures: list[str], message: str) -> None:
    failures.append(message)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    paraphrase_path = Path(data_cfg["paraphrase_path"])
    group_size = int(data_cfg.get("group_size", 5))
    cutoff = parse_datetime(args.model_cutoff)
    if cutoff is None:
        raise ValueError(f"Invalid --model-cutoff {args.model_cutoff!r}")

    rows = read_jsonl(paraphrase_path)
    groups = group_paraphrases(rows, expected_group_size=group_size)
    questions = [str(row["question"]) for row in rows]
    failures: list[str] = []

    if len(questions) != len(set(questions)):
        duplicate_count = len(questions) - len(set(questions))
        fail(failures, f"Found {duplicate_count} exact duplicate question texts")

    for pattern in KNOWN_BAD_PATTERNS:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        matches = [row["variant_id"] for row in rows if regex.search(str(row["question"]))]
        if matches:
            fail(
                failures,
                f"Bad question pattern {pattern!r} matched {len(matches)} variants, e.g. {matches[:5]}",
            )

    for gid, items in groups.items():
        first = items[0]
        freeze = parse_datetime(first.get("freeze_datetime"))
        resolved = parse_datetime(first.get("resolved_at"))
        if freeze is None or resolved is None:
            fail(failures, f"Group {gid} has invalid forecast/resolution dates")
            continue
        if not (cutoff < freeze < resolved):
            fail(
                failures,
                f"Group {gid} has unsafe dates: cutoff={cutoff.date()}, "
                f"freeze={freeze.date()}, resolved={resolved.date()}",
            )
        if not (first.get("background") or first.get("resolution_criteria")):
            fail(failures, f"Group {gid} has no background/resolution context")
        for match in BACKGROUND_UPDATE_DATE.finditer(str(first.get("background") or "")):
            update_date = parse_datetime(match.group(1))
            if update_date is not None and update_date > freeze:
                fail(
                    failures,
                    f"Group {gid} background has update date {update_date.date()} "
                    f"after forecast date {freeze.date()}",
                )
        forbidden_prompt_text = json.dumps(first, ensure_ascii=False)
        if "resolved_to" in forbidden_prompt_text:
            fail(failures, f"Group {gid} still contains resolved_to in paraphrase row")

    split_cfg = SplitConfig(
        train_frac=float(data_cfg.get("train_frac", 0.8)),
        val_frac=float(data_cfg.get("val_frac", 0.1)),
        seed=int(data_cfg.get("seed", 17)),
        stratify_by=str(data_cfg.get("stratify_by", "resolved_month")),
    )
    splits = load_grouped_prompt_splits(paraphrase_path, split_cfg, expected_group_size=group_size)
    split_summary: dict[str, dict[str, float | int]] = {}
    for name, split_rows in splits.items():
        split_groups = group_paraphrases(split_rows, expected_group_size=group_size)
        outcomes = [int(items[0]["outcome"]) for items in split_groups.values()]
        split_summary[name] = {
            "groups": len(split_groups),
            "rows": len(split_rows),
            "yes_rate": mean(outcomes) if outcomes else 0.0,
        }
        if outcomes and not (0.2 <= split_summary[name]["yes_rate"] <= 0.8):
            fail(failures, f"{name} base rate outside [0.2, 0.8]: {split_summary[name]['yes_rate']:.3f}")

    csv_path = (
        paraphrase_path.parent.parent
        / "processed"
        / "forecastbench_current_after_2025-08-31_5x_paraphrases.csv"
    )
    csv_questions = read_csv_questions(csv_path)
    if csv_questions is not None and csv_questions != questions:
        fail(failures, f"CSV questions do not match {paraphrase_path}")

    human_path = Path(args.human_file)
    if human_path.exists():
        human_questions = human_path.read_text(encoding="utf-8").splitlines()
        if human_questions != questions:
            fail(failures, f"Human review file {human_path} does not match paraphrase JSONL")

    sources = Counter(str(items[0].get("source")) for items in groups.values())
    freeze_dates = [parse_datetime(items[0].get("freeze_datetime")) for items in groups.values()]
    resolved_dates = [parse_datetime(items[0].get("resolved_at")) for items in groups.values()]
    print("audit ok" if not failures else "audit failed")
    print("rows:", len(rows), "groups:", len(groups), "group_size:", group_size)
    print("sources:", dict(sorted(sources.items())))
    print(
        "date range:",
        min(date for date in freeze_dates if date).date(),
        "to",
        max(date for date in resolved_dates if date).date(),
    )
    print("splits:", json.dumps(split_summary, sort_keys=True))

    if failures:
        for message in failures:
            print("ERROR:", message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
