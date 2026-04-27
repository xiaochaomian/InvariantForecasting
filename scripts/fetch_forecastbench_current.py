#!/usr/bin/env python3
"""Fetch ForecastBench/current resolved binary questions after a cutoff date."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HF_REPO = "forecastingresearch/forecastbench-datasets"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
DEFAULT_SINCE = "2025-08-31"
DEFAULT_MAX_QUESTIONS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ForecastBench resolved binary questions after a cutoff date."
    )
    parser.add_argument("--since", default=DEFAULT_SINCE, help="Keep resolution_date strictly after YYYY-MM-DD.")
    parser.add_argument("--max-questions", type=int, default=DEFAULT_MAX_QUESTIONS, help="Maximum output rows. Use 0 for no cap.")
    parser.add_argument("--sort", choices=["oldest", "newest"], default="newest", help="Which rows to keep first when capped.")
    parser.add_argument("--raw-dir", default="data/raw/forecastbench_current")
    parser.add_argument("--jsonl", default="data/raw/forecastbench_current_after_2025-08-31.jsonl")
    parser.add_argument("--csv", default="data/processed/forecastbench_current_after_2025-08-31.csv")
    return parser.parse_args()


def parse_date(value: Any) -> datetime | None:
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


def request_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "InvariantForecasting/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def list_dir(path: str) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote(path.strip("/"))
    payload = request_json(f"{HF_API}/{encoded}?recursive=false")
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected Hugging Face tree response for {path}: {type(payload)}")
    return payload


def download_json(repo_path: str, raw_dir: Path) -> Any:
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / repo_path.replace("/", "__")
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    url = f"{HF_BASE}/resolve/main/{repo_path}"
    print(f"downloading {repo_path}", file=sys.stderr)
    payload = request_json(url)
    local_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def snapshot_date(path: str) -> datetime | None:
    name = Path(path).name
    return parse_date(name[:10])


def is_binary_resolution(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return float(value) in (0.0, 1.0)
    return str(value).strip().lower() in {"0", "1", "0.0", "1.0", "yes", "no", "true", "false"}


def normalize_outcome(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return int(float(value))
    text = str(value).strip().lower()
    if text in {"1", "1.0", "yes", "true"}:
        return 1
    if text in {"0", "0.0", "no", "false"}:
        return 0
    return None


def make_row(question: dict[str, Any], resolution: dict[str, Any], resolution_snapshot: str) -> dict[str, Any] | None:
    outcome = normalize_outcome(resolution.get("resolved_to"))
    if outcome is None:
        return None
    resolved_at = parse_date(resolution.get("resolution_date"))
    if resolved_at is None:
        return None

    return {
        "id": resolution.get("id") or question.get("id"),
        "question": question.get("question"),
        "background": question.get("background"),
        "resolution_criteria": question.get("resolution_criteria"),
        "source": resolution.get("source") or question.get("source"),
        "url": question.get("url"),
        "resolved_at": resolved_at.date().isoformat(),
        "outcome": outcome,
        "resolved_to": resolution.get("resolved_to"),
        "forecastbench_resolution_snapshot": resolution_snapshot,
        "forecastbench_question_set": question.get("_question_set"),
        "freeze_datetime": question.get("freeze_datetime"),
        "market_info_open_datetime": question.get("market_info_open_datetime"),
        "market_info_close_datetime": question.get("market_info_close_datetime"),
        "market_info_resolution_criteria": question.get("market_info_resolution_criteria"),
    }


def write_outputs(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = [
        "id",
        "question",
        "background",
        "resolution_criteria",
        "source",
        "url",
        "resolved_at",
        "outcome",
        "resolved_to",
        "forecastbench_resolution_snapshot",
        "forecastbench_question_set",
        "freeze_datetime",
        "market_info_open_datetime",
        "market_info_close_datetime",
        "market_info_resolution_criteria",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    since = parse_date(args.since)
    if since is None:
        print(f"error: invalid --since date {args.since!r}", file=sys.stderr)
        return 2

    raw_dir = Path(args.raw_dir)
    resolution_files = [
        item["path"]
        for item in list_dir("datasets/resolution_sets")
        if item.get("type") == "file" and item.get("path", "").endswith("_resolution_set.json")
    ]
    resolution_files = [path for path in resolution_files if (snapshot_date(path) or datetime.min.replace(tzinfo=timezone.utc)) > since]
    resolution_files.sort()

    rows_by_id: dict[str, dict[str, Any]] = {}
    question_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for resolution_path in resolution_files:
        resolution_payload = download_json(resolution_path, raw_dir)
        question_set = resolution_payload.get("question_set")
        if not question_set:
            print(f"skipping {resolution_path}: missing question_set", file=sys.stderr)
            continue

        question_path = f"datasets/question_sets/{question_set}"
        if question_set not in question_cache:
            question_payload = download_json(question_path, raw_dir)
            questions = question_payload.get("questions", [])
            question_map: dict[str, dict[str, Any]] = {}
            for question in questions:
                if not isinstance(question, dict):
                    continue
                copied = dict(question)
                copied["_question_set"] = question_set
                question_map[str(copied.get("id"))] = copied
            question_cache[question_set] = question_map
        else:
            question_map = question_cache[question_set]

        kept_from_snapshot = 0
        for resolution in resolution_payload.get("resolutions", []):
            if not isinstance(resolution, dict) or not resolution.get("resolved"):
                continue
            if not is_binary_resolution(resolution.get("resolved_to")):
                continue
            resolved_at = parse_date(resolution.get("resolution_date"))
            if resolved_at is None or resolved_at <= since:
                continue
            question_id = str(resolution.get("id"))
            question = question_map.get(question_id)
            if question is None:
                continue
            row = make_row(question, resolution, Path(resolution_path).name)
            if row is None:
                continue
            rows_by_id[str(row["id"])] = row
            kept_from_snapshot += 1
        print(f"{Path(resolution_path).name}: kept {kept_from_snapshot}", file=sys.stderr)

    rows = list(rows_by_id.values())
    reverse = args.sort == "newest"
    rows.sort(key=lambda row: (row["resolved_at"], row["id"]), reverse=reverse)
    if args.max_questions > 0:
        rows = rows[: args.max_questions]

    write_outputs(rows, Path(args.jsonl), Path(args.csv))
    print(f"kept {len(rows)} ForecastBench/current rows resolved after {since.date()}")
    print(f"jsonl: {args.jsonl}")
    print(f"csv:   {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
