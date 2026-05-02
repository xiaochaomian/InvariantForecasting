"""Mantic Q2 2025 AIB test-set handler.

Mantic's blog post says the question list "can be accessed on GitHub" but does
not link the file directly. We support two ingestion paths:

  1. **Metaculus tournament-slug pull (preferred).** The Q2 2025 AI Forecasting
     Benchmark Tournament has a Metaculus slug; pull the resolved binary
     subset and tag rows with ``source="mantic_q2_aib"``. Use this if you can
     identify the tournament slug.

  2. **GitHub drop-in.** If you find the exact list (a JSON, CSV, or plain
     ID-list), drop it into ``data/raw/mantic_aib.<ext>`` and run
     ``ingest_drop_in()`` to convert it to unified ``Question`` rows by joining
     against an already-pulled Metaculus JSONL.

The point is to carve out a *fixed test split* that lets us report a number on
exactly the same questions Mantic reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from .metaculus import ENV_TOKEN, pull as metaculus_pull
from .schema import GPT_OSS_120B_CUTOFF, Question, read_jsonl, write_jsonl


def pull_via_tournament(
    tournament_slug: str,
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    cache_dir: Path | None = Path("data/raw/metaculus_cache"),
    token: str | None = None,
) -> list[Question]:
    """Pull AIB Q2 2025 questions via the Metaculus tournament filter.

    The default Metaculus AIB Q2 2025 slug is empirically ``"aibq2"`` but
    Metaculus has historically renamed slugs; verify on their tournament index
    if the pull comes up empty.
    """

    rows, _ = metaculus_pull(
        tournaments=tournament_slug,
        tournament_label="mantic_q2_aib",
        cutoff_date=cutoff_date,
        cache_dir=cache_dir,
        token=token,
    )
    # Re-stamp source so downstream filtering by source works.
    out: list[Question] = []
    for r in rows:
        d = r.to_dict()
        d["source"] = "mantic_q2_aib"
        d["id"] = d["id"].replace("metaculus::", "mantic_q2_aib::")
        if "tournament:mantic_q2_aib" not in d["categories"]:
            d["categories"] = d["categories"] + ["tournament:mantic_q2_aib"]
        out.append(Question(**d))
    return out


def _load_drop_in_ids(path: Path) -> list[str]:
    """Best-effort parse of a drop-in question-list file.

    Supports JSON arrays of ints/strings, JSONL with ``id`` fields, and plain
    text (one id per line). Returns Metaculus post or question IDs as strings.
    """

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x.get("id") if isinstance(x, dict) else x) for x in payload]
        if isinstance(payload, dict) and "questions" in payload:
            return [str(q.get("id") or q.get("post_id")) for q in payload["questions"]]
        raise ValueError(f"Don't know how to parse {path} JSON shape")
    if suffix == ".jsonl":
        ids: list[str] = []
        for line in text.splitlines():
            if line.strip():
                obj = json.loads(line)
                ids.append(str(obj.get("id") or obj.get("post_id")))
        return ids
    if suffix == ".csv":
        ids: list[str] = []
        for row in csv.DictReader(text.splitlines()):
            for key in ("id", "post_id", "question_id", "metaculus_id"):
                if row.get(key):
                    ids.append(str(row[key]))
                    break
        return ids
    # Fall back to plain text
    return [line.strip() for line in text.splitlines() if line.strip()]


def ingest_drop_in(
    drop_in_path: Path,
    *,
    metaculus_jsonl: Path,
) -> list[Question]:
    """Match a drop-in id list against an already-pulled Metaculus JSONL.

    Marks matched rows as ``source="mantic_q2_aib"``. Unmatched ids are
    reported on stderr; you can re-pull Metaculus with a wider tournament
    filter if many ids are missing.
    """

    ids = set(_load_drop_in_ids(drop_in_path))
    if not ids:
        return []

    metaculus_rows = read_jsonl(metaculus_jsonl)
    matched: list[Question] = []
    seen: set[str] = set()
    for r in metaculus_rows:
        post_id = str(r.raw.get("metaculus_post_id") or "")
        question_id = str(r.raw.get("metaculus_question_id") or "")
        if post_id in ids or question_id in ids:
            d = r.to_dict()
            d["source"] = "mantic_q2_aib"
            d["id"] = d["id"].replace("metaculus::", "mantic_q2_aib::")
            if "tournament:mantic_q2_aib" not in d["categories"]:
                d["categories"] = d["categories"] + ["tournament:mantic_q2_aib"]
            try:
                matched.append(Question(**d))
                seen.add(post_id)
                seen.add(question_id)
            except ValueError:
                continue
    missing = ids - seen - {""}
    if missing:
        print(
            f"warning: {len(missing)} drop-in ids not found in {metaculus_jsonl}; "
            f"re-pull metaculus with broader filter or check the list",
            file=sys.stderr,
        )
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Mantic Q2 2025 AIB test split via tournament slug or drop-in list."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    sub_t = sub.add_parser("tournament", help="Pull via Metaculus tournament slug.")
    sub_t.add_argument("--slug", default="aibq2")
    sub_t.add_argument("--cutoff", default=GPT_OSS_120B_CUTOFF)
    sub_t.add_argument("--cache-dir", default="data/raw/metaculus_cache")
    sub_t.add_argument("--token", default=None,
                       help=f"Metaculus API token; falls back to ${ENV_TOKEN} env var.")
    sub_t.add_argument("--output", default="data/raw/mantic_q2_aib.jsonl")

    sub_d = sub.add_parser("drop-in", help="Match a hand-supplied id list against metaculus.jsonl.")
    sub_d.add_argument("--drop-in", required=True)
    sub_d.add_argument("--metaculus-jsonl", default="data/raw/metaculus.jsonl")
    sub_d.add_argument("--output", default="data/raw/mantic_q2_aib.jsonl")

    args = parser.parse_args()
    if args.mode == "tournament":
        token = args.token or os.environ.get(ENV_TOKEN)
        rows = pull_via_tournament(
            args.slug,
            cutoff_date=args.cutoff,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            token=token,
        )
    else:
        rows = ingest_drop_in(
            Path(args.drop_in),
            metaculus_jsonl=Path(args.metaculus_jsonl),
        )
    n = write_jsonl(rows, args.output)
    print(f"mantic_q2_aib: wrote {n} questions to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
