"""ForecastBench puller (offline, against ``datasets/`` snapshots in this repo).

ForecastBench publishes paired snapshots: ``question_sets/<date>-llm.json`` and
``resolution_sets/<date>_resolution_set.json``. Each resolution-set carries a
``question_set`` field naming the question file it resolves.

This module walks every resolution-set in ``datasets/resolution_sets`` and emits
unified ``Question`` rows for every resolved binary question. The same question
can appear across multiple snapshots (it gets re-resolved by every later set
that includes it); we keep the earliest valid snapshot in which the question
*resolved*, since that's the most-conservative leakage profile.

We do not hit Hugging Face; everything we need is in-tree. If you want fresher
snapshots, drop them into ``datasets/`` and re-run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import GPT_OSS_120B_CUTOFF, Question, parse_iso_date, write_jsonl


DATASETS_DIR = Path("datasets")

# ForecastBench question templates contain literal {resolution_date} and
# {forecast_due_date} tokens that need to be substituted with the question's
# actual dates before the question is shown to a model. We do that
# substitution here in the puller so every downstream stage sees rendered
# question text.
_PLACEHOLDER_RESOLUTION = "{resolution_date}"
_PLACEHOLDER_FORECAST = "{forecast_due_date}"


def render_question_dates(
    question_text: str, *, freeze_date: str, resolved_at: str
) -> str:
    """Substitute date placeholders. Both args are YYYY-MM-DD strings."""

    rendered = question_text
    if _PLACEHOLDER_RESOLUTION in rendered:
        rendered = rendered.replace(_PLACEHOLDER_RESOLUTION, resolved_at)
    if _PLACEHOLDER_FORECAST in rendered:
        rendered = rendered.replace(_PLACEHOLDER_FORECAST, freeze_date)
    return rendered


def _is_binary_outcome(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return float(value) in (0.0, 1.0)
    if isinstance(value, str):
        return value.strip().lower() in {"0", "1", "0.0", "1.0", "yes", "no", "true", "false"}
    return False


def _normalize_outcome(value: Any) -> int | None:
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


def load_question_set(path: Path) -> dict[str, dict[str, Any]]:
    """Return ``{question_id: question_dict}`` for one question-set file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    out: dict[str, dict[str, Any]] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = q.get("id")
        if not qid:
            continue
        copy = dict(q)
        copy["_question_set"] = path.name
        out[str(qid)] = copy
    return out


def iter_resolved_rows(
    datasets_dir: Path = DATASETS_DIR,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
) -> list[Question]:
    """Walk every resolution-set under ``datasets_dir``; emit unified rows.

    Filters: binary resolution, post-cutoff freeze and resolution dates,
    valid date ordering (freeze < resolved). Earliest-valid-snapshot wins
    on duplicates; later (re-)resolutions are dropped.
    """

    res_dir = datasets_dir / "resolution_sets"
    qs_dir = datasets_dir / "question_sets"
    if not res_dir.exists() or not qs_dir.exists():
        raise FileNotFoundError(
            f"Expected {res_dir} and {qs_dir}; check that the repo's datasets/ is intact"
        )

    qs_cache: dict[str, dict[str, dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    out: list[Question] = []
    skipped = Counter()

    res_files = sorted(res_dir.glob("*_resolution_set.json"))
    for res_path in res_files:
        payload = json.loads(res_path.read_text(encoding="utf-8"))
        question_set_name = payload.get("question_set")
        if not question_set_name:
            skipped["resolution_set_missing_question_set"] += 1
            continue

        if question_set_name not in qs_cache:
            qs_path = qs_dir / question_set_name
            if not qs_path.exists():
                skipped["question_set_file_missing"] += 1
                continue
            qs_cache[question_set_name] = load_question_set(qs_path)

        questions = qs_cache[question_set_name]

        for resolution in payload.get("resolutions", []):
            if not isinstance(resolution, dict):
                skipped["resolution_not_dict"] += 1
                continue
            if not resolution.get("resolved"):
                skipped["not_resolved"] += 1
                continue
            if not _is_binary_outcome(resolution.get("resolved_to")):
                skipped["non_binary_outcome"] += 1
                continue

            outcome = _normalize_outcome(resolution.get("resolved_to"))
            if outcome is None:
                skipped["unparseable_outcome"] += 1
                continue

            # Skip ForecastBench "combination" questions: rows where the
            # native id is a list of two ids (e.g. ['15796', '19397']).
            # These are joint-probability meta-prompts ("we are presenting
            # you with two probability questions...") not standalone binary
            # questions — they're not paraphraseable as forecasting questions
            # and the constituent binary questions are already in the dataset.
            raw_id = resolution.get("id")
            if isinstance(raw_id, list):
                skipped["combo_question"] += 1
                continue

            qid = str(raw_id or "")
            if qid.startswith("[") or "," in qid:
                # Defensive: catch combo ids that arrive pre-stringified.
                skipped["combo_question"] += 1
                continue

            question = questions.get(qid)
            if question is None:
                skipped["question_missing_for_resolution"] += 1
                continue

            resolved_at = parse_iso_date(resolution.get("resolution_date"))
            freeze_date = parse_iso_date(question.get("freeze_datetime"))
            if resolved_at is None or freeze_date is None:
                skipped["unparseable_dates"] += 1
                continue
            if freeze_date > resolved_at:
                skipped["freeze_after_resolution"] += 1
                continue
            if freeze_date <= cutoff_date or resolved_at <= cutoff_date:
                skipped["pre_cutoff"] += 1
                continue

            unified_id = f"forecastbench::{qid}"
            if unified_id in seen_ids:
                skipped["duplicate_question_id"] += 1
                continue
            seen_ids.add(unified_id)

            raw_question_text = str(question.get("question", ""))
            rendered_question = render_question_dates(
                raw_question_text,
                freeze_date=freeze_date,
                resolved_at=resolved_at,
            )
            # Defensive: if either placeholder slipped through (different
            # template not in our substitution list), drop the row rather
            # than silently feed a templated question to Claude.
            if "{" in rendered_question and "}" in rendered_question:
                # Best-effort heuristic for residual placeholders.
                if (_PLACEHOLDER_RESOLUTION in rendered_question
                        or _PLACEHOLDER_FORECAST in rendered_question
                        or re.search(r"\{[a-z_]+\}", rendered_question)):
                    skipped["unresolved_placeholder"] += 1
                    continue

            try:
                out.append(
                    Question(
                        id=unified_id,
                        question=rendered_question,
                        outcome=outcome,
                        freeze_date=freeze_date,
                        resolved_at=resolved_at,
                        source="forecastbench",
                        url=question.get("url"),
                        background=question.get("background"),
                        resolution_criteria=question.get("resolution_criteria"),
                        categories=[str(question.get("source"))]
                        if question.get("source")
                        else [],
                        raw={
                            "forecastbench_question_set": question_set_name,
                            "forecastbench_resolution_snapshot": res_path.name,
                            "fb_source": question.get("source"),
                            "freeze_datetime": question.get("freeze_datetime"),
                            "market_info_open_datetime": question.get(
                                "market_info_open_datetime"
                            ),
                            "market_info_close_datetime": question.get(
                                "market_info_close_datetime"
                            ),
                            "market_info_resolution_criteria": question.get(
                                "market_info_resolution_criteria"
                            ),
                            "freeze_datetime_value": question.get("freeze_datetime_value"),
                        },
                    )
                )
            except ValueError as exc:
                # Don't crash on a malformed row; skip and tally.
                skipped[f"schema_reject:{exc}"] += 1
                continue

    return _sort_and_finalize(out), skipped


def _sort_and_finalize(rows: list[Question]) -> list[Question]:
    rows.sort(key=lambda r: (r.resolved_at, r.id))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull ForecastBench resolved binary questions into the unified schema."
    )
    parser.add_argument(
        "--datasets-dir",
        default=str(DATASETS_DIR),
        help="Path to the in-tree datasets/ directory.",
    )
    parser.add_argument(
        "--cutoff",
        default=GPT_OSS_120B_CUTOFF,
        help="Strict-inequality cutoff date (default: gpt-oss-120b's).",
    )
    parser.add_argument(
        "--output",
        default="data/raw/forecastbench.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    rows, skipped = iter_resolved_rows(
        datasets_dir=Path(args.datasets_dir),
        cutoff_date=args.cutoff,
    )
    n = write_jsonl(rows, args.output)

    print(f"forecastbench: wrote {n} questions to {args.output}")
    print(f"sources:")
    for src, count in Counter(
        (r.raw.get("fb_source") or "?") for r in rows
    ).most_common():
        print(f"  {src:>14s}: {count}")
    print(f"date range: {rows[0].freeze_date} to {rows[-1].resolved_at}" if rows else "no rows")
    print(f"skipped: {dict(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
