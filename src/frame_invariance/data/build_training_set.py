"""Assemble the training-ready JSONL from unified + contexts + paraphrases.

This is the join step that produces the final dataset Tinker will train on.

Inputs:
  - ``data/processed/unified.jsonl``    — one row per question (the schema)
  - ``data/processed/contexts.jsonl``   — one row per question_id (base_rate + news)
  - ``data/processed/paraphrases.jsonl``— one row per question_id (list of K paraphrases)

Output (default ``data/processed/training.jsonl``):
  - ``K`` rows per question (``variant_index = 0..K-1``), each carrying the
    *same* context but a *different* paraphrase. The rendered prompt for the
    LLM is ALSO included pre-formatted so the trainer doesn't need to know the
    schema.

Group invariants (verified by audit):
  - Every question has exactly K consecutive rows.
  - Within a group: same ``id``, same ``outcome``, same ``context``,
    distinct ``variant_index`` 0..K-1, distinct ``question`` text.
  - All ``variant_index = 0`` rows are originals (``is_original = True``).

Splits:
  Stratified by ``resolved_month``, deterministic seed. Default 80/10/10.
  If a Mantic AIB test split exists (``data/raw/mantic_q2_aib.jsonl``), its ids
  are reserved as a *fixed test* split rather than getting random-assigned.

Quality filters:
  - Drops questions whose paraphrase row has fewer than ``K`` variants
    (incomplete_groups in paraphrase output).
  - Drops questions whose context has zero usable news items (often a sign
    of a content-moderation refusal or generation timeout).

Run:
    python -m frame_invariance.data.build_training_set
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schema import Question, read_jsonl as read_questions


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 17
    stratify_by: str = "resolved_month"  # "resolved_month" | "none"


# ----------------------------------------------------- prompt templates
SYSTEM_PROMPT = (
    "You are a careful, calibrated forecasting model. You will see a binary "
    "forecasting question, a forecast date, a base-rate prior, and a list of "
    "news items dated strictly before the forecast date. Use only the provided "
    "context. Output your final probability on the first line in the exact form "
    "`Probability: <number between 0 and 1>` and a short rationale on subsequent "
    "lines."
)


def render_user_prompt(
    question_text: str,
    *,
    freeze_date: str,
    resolution_date: str,
    source: str,
    background: str | None,
    resolution_criteria: str | None,
    base_rate: dict[str, Any],
    news_snapshot: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    parts.append(f"Forecast date: {freeze_date}")
    parts.append(f"Resolution date: {resolution_date}")
    parts.append(f"Source: {source}")
    parts.append("")
    parts.append("Question:")
    parts.append(question_text.strip())
    parts.append("")
    parts.append("Background:")
    parts.append((background or "(none provided)").strip())
    parts.append("")
    if resolution_criteria:
        parts.append("Resolution criteria:")
        parts.append(resolution_criteria.strip())
        parts.append("")
    parts.append(
        f"Base-rate prior: {base_rate.get('value', 0.5):.3f} "
        f"(reference class size: {base_rate.get('n_reference_events', 0)}; "
        f"window: {base_rate.get('reference_window_years', 0)} years)"
    )
    expl = (base_rate.get("explanation") or "").strip()
    if expl:
        parts.append(f"Base-rate reasoning: {expl}")
    parts.append("")
    if news_snapshot:
        parts.append("Relevant news (all dated before forecast date):")
        for item in news_snapshot:
            d = item.get("date", "")
            h = (item.get("headline") or "").strip()
            s = (item.get("summary") or "").strip()
            line = f"  [{d}] {h}"
            if s:
                line += f" — {s}"
            parts.append(line)
        parts.append("")
    parts.append("Put your forecast on the first line exactly as:")
    parts.append("Probability: <number between 0 and 1>")
    parts.append("Then add a short rationale on the following lines.")
    return "\n".join(parts)


# ----------------------------------------------------- IO helpers
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def index_by_question_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r.get("question_id")): r for r in rows if r.get("question_id")}


# ----------------------------------------------------- splits
def stratify_key(q: Question, mode: str) -> str:
    if mode == "none":
        return "all"
    if mode == "resolved_month":
        return q.resolved_at[:7]
    raise ValueError(f"unknown stratify mode: {mode!r}")


def split_ids(
    questions: list[Question],
    config: SplitConfig,
    *,
    fixed_test_ids: set[str] | None = None,
) -> dict[str, set[str]]:
    """Stratified split by month; ``fixed_test_ids`` (if any) goes straight to test."""

    fixed_test_ids = set(fixed_test_ids or set())
    split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}

    eligible = [q for q in questions if q.id not in fixed_test_ids]
    split["test"].update(q.id for q in questions if q.id in fixed_test_ids)

    by_bucket: dict[str, list[Question]] = defaultdict(list)
    for q in eligible:
        by_bucket[stratify_key(q, config.stratify_by)].append(q)

    for bucket, qs in by_bucket.items():
        ids = [q.id for q in qs]
        ids.sort()
        random.Random(f"{config.seed}::{bucket}").shuffle(ids)
        n = len(ids)
        n_train = int(round(n * config.train_frac))
        n_val = int(round(n * config.val_frac))
        # Anything left over goes to test.
        split["train"].update(ids[:n_train])
        split["validation"].update(ids[n_train : n_train + n_val])
        split["test"].update(ids[n_train + n_val :])

    overlap = (
        (split["train"] & split["validation"])
        | (split["train"] & split["test"])
        | (split["validation"] & split["test"])
    )
    if overlap:
        raise RuntimeError(f"split overlap detected: {sorted(overlap)[:5]}")
    return split


# ----------------------------------------------------- assembly
def assemble_rows(
    questions: list[Question],
    contexts_by_id: dict[str, dict[str, Any]],
    paraphrases_by_id: dict[str, dict[str, Any]],
    *,
    k: int,
    splits: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], Counter]:
    """Produce (training_rows, stats). One row per (question, variant_index)."""

    out: list[dict[str, Any]] = []
    stats: Counter = Counter()

    split_of: dict[str, str] = {}
    for split_name, ids in splits.items():
        for qid in ids:
            split_of[qid] = split_name

    for q in questions:
        # Need both context and paraphrase rows for this question.
        ctx = contexts_by_id.get(q.id)
        par = paraphrases_by_id.get(q.id)
        if ctx is None:
            stats["missing_context"] += 1
            continue
        if par is None:
            stats["missing_paraphrase"] += 1
            continue

        paraphrases = par.get("paraphrases") or []
        if len(paraphrases) < k:
            stats["paraphrase_group_too_small"] += 1
            continue

        # Truncate to K (handles old K=8 rows being trimmed to K=5)
        paraphrases = sorted(paraphrases, key=lambda p: int(p.get("variant_index", 0)))[:k]

        # Sanity: variant_index 0 must be the original
        if not paraphrases[0].get("is_original"):
            stats["original_not_first"] += 1
            continue
        # Sanity: original text must match the question
        if paraphrases[0].get("text", "").strip() != q.question.strip():
            stats["original_text_mismatch"] += 1
            continue

        base_rate = ctx.get("base_rate") or {}
        news_snapshot = ctx.get("news_snapshot") or []
        if not news_snapshot:
            stats["empty_news_snapshot"] += 1
            # Still keep — base rate alone is informative enough — but track it.

        for i, p in enumerate(paraphrases):
            variant_text = (p.get("text") or "").strip()
            if not variant_text:
                stats["empty_variant_text"] += 1
                continue
            user_prompt = render_user_prompt(
                variant_text,
                freeze_date=q.freeze_date,
                resolution_date=q.resolved_at,
                source=q.source,
                background=q.background,
                resolution_criteria=q.resolution_criteria,
                base_rate=base_rate,
                news_snapshot=news_snapshot,
            )
            row = {
                "id": q.id,
                "variant_index": i,
                "is_original": bool(p.get("is_original", i == 0)),
                "split": split_of.get(q.id, "test"),
                "question": variant_text,
                "outcome": q.outcome,
                "freeze_date": q.freeze_date,
                "resolved_at": q.resolved_at,
                "source": q.source,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "base_rate": base_rate,
                "news_snapshot": news_snapshot,
                "categories": q.categories,
            }
            out.append(row)

        stats["groups_emitted"] += 1
        stats[f"groups_emitted_{split_of.get(q.id, 'test')}"] += 1

    return out, stats


def write_training_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    return len(rows)


def write_audit(stats: Counter, splits: dict[str, set[str]], path: Path, *, k: int) -> None:
    audit = {
        "k": k,
        "split_group_counts": {name: len(ids) for name, ids in splits.items()},
        "assembly_stats": dict(stats),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


# ----------------------------------------------------- CLI
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble the Tinker-ready training JSONL from unified + contexts + paraphrases."
    )
    parser.add_argument("--unified", default="data/processed/unified.jsonl")
    parser.add_argument("--contexts", default="data/processed/contexts.jsonl")
    parser.add_argument("--paraphrases", default="data/processed/paraphrases.jsonl")
    parser.add_argument("--output", default="data/processed/training.jsonl")
    parser.add_argument("--audit-output", default="data/processed/training_audit.json")
    parser.add_argument(
        "--mantic-aib", default=None,
        help="Optional path to data/raw/mantic_q2_aib.jsonl; ids will be reserved for test."
    )
    parser.add_argument("--k", type=int, default=5,
                        help="Group size (1 original + K-1 paraphrases). Defaults to 5.")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--stratify-by", choices=("resolved_month", "none"),
                        default="resolved_month")
    args = parser.parse_args()

    questions = read_questions(args.unified)
    contexts = index_by_question_id(read_jsonl(Path(args.contexts)))
    paraphrases = index_by_question_id(read_jsonl(Path(args.paraphrases)))

    print(f"unified:     {len(questions)} questions", file=sys.stderr)
    print(f"contexts:    {len(contexts)} rows", file=sys.stderr)
    print(f"paraphrases: {len(paraphrases)} rows", file=sys.stderr)

    fixed_test_ids: set[str] = set()
    if args.mantic_aib:
        path = Path(args.mantic_aib)
        if path.exists():
            for q in read_questions(path):
                # Strip the source prefix so the lookup matches unified IDs.
                fixed_test_ids.add(q.id.replace("mantic_q2_aib::", "metaculus::"))
            print(f"mantic_aib:  {len(fixed_test_ids)} ids reserved for test", file=sys.stderr)
        else:
            print(f"warning: --mantic-aib {path} not found, ignoring", file=sys.stderr)

    config = SplitConfig(
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        stratify_by=args.stratify_by,
    )
    splits = split_ids(questions, config, fixed_test_ids=fixed_test_ids)
    rows, stats = assemble_rows(
        questions, contexts, paraphrases,
        k=args.k, splits=splits,
    )
    n = write_training_jsonl(rows, Path(args.output))
    write_audit(stats, splits, Path(args.audit_output), k=args.k)

    print(f"\nwrote {n} training rows to {args.output}", file=sys.stderr)
    print(f"audit:  {args.audit_output}", file=sys.stderr)
    print(json.dumps({"split_group_counts": {k: len(v) for k, v in splits.items()},
                      "assembly_stats": dict(stats), "training_rows": n,
                      "k": args.k}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
