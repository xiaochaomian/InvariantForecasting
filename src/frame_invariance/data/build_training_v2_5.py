"""Assemble prior/update/posterior V2.5 training rows.

V1 grouped by question paraphrases. V2.5 keeps those question paraphrases, but
turns each one into its own update-invariance group:

  group id = original question id + question paraphrase index
  variants = five strong evidence-update framings

That means Tinker still sees group size 5, but the invariance pressure is now
about rhetorical framing of the update packet rather than only surface wording
of the question.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_training_set import read_jsonl, write_training_jsonl
from .schema import Question, read_jsonl as read_questions


DEFAULT_TRAINING = "data/processed/training.jsonl"
DEFAULT_REWRITES = "data/processed/update_rewrites_strong.jsonl"
DEFAULT_OUTPUT = "data/processed/training_v2_5.jsonl"
DEFAULT_AUDIT = "data/processed/training_v2_5_audit.json"


SYSTEM_PROMPT = (
    "You are a careful, calibrated forecasting model. You will see a binary "
    "forecasting question in two phases. First, you receive the prior phase: "
    "the question, background, resolution criteria, and base-rate prior. Then "
    "you receive an update phase containing dated evidence from before the "
    "forecast date. Treat the prior as your starting point and update only "
    "from the provided evidence. Output your final posterior probability on "
    "the first line exactly as `Probability: <number between 0 and 1>` and "
    "then add a short rationale."
)


def render_prior_prompt(
    question_text: str,
    *,
    freeze_date: str,
    resolution_date: str,
    source: str,
    background: str | None,
    resolution_criteria: str | None,
    base_rate: dict[str, Any],
) -> str:
    parts: list[str] = []
    parts.append("Prior phase")
    parts.append("")
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
        f"Base-rate prior: {float(base_rate.get('value', 0.5)):.3f} "
        f"(reference class size: {base_rate.get('n_reference_events', 0)}; "
        f"window: {base_rate.get('reference_window_years', 0)} years)"
    )
    explanation = str(base_rate.get("explanation") or "").strip()
    if explanation:
        parts.append(f"Base-rate reasoning: {explanation}")
    parts.append("")
    parts.append(
        "Do not answer yet. Store this as the prior; the next message will "
        "provide new dated evidence for the update phase."
    )
    return "\n".join(parts)


def render_update_prompt(update_text: str) -> str:
    return "\n".join(
        [
            "Update phase",
            "",
            "New information, dated before the forecast date:",
            update_text.strip(),
            "",
            "Now update from the prior phase using only this new information.",
            "Put your final posterior forecast on the first line exactly as:",
            "Probability: <number between 0 and 1>",
            "Then add a short rationale on the following lines.",
        ]
    )


def prior_assistant_message(base_rate: dict[str, Any]) -> str:
    return f"Prior probability: {float(base_rate.get('value', 0.5)):.3f}"


def index_questions(path: Path) -> dict[str, Question]:
    return {q.id: q for q in read_questions(path)}


def index_rewrites(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {
        str(row.get("question_id")): row
        for row in rows
        if row.get("question_id") and row.get("valid") is not False
    }


def assemble_v2_rows(
    *,
    training_rows: list[dict[str, Any]],
    questions_by_id: dict[str, Question],
    rewrites_by_id: dict[str, dict[str, Any]],
    k_updates: int,
    question_variants: str = "all",
) -> tuple[list[dict[str, Any]], Counter]:
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()

    for row in training_rows:
        question_variant_index = int(row.get("variant_index", 0))
        if question_variants == "original-only" and question_variant_index != 0:
            stats["skipped_non_original_question_variant"] += 1
            continue
        original_id = str(row["id"])
        q = questions_by_id.get(original_id)
        rewrite_row = rewrites_by_id.get(original_id)
        if q is None:
            stats["missing_question"] += 1
            continue
        if rewrite_row is None:
            stats["missing_rewrites"] += 1
            continue
        rewrites = list(rewrite_row.get("rewrites") or [])
        if len(rewrites) < k_updates:
            stats["rewrite_group_too_small"] += 1
            continue

        group_id = f"{original_id}::qv{question_variant_index}"
        base_rate = row.get("base_rate") or {}
        prior_prompt = render_prior_prompt(
            str(row.get("question") or q.question),
            freeze_date=str(row.get("freeze_date") or q.freeze_date),
            resolution_date=str(row.get("resolved_at") or q.resolved_at),
            source=str(row.get("source") or q.source),
            background=q.background,
            resolution_criteria=q.resolution_criteria,
            base_rate=base_rate,
        )
        prior_message = prior_assistant_message(base_rate)

        for i, rewrite in enumerate(rewrites[:k_updates]):
            update_text = str(rewrite.get("update_text") or "").strip()
            if not update_text:
                stats["empty_update_text"] += 1
                continue
            out.append(
                {
                    "id": group_id,
                    "original_id": original_id,
                    "question_variant_index": question_variant_index,
                    "variant_index": i,
                    "update_variant_index": i,
                    "update_style": rewrite.get("style"),
                    "update_stance": rewrite.get("stance"),
                    "update_strength": rewrite.get("strength"),
                    "update_source_news_hash": rewrite_row.get("source_news_hash"),
                    "is_original": i == 0,
                    "split": row.get("split"),
                    "question": row.get("question"),
                    "outcome": row.get("outcome"),
                    "freeze_date": row.get("freeze_date"),
                    "resolved_at": row.get("resolved_at"),
                    "source": row.get("source"),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prior_prompt},
                        {"role": "assistant", "content": prior_message},
                        {"role": "user", "content": render_update_prompt(update_text)},
                    ],
                    "base_rate": base_rate,
                    "news_snapshot": row.get("news_snapshot") or [],
                    "update_text": update_text,
                    "categories": row.get("categories") or [],
                    "v2_5_format": "prior_update_posterior",
                }
            )
        stats["groups_emitted"] += 1
        stats[f"groups_emitted_{row.get('split')}"] += 1

    return out, stats


def audit_rows(rows: list[dict[str, Any]], stats: Counter, *, k_updates: int) -> dict[str, Any]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["id"])].append(row)

    bad_groups: list[dict[str, Any]] = []
    split_groups: Counter = Counter()
    for group_id, group in by_id.items():
        split_groups[str(group[0].get("split"))] += 1
        idxs = sorted(int(r["variant_index"]) for r in group)
        outcomes = {int(r["outcome"]) for r in group}
        priors = {json.dumps(r.get("base_rate") or {}, sort_keys=True) for r in group}
        if idxs != list(range(k_updates)) or len(outcomes) != 1 or len(priors) != 1:
            bad_groups.append(
                {
                    "id": group_id,
                    "variant_indices": idxs,
                    "outcomes": sorted(outcomes),
                    "n_distinct_priors": len(priors),
                }
            )

    return {
        "k_updates": k_updates,
        "training_rows": len(rows),
        "group_count": len(by_id),
        "split_group_counts": dict(split_groups),
        "assembly_stats": dict(stats),
        "bad_group_count": len(bad_groups),
        "bad_group_samples": bad_groups[:20],
    }


def write_audit(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build prior/update/posterior V2.5 training JSONL.")
    parser.add_argument("--training", default=DEFAULT_TRAINING)
    parser.add_argument("--unified", default="data/processed/unified.jsonl")
    parser.add_argument("--rewrites", default=DEFAULT_REWRITES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", default=DEFAULT_AUDIT)
    parser.add_argument("--k-updates", type=int, default=5)
    parser.add_argument(
        "--question-variants",
        choices=("all", "original-only"),
        default="all",
        help="Use all V1 question paraphrases, or only variant_index=0 for a cheaper V2.5 dataset.",
    )
    args = parser.parse_args(argv)

    training_rows = read_jsonl(Path(args.training))
    questions_by_id = index_questions(Path(args.unified))
    rewrites_by_id = index_rewrites(Path(args.rewrites))

    print(f"training rows: {len(training_rows)}", file=sys.stderr)
    print(f"questions:     {len(questions_by_id)}", file=sys.stderr)
    print(f"rewrite rows:  {len(rewrites_by_id)}", file=sys.stderr)

    rows, stats = assemble_v2_rows(
        training_rows=training_rows,
        questions_by_id=questions_by_id,
        rewrites_by_id=rewrites_by_id,
        k_updates=args.k_updates,
        question_variants=args.question_variants,
    )
    audit = audit_rows(rows, stats, k_updates=args.k_updates)
    if audit["bad_group_count"]:
        raise RuntimeError(f"bad V2.5 groups found: {audit['bad_group_samples'][:3]}")

    n = write_training_jsonl(rows, Path(args.output))
    write_audit(Path(args.audit_output), audit)
    print(f"wrote {n} V2.5 rows to {args.output}", file=sys.stderr)
    print(f"audit: {args.audit_output}", file=sys.stderr)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
