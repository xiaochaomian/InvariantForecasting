"""Cross-model metacognitive update parity test.

This script asks an observer model to predict a target model's posterior update
on the exact same evidence used by the self-prediction evaluator. It reuses a
completed target self-run CSV for the target prior and actual posterior, so the
cross-model comparison does not need to resample the target model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frame_invariance.data.build_training_set import read_jsonl
from frame_invariance.data.build_training_v2_5 import (
    SYSTEM_PROMPT,
    index_questions,
    render_prior_prompt,
)
from frame_invariance.eval.baseline import (
    OpenAIChatPredictor,
    TinkerSamplingPredictor,
    normalize_messages,
)
from frame_invariance.eval.metacognitive_update import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_REWRITES,
    DEFAULT_TRAINING,
    DEFAULT_UNIFIED,
    append_checkpoint_row,
    frac,
    latest_valid_neutral_rewrites,
    logit,
    mean,
    read_checkpoint_rows,
    selected_original_rows,
    summarize_prediction_quality,
)
from frame_invariance.eval.metrics import parse_probability


DEFAULT_OBSERVER_MODEL = "openai/gpt-oss-20b"
DEFAULT_TINKER_BASE_MODEL = "openai/gpt-oss-20b"


@dataclass(frozen=True)
class CrossConfig:
    training: Path
    unified: Path
    rewrites: Path
    target_rows: Path
    split: str
    run_name: str
    results_dir: Path
    mode: str
    observer_model: str
    observer_label: str
    target_label: str
    limit_groups: int | None
    max_tokens: int
    temperature: float
    top_p: float
    cache_dir: Path | None
    no_cache: bool
    api_key_env: str
    base_url: str | None
    tinker_api_key_env: str
    tinker_base_model: str


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_target_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {str(row["id"]): row for row in csv.DictReader(handle)}


def finite_probability(value: Any) -> float | None:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(prob) and 0.0 <= prob <= 1.0:
        return prob
    return None


def target_has_parseable_actual(row: dict[str, Any]) -> bool:
    return (
        truthy(row.get("prior_parseable"))
        and truthy(row.get("actual_parseable"))
        and finite_probability(row.get("prior_prob")) is not None
        and finite_probability(row.get("actual_prob")) is not None
    )


def observer_messages(
    *,
    prior_prompt: str,
    target_label: str,
    observer_label: str,
    target_prior_prob: float,
    target_prior_completion: str,
    update_text: str,
) -> list[dict[str, str]]:
    target_prior_block = target_prior_completion.strip()
    if target_prior_block:
        target_prior_block = "\n".join(
            [
                f"In a separate prior-only context, {target_label} answered:",
                target_prior_block,
                "",
            ]
        )

    return normalize_messages(
        [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        SYSTEM_PROMPT,
                        "",
                        "You are now doing behavioral forecasting about another model.",
                        "Predict that model's posterior, not your own posterior.",
                    ]
                ),
            },
            {"role": "user", "content": prior_prompt},
            {
                "role": "assistant",
                "content": (
                    "Understood. I will treat the prior phase as shared context "
                    "for predicting the target model."
                ),
            },
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Cross-model posterior prediction",
                        "",
                        target_prior_block.rstrip(),
                        f"Target model: {target_label}",
                        f"Observer model: {observer_label}",
                        f"{target_label}'s current forecast is P(X) = "
                        f"{target_prior_prob:.6f}.",
                        "",
                        "The target model will receive this new information in a fresh",
                        "ordinary forecast context:",
                        "",
                        "New information:",
                        update_text.strip(),
                        "",
                        f"Predict what {target_label} would output as its posterior "
                        "probability after incorporating this information.",
                        "Do not give your own posterior unless it matches your prediction",
                        f"of {target_label}'s posterior.",
                        "Put the predicted target posterior on the first line exactly as:",
                        "Probability: <number between 0 and 1>",
                        "Then add a short rationale.",
                    ]
                ),
            },
        ]
    )


def make_observer_predictor(config: CrossConfig) -> OpenAIChatPredictor | TinkerSamplingPredictor:
    if config.mode == "api":
        return OpenAIChatPredictor(
            model=config.observer_model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            cache_dir=config.cache_dir,
            use_cache=not config.no_cache,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    if config.mode == "tinker":
        return TinkerSamplingPredictor(
            model=config.observer_model,
            base_model=config.tinker_base_model,
            api_key_env=config.tinker_api_key_env,
            cache_dir=config.cache_dir,
            use_cache=not config.no_cache,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    raise ValueError(f"unknown mode {config.mode!r}")


def evaluate(config: CrossConfig) -> tuple[dict[str, Any], Path, Path]:
    started = time.time()
    questions_by_id = index_questions(config.unified)
    rewrites_by_id = latest_valid_neutral_rewrites(config.rewrites)
    target_rows_by_id = read_target_rows(config.target_rows)
    training_rows = selected_original_rows(
        read_jsonl(config.training), split=config.split, limit_groups=config.limit_groups
    )
    predictor = make_observer_predictor(config)

    run_dir = config.results_dir / config.run_name / config.split
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "cross_model_update_rows.csv"
    checkpoint_path = run_dir / "cross_model_update_rows.jsonl"
    summary_path = run_dir / "summary.json"

    out_rows = read_checkpoint_rows(checkpoint_path)
    completed_ids = {str(row.get("id")) for row in out_rows}
    if completed_ids:
        print(
            f"resuming from {checkpoint_path}: {len(completed_ids)} completed groups",
            flush=True,
        )

    candidates = [
        row
        for row in training_rows
        if target_has_parseable_actual(target_rows_by_id.get(str(row.get("id")), {}))
    ]

    for idx, row in enumerate(candidates, start=1):
        qid = str(row["id"])
        if qid in completed_ids:
            print(f"cross-model update {idx}/{len(candidates)} cached", flush=True)
            continue

        q = questions_by_id.get(qid)
        rewrite = rewrites_by_id.get(qid)
        target_row = target_rows_by_id[qid]
        prior_prob = finite_probability(target_row.get("prior_prob"))
        actual_prob = finite_probability(target_row.get("actual_prob"))
        if q is None or rewrite is None or prior_prob is None or actual_prob is None:
            continue

        update_text = str(rewrite["update_text"])
        prior_prompt = render_prior_prompt(
            str(row.get("question") or q.question),
            freeze_date=str(row.get("freeze_date") or q.freeze_date),
            resolution_date=str(row.get("resolved_at") or q.resolved_at),
            source=str(row.get("source") or q.source),
            background=q.background,
            resolution_criteria=q.resolution_criteria,
            base_rate=row.get("base_rate") or {},
        )
        completion = predictor.complete(
            observer_messages(
                prior_prompt=prior_prompt,
                target_label=config.target_label,
                observer_label=config.observer_label,
                target_prior_prob=prior_prob,
                target_prior_completion=str(target_row.get("prior_completion") or ""),
                update_text=update_text,
            )
        )
        predicted_prob = parse_probability(completion)

        row_out: dict[str, Any] = {
            "id": qid,
            "split": config.split,
            "outcome": int(row["outcome"]),
            "observer_model": config.observer_model,
            "observer_label": config.observer_label,
            "target_label": config.target_label,
            "prior_source": "target_elicited",
            "stated_prior_prob": row.get("base_rate", {}).get("value"),
            "prior_prob": prior_prob,
            "prior_parseable": True,
            "hypothetical_prob": predicted_prob,
            "actual_prob": actual_prob,
            "hypothetical_parseable": predicted_prob is not None,
            "actual_parseable": True,
            "question": row.get("question", ""),
            "prior_completion": str(target_row.get("prior_completion") or ""),
            "hypothetical_completion": completion,
            "actual_completion": str(target_row.get("actual_completion") or ""),
        }
        if predicted_prob is not None:
            hyp_shift = logit(float(predicted_prob)) - logit(prior_prob)
            act_shift = logit(actual_prob) - logit(prior_prob)
            row_out.update(
                {
                    "hypothetical_logodds_shift": hyp_shift,
                    "actual_logodds_shift": act_shift,
                    "logodds_shift_error": hyp_shift - act_shift,
                    "abs_logodds_shift_error": abs(hyp_shift - act_shift),
                    "abs_posterior_gap": abs(float(predicted_prob) - actual_prob),
                    "actual_abs_logodds_shift": abs(act_shift),
                    "hypothetical_abs_logodds_shift": abs(hyp_shift),
                }
            )

        out_rows.append(row_out)
        completed_ids.add(qid)
        append_checkpoint_row(checkpoint_path, row_out)
        print(f"cross-model update {idx}/{len(candidates)}", flush=True)

    out_rows.sort(key=lambda r: str(r.get("id")))
    write_csv(rows_path, out_rows)
    parseable = [
        r
        for r in out_rows
        if r.get("prior_parseable")
        and r.get("hypothetical_parseable")
        and r.get("actual_parseable")
    ]
    metric_bundle = summarize_prediction_quality(parseable)
    summary = {
        "run_name": config.run_name,
        "split": config.split,
        "mode": config.mode,
        "observer_model": config.observer_model,
        "observer_label": config.observer_label,
        "target_label": config.target_label,
        "target_rows": str(config.target_rows),
        "n_groups": len(out_rows),
        "checkpoint_path": str(checkpoint_path),
        "n_prior_parseable": sum(1 for r in out_rows if r.get("prior_parseable")),
        "n_parseable_pairs": len(parseable),
        "pair_coverage": len(parseable) / len(out_rows) if out_rows else 0.0,
        "mean_abs_logodds_shift_error": mean(
            r.get("abs_logodds_shift_error") for r in parseable
        ),
        "mean_abs_posterior_gap": mean(r.get("abs_posterior_gap") for r in parseable),
        "mean_actual_abs_logodds_shift": mean(
            r.get("actual_abs_logodds_shift") for r in parseable
        ),
        "mean_hypothetical_abs_logodds_shift": mean(
            r.get("hypothetical_abs_logodds_shift") for r in parseable
        ),
        "actual_shift_gt_0_05_frac": frac(
            r.get("actual_abs_logodds_shift", 0.0) > 0.05 for r in parseable
        ),
        "hypothetical_shift_gt_0_05_frac": frac(
            r.get("hypothetical_abs_logodds_shift", 0.0) > 0.05 for r in parseable
        ),
        **metric_bundle,
        "elapsed_s": round(time.time() - started, 3),
        "limit_groups": config.limit_groups,
        "neutral_only": True,
        "protocol": "observer_predicts_target_posterior",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary, rows_path, summary_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "split",
        "outcome",
        "observer_model",
        "observer_label",
        "target_label",
        "prior_source",
        "stated_prior_prob",
        "prior_prob",
        "prior_parseable",
        "hypothetical_prob",
        "actual_prob",
        "hypothetical_parseable",
        "actual_parseable",
        "hypothetical_logodds_shift",
        "actual_logodds_shift",
        "logodds_shift_error",
        "abs_logodds_shift_error",
        "abs_posterior_gap",
        "actual_abs_logodds_shift",
        "hypothetical_abs_logodds_shift",
        "question",
        "prior_completion",
        "hypothetical_completion",
        "actual_completion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> CrossConfig:
    parser = argparse.ArgumentParser(description="Run cross-model posterior parity test.")
    parser.add_argument("--training", default=DEFAULT_TRAINING)
    parser.add_argument("--unified", default=DEFAULT_UNIFIED)
    parser.add_argument("--rewrites", default=DEFAULT_REWRITES)
    parser.add_argument("--target-rows", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--mode", choices=("api", "tinker"), default="tinker")
    parser.add_argument("--observer-model", default=DEFAULT_OBSERVER_MODEL)
    parser.add_argument("--observer-label", default="gpt-oss-20B")
    parser.add_argument("--target-label", default="gpt-oss-120B")
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=500,
        help="Debug cap on original question groups; use 0 for the full split.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="data/cache/cross_model_update_parity")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--tinker-base-model", default=DEFAULT_TINKER_BASE_MODEL)
    args = parser.parse_args(argv)
    return CrossConfig(
        training=Path(args.training),
        unified=Path(args.unified),
        rewrites=Path(args.rewrites),
        target_rows=Path(args.target_rows),
        split=args.split,
        run_name=args.run_name,
        results_dir=Path(args.results_dir),
        mode=args.mode,
        observer_model=args.observer_model,
        observer_label=args.observer_label,
        target_label=args.target_label,
        limit_groups=None if args.limit_groups == 0 else args.limit_groups,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cache_dir=None if args.cache_dir in {"", "none", "None"} else Path(args.cache_dir),
        no_cache=args.no_cache,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        tinker_api_key_env=args.tinker_api_key_env,
        tinker_base_model=args.tinker_base_model,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    summary, rows_path, summary_path = evaluate(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"rows:    {rows_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
