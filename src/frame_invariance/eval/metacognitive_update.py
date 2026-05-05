"""Metacognitive update-prediction smoke test.

This evaluator asks two closely related questions for each forecast:

1. Hypothetical: given the prior phase and a stated current forecast pi, what
   posterior would the model expect to have if it received neutral evidence Y?
2. Actual: in a fresh chat with the same prior phase, actually provide Y and
   ask for the posterior.

The resulting metric compares predicted vs actual log-odds shifts from pi. A
good forecaster should not merely say "I would update" abstractly; its
hypothetical update should match its actual update when the evidence is shown.
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
    prior_assistant_message,
    render_prior_prompt,
    render_update_prompt,
)
from frame_invariance.eval.baseline import (
    OpenAIChatPredictor,
    TinkerSamplingPredictor,
    normalize_messages,
)
from frame_invariance.eval.metrics import parse_probability


DEFAULT_TRAINING = "data/processed/training.jsonl"
DEFAULT_UNIFIED = "data/processed/unified.jsonl"
DEFAULT_REWRITES = "data/processed/update_rewrites_strong.jsonl"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TINKER_BASE_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class Config:
    training: Path
    unified: Path
    rewrites: Path
    split: str
    run_name: str
    results_dir: Path
    mode: str
    model: str
    limit_groups: int | None
    max_workers: int
    max_tokens: int
    temperature: float
    top_p: float
    cache_dir: Path | None
    no_cache: bool
    api_key_env: str
    base_url: str | None
    tinker_api_key_env: str
    tinker_base_model: str


def latest_valid_neutral_rewrites(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("valid") is not True:
            continue
        qid = str(row.get("question_id", ""))
        neutral = None
        for rewrite in row.get("rewrites") or []:
            if int(rewrite.get("update_variant_index", -1)) == 0:
                neutral = rewrite
                break
        if qid and neutral and str(neutral.get("update_text") or "").strip():
            out[qid] = {
                "source_news_hash": row.get("source_news_hash"),
                "update_text": str(neutral["update_text"]).strip(),
            }
    return out


def selected_original_rows(rows: list[dict[str, Any]], *, split: str, limit_groups: int | None) -> list[dict[str, Any]]:
    selected = [
        r
        for r in rows
        if r.get("split") == split and int(r.get("variant_index", 0)) == 0
    ]
    selected.sort(key=lambda r: str(r.get("id")))
    if limit_groups is not None:
        selected = selected[:limit_groups]
    return selected


def hypothetical_messages(prior_prompt: str, prior_prob: float, update_text: str) -> list[dict[str, str]]:
    return normalize_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prior_prompt},
            {"role": "assistant", "content": f"Prior probability: {prior_prob:.3f}"},
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Metacognitive update prediction",
                        "",
                        f"Assume your current forecast is P(X) = {prior_prob:.3f}.",
                        "You are about to receive the following new information.",
                        "",
                        "New information:",
                        update_text.strip(),
                        "",
                        "Before making the update in an ordinary forecast, predict what your",
                        "own posterior forecast would be after incorporating this information.",
                        "Put that predicted posterior on the first line exactly as:",
                        "Probability: <number between 0 and 1>",
                        "Then add a short rationale.",
                    ]
                ),
            },
        ]
    )


def actual_messages(prior_prompt: str, base_rate: dict[str, Any], update_text: str) -> list[dict[str, str]]:
    return normalize_messages(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prior_prompt},
            {"role": "assistant", "content": prior_assistant_message(base_rate)},
            {"role": "user", "content": render_update_prompt(update_text)},
        ]
    )


def logit(p: float, *, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def make_predictor(config: Config) -> OpenAIChatPredictor | TinkerSamplingPredictor:
    if config.mode == "api":
        return OpenAIChatPredictor(
            model=config.model,
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
            model=config.model,
            base_model=config.tinker_base_model,
            api_key_env=config.tinker_api_key_env,
            cache_dir=config.cache_dir,
            use_cache=not config.no_cache,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    raise ValueError(f"unknown mode {config.mode!r}")


def evaluate(config: Config) -> tuple[dict[str, Any], Path, Path]:
    started = time.time()
    questions_by_id = index_questions(config.unified)
    rewrites_by_id = latest_valid_neutral_rewrites(config.rewrites)
    training_rows = selected_original_rows(
        read_jsonl(config.training), split=config.split, limit_groups=config.limit_groups
    )
    predictor = make_predictor(config)

    run_dir = config.results_dir / config.run_name / config.split
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "metacognitive_update_rows.csv"
    summary_path = run_dir / "summary.json"

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(training_rows, start=1):
        qid = str(row["id"])
        q = questions_by_id.get(qid)
        rewrite = rewrites_by_id.get(qid)
        base_rate = row.get("base_rate") or {}
        try:
            prior_prob = float(base_rate.get("value"))
        except (TypeError, ValueError):
            prior_prob = math.nan
        if q is None or rewrite is None or not math.isfinite(prior_prob):
            continue
        update_text = str(rewrite["update_text"])
        prior_prompt = render_prior_prompt(
            str(row.get("question") or q.question),
            freeze_date=str(row.get("freeze_date") or q.freeze_date),
            resolution_date=str(row.get("resolved_at") or q.resolved_at),
            source=str(row.get("source") or q.source),
            background=q.background,
            resolution_criteria=q.resolution_criteria,
            base_rate=base_rate,
        )

        hyp_completion = predictor.complete(
            hypothetical_messages(prior_prompt, prior_prob, update_text)
        )
        act_completion = predictor.complete(
            actual_messages(prior_prompt, base_rate, update_text)
        )
        hyp_prob = parse_probability(hyp_completion)
        act_prob = parse_probability(act_completion)
        row_out: dict[str, Any] = {
            "id": qid,
            "split": config.split,
            "outcome": int(row["outcome"]),
            "prior_prob": prior_prob,
            "hypothetical_prob": hyp_prob,
            "actual_prob": act_prob,
            "hypothetical_parseable": hyp_prob is not None,
            "actual_parseable": act_prob is not None,
            "question": row.get("question", ""),
            "hypothetical_completion": hyp_completion,
            "actual_completion": act_completion,
        }
        if hyp_prob is not None and act_prob is not None:
            hyp_shift = logit(float(hyp_prob)) - logit(prior_prob)
            act_shift = logit(float(act_prob)) - logit(prior_prob)
            row_out.update(
                {
                    "hypothetical_logodds_shift": hyp_shift,
                    "actual_logodds_shift": act_shift,
                    "logodds_shift_error": hyp_shift - act_shift,
                    "abs_logodds_shift_error": abs(hyp_shift - act_shift),
                    "abs_posterior_gap": abs(float(hyp_prob) - float(act_prob)),
                    "actual_abs_logodds_shift": abs(act_shift),
                    "hypothetical_abs_logodds_shift": abs(hyp_shift),
                }
            )
        out_rows.append(row_out)
        print(f"metacognitive update {idx}/{len(training_rows)}", flush=True)

    write_csv(rows_path, out_rows)
    parseable = [
        r for r in out_rows if r.get("hypothetical_parseable") and r.get("actual_parseable")
    ]
    summary = {
        "run_name": config.run_name,
        "split": config.split,
        "mode": config.mode,
        "model": config.model,
        "n_groups": len(out_rows),
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
        "elapsed_s": round(time.time() - started, 3),
        "limit_groups": config.limit_groups,
        "neutral_only": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary, rows_path, summary_path


def mean(values: Any) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def frac(values: Any) -> float | None:
    vals = [bool(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "split",
        "outcome",
        "prior_prob",
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
        "hypothetical_completion",
        "actual_completion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description="Run metacognitive update-prediction smoke.")
    parser.add_argument("--training", default=DEFAULT_TRAINING)
    parser.add_argument("--unified", default=DEFAULT_UNIFIED)
    parser.add_argument("--rewrites", default=DEFAULT_REWRITES)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--mode", choices=("api", "tinker"), default="tinker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit-groups", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cache-dir", default="data/cache/metacognitive_update")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")
    parser.add_argument("--tinker-base-model", default=DEFAULT_TINKER_BASE_MODEL)
    args = parser.parse_args(argv)
    return Config(
        training=Path(args.training),
        unified=Path(args.unified),
        rewrites=Path(args.rewrites),
        split=args.split,
        run_name=args.run_name,
        results_dir=Path(args.results_dir),
        mode=args.mode,
        model=args.model,
        limit_groups=args.limit_groups,
        max_workers=max(1, args.max_workers),
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
