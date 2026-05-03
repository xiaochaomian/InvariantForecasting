"""Tests for baseline evaluation. No network calls."""

from __future__ import annotations

import json
from pathlib import Path

from frame_invariance.eval.baseline import (
    EvalConfig,
    evaluate,
    filter_rows,
    train_base_rate,
)
from frame_invariance.eval.metrics import compute_metrics, parse_probability


def _row(qid: str, variant: int, split: str, outcome: int, base_rate: float = 0.25):
    return {
        "id": f"forecastbench::{qid}",
        "variant_index": variant,
        "is_original": variant == 0,
        "split": split,
        "question": f"Will {qid} happen variant {variant}?",
        "outcome": outcome,
        "messages": [
            {"role": "system", "content": "You are calibrated."},
            {"role": "user", "content": "Probability: <number>"},
        ],
        "base_rate": {
            "value": base_rate,
            "n_reference_events": 10,
            "reference_window_years": 5,
            "explanation": "test",
        },
        "news_snapshot": [],
    }


def _write_training(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_parse_probability_requires_probability_line_by_default():
    assert parse_probability("Probability: 0.37") == 0.37
    assert parse_probability("Probability: 37%") == 0.37
    assert parse_probability("Final probability: .4") == 0.4
    assert parse_probability("analysis...assistantfinalProbability: 0.05") == 0.05
    assert parse_probability("I think 0.4") is None
    assert parse_probability("I think 0.4", allow_loose=True) == 0.4
    assert parse_probability("Probability: 130%") is None
    assert parse_probability("Probability: -0.1") is None


def test_compute_metrics_grouped_parasd_and_frie():
    rows = [
        {"id": "a", "variant_index": 0, "split": "test", "outcome": 1,
         "prediction": 0.2, "parseable": True},
        {"id": "a", "variant_index": 1, "split": "test", "outcome": 1,
         "prediction": 0.4, "parseable": True},
        {"id": "b", "variant_index": 0, "split": "test", "outcome": 0,
         "prediction": 0.1, "parseable": True},
        {"id": "b", "variant_index": 1, "split": "test", "outcome": 0,
         "prediction": 0.1, "parseable": True},
    ]
    bundle = compute_metrics(rows)
    summary = bundle.summary
    # Brier = (0.8^2 + 0.6^2 + 0.1^2 + 0.1^2) / 4 = 0.255
    assert abs(summary["mean_brier"] - 0.255) < 1e-9
    # Group a pop-std is 0.1, group b pop-std is 0.0 => mean ParaSD 0.05
    assert abs(summary["mean_parasd"] - 0.05) < 1e-9
    assert abs(summary["frie_lambda_1"] - 0.305) < 1e-9
    assert summary["variant_coverage"] == 1.0
    assert summary["group_full_coverage"] == 1.0


def test_compute_metrics_all_unparseable_does_not_crash():
    rows = [
        {"id": "a", "variant_index": 0, "split": "test", "outcome": 1,
         "prediction": None, "parseable": False},
        {"id": "a", "variant_index": 1, "split": "test", "outcome": 1,
         "prediction": None, "parseable": False},
    ]
    summary = compute_metrics(rows).summary
    assert summary["variant_coverage"] == 0.0
    assert summary["n_parseable_groups"] == 0
    assert summary["mean_brier"] is None
    assert summary["mean_parasd"] is None
    assert summary["frie_lambda_1"] is None
    assert summary["frie_lambda_5"] is None


def test_filter_rows_keeps_complete_limited_groups():
    rows = []
    for i in range(3):
        for v in range(5):
            rows.append(_row(f"q{i}", v, "validation", outcome=i % 2))
    limited = filter_rows(rows, split="validation", limit_groups=2)
    assert len(limited) == 10
    assert {r["id"] for r in limited} == {"forecastbench::q0", "forecastbench::q1"}


def test_train_base_rate_uses_groups_not_variants():
    rows = []
    for v in range(5):
        rows.append(_row("yes", v, "train", outcome=1))
        rows.append(_row("no", v, "train", outcome=0))
    assert train_base_rate(rows) == 0.5


def test_constant_baseline_writes_artifacts(tmp_path: Path):
    training = tmp_path / "training.jsonl"
    rows = []
    for v in range(5):
        rows.append(_row("train_yes", v, "train", outcome=1))
        rows.append(_row("val_no", v, "validation", outcome=0))
        rows.append(_row("val_yes", v, "validation", outcome=1))
    _write_training(training, rows)

    config = EvalConfig(
        input_path=training,
        split="validation",
        run_name="constant_smoke",
        results_dir=tmp_path / "results",
        mode="constant",
        model="none",
        constant_prob=0.5,
        limit_groups=None,
        max_workers=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        allow_loose_parse=False,
        cache_dir=None,
        no_cache=True,
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        tinker_api_key_env="TINKER_API_KEY",
        tinker_base_model="openai/gpt-oss-120b",
    )
    summary, variant_path, group_path, summary_path = evaluate(config)
    assert summary["n_groups"] == 2
    assert summary["n_variant_predictions"] == 10
    assert summary["variant_coverage"] == 1.0
    assert summary["mean_brier"] == 0.25
    assert variant_path.exists()
    assert group_path.exists()
    assert summary_path.exists()


def test_context_base_rate_baseline(tmp_path: Path):
    training = tmp_path / "training.jsonl"
    rows = [
        _row("val_yes", v, "validation", outcome=1, base_rate=0.8)
        for v in range(5)
    ]
    _write_training(training, rows)

    config = EvalConfig(
        input_path=training,
        split="validation",
        run_name="context_base_rate",
        results_dir=tmp_path / "results",
        mode="context-base-rate",
        model="context",
        constant_prob=0.5,
        limit_groups=None,
        max_workers=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        allow_loose_parse=False,
        cache_dir=None,
        no_cache=True,
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        tinker_api_key_env="TINKER_API_KEY",
        tinker_base_model="openai/gpt-oss-120b",
    )
    summary, *_ = evaluate(config)
    assert summary["variant_coverage"] == 1.0
    assert abs(summary["mean_brier"] - 0.04) < 1e-9
