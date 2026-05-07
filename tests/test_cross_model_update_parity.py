"""Tests for the cross-model metacognitive parity evaluator."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import cross_model_update_parity as cross


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _training_row() -> dict:
    return {
        "id": "forecastbench::q1",
        "variant_index": 0,
        "split": "train",
        "question": "Will X happen?",
        "outcome": 1,
        "freeze_date": "2025-08-01",
        "resolved_at": "2025-09-01",
        "source": "forecastbench",
        "base_rate": {
            "value": 0.25,
            "n_reference_events": 10,
            "reference_window_years": 5,
            "explanation": "test prior",
        },
    }


def _question_row() -> dict:
    return {
        "id": "forecastbench::q1",
        "question": "Will X happen?",
        "outcome": 1,
        "freeze_date": "2025-08-01",
        "resolved_at": "2025-09-01",
        "source": "forecastbench",
        "background": "background text",
        "resolution_criteria": "resolution criteria",
        "categories": [],
        "raw": {},
    }


def _rewrite_row() -> dict:
    return {
        "question_id": "forecastbench::q1",
        "valid": True,
        "source_news_hash": "abc",
        "rewrites": [
            {
                "update_variant_index": 0,
                "style": "neutral",
                "stance": "neutral",
                "strength": 0,
                "update_text": "Neutral evidence.",
            }
        ],
    }


def _write_target_rows(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "prior_parseable",
        "actual_parseable",
        "prior_prob",
        "actual_prob",
        "prior_completion",
        "actual_completion",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "id": "forecastbench::q1",
                "prior_parseable": "True",
                "actual_parseable": "True",
                "prior_prob": "0.40",
                "actual_prob": "0.60",
                "prior_completion": "Probability: 0.40\nTarget prior rationale.",
                "actual_completion": "Probability: 0.60\nTarget actual rationale.",
            }
        )


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        last = messages[-1]["content"]
        assert "Cross-model posterior prediction" in last
        assert "Target model: gpt-oss-120B" in last
        assert "gpt-oss-120B's current forecast is P(X) = 0.400000" in last
        assert "Neutral evidence." in last
        return "Probability: 0.55\nObserver rationale."


def test_cross_model_parity_uses_target_rows_and_checkpoints(
    tmp_path: Path, monkeypatch
):
    training = tmp_path / "training.jsonl"
    unified = tmp_path / "unified.jsonl"
    rewrites = tmp_path / "rewrites.jsonl"
    target_rows = tmp_path / "target_rows.csv"
    _write_jsonl(training, [_training_row()])
    _write_jsonl(unified, [_question_row()])
    _write_jsonl(rewrites, [_rewrite_row()])
    _write_target_rows(target_rows)

    fake = FakePredictor()
    monkeypatch.setattr(cross, "make_observer_predictor", lambda config: fake)

    config = cross.CrossConfig(
        training=training,
        unified=unified,
        rewrites=rewrites,
        target_rows=target_rows,
        split="train",
        run_name="cross_smoke",
        results_dir=tmp_path / "results",
        mode="tinker",
        observer_model="openai/gpt-oss-20b",
        observer_label="gpt-oss-20B",
        target_label="gpt-oss-120B",
        limit_groups=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        cache_dir=None,
        no_cache=True,
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        tinker_api_key_env="TINKER_API_KEY",
        tinker_base_model="openai/gpt-oss-20b",
    )
    summary, rows_path, summary_path = cross.evaluate(config)

    assert summary["protocol"] == "observer_predicts_target_posterior"
    assert summary["observer_label"] == "gpt-oss-20B"
    assert summary["target_label"] == "gpt-oss-120B"
    assert summary["n_parseable_pairs"] == 1
    assert summary["pair_coverage"] == 1.0
    assert summary_path.exists()
    assert len(fake.calls) == 1

    row = next(csv.DictReader(rows_path.open()))
    assert row["prior_source"] == "target_elicited"
    assert row["prior_prob"] == "0.4"
    assert row["hypothetical_prob"] == "0.55"
    assert row["actual_prob"] == "0.6"
    assert row["hypothetical_parseable"] == "True"
    assert row["actual_parseable"] == "True"
    checkpoint = rows_path.with_name("cross_model_update_rows.jsonl")
    assert checkpoint.exists()
    assert len(checkpoint.read_text().strip().splitlines()) == 1

    second_fake = FakePredictor()
    monkeypatch.setattr(cross, "make_observer_predictor", lambda config: second_fake)
    second_summary, second_rows_path, _ = cross.evaluate(config)
    assert second_summary["n_parseable_pairs"] == 1
    assert second_rows_path == rows_path
    assert second_fake.calls == []
