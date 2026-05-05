"""Tests for metacognitive update prediction evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from frame_invariance.eval import metacognitive_update as mcu


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _training_row() -> dict:
    return {
        "id": "forecastbench::q1",
        "variant_index": 0,
        "split": "validation",
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


class FakePredictor:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        last = messages[-1]["content"]
        if "give your current forecast now" in last:
            return "Probability: 0.40\nPrior rationale."
        if "Metacognitive update prediction" in last:
            return "Probability: 0.70\nHypothetical rationale."
        if "Update phase" in last:
            return "Probability: 0.60\nActual rationale."
        raise AssertionError(f"unexpected prompt: {last}")


def test_evaluate_exact_protocol_with_elicited_prior(tmp_path: Path, monkeypatch):
    training = tmp_path / "training.jsonl"
    unified = tmp_path / "unified.jsonl"
    rewrites = tmp_path / "rewrites.jsonl"
    _write_jsonl(training, [_training_row()])
    _write_jsonl(unified, [_question_row()])
    _write_jsonl(rewrites, [_rewrite_row()])

    fake = FakePredictor()
    monkeypatch.setattr(mcu, "make_predictor", lambda config: fake)

    config = mcu.Config(
        training=training,
        unified=unified,
        rewrites=rewrites,
        split="validation",
        run_name="exact_smoke",
        results_dir=tmp_path / "results",
        mode="tinker",
        model="fake",
        prior_source="elicited",
        limit_groups=1,
        max_workers=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        cache_dir=None,
        no_cache=True,
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        tinker_api_key_env="TINKER_API_KEY",
        tinker_base_model="openai/gpt-oss-120b",
    )
    summary, rows_path, summary_path = mcu.evaluate(config)

    assert summary["prior_source"] == "elicited"
    assert summary["n_prior_parseable"] == 1
    assert summary["n_parseable_pairs"] == 1
    assert summary["pair_coverage"] == 1.0
    assert "logodds_shift_spearman" in summary
    assert summary_path.exists()
    assert len(fake.calls) == 3
    assert "give your current forecast now" in fake.calls[0][-1]["content"]
    assert fake.calls[1][2]["content"].startswith("Probability: 0.40")
    assert fake.calls[2][2]["content"] == (
        "Understood. I will use the prior phase as the starting point."
    )

    row = next(csv.DictReader(rows_path.open()))
    assert row["prior_source"] == "elicited"
    assert row["stated_prior_prob"] == "0.25"
    assert row["prior_prob"] == "0.4"
    assert row["hypothetical_prob"] == "0.7"
    assert row["actual_prob"] == "0.6"
    assert row["prior_parseable"] == "True"
