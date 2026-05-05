"""Tests for prior/update/posterior V2.5 assembly."""

from __future__ import annotations

from frame_invariance.data.build_training_v2_5 import assemble_v2_rows, audit_rows
from frame_invariance.data.schema import Question


def _question() -> Question:
    return Question(
        id="forecastbench::q1",
        question="Will X happen?",
        outcome=1,
        freeze_date="2025-08-01",
        resolved_at="2025-09-01",
        source="forecastbench",
        background="background text",
        resolution_criteria="resolution text",
    )


def _training_rows():
    return [
        {
            "id": "forecastbench::q1",
            "variant_index": 0,
            "split": "train",
            "question": "Will X happen?",
            "outcome": 1,
            "freeze_date": "2025-08-01",
            "resolved_at": "2025-09-01",
            "source": "forecastbench",
            "base_rate": {"value": 0.4, "n_reference_events": 10, "reference_window_years": 5},
            "news_snapshot": [{"date": "2025-07-01", "headline": "A", "summary": "B"}],
            "categories": ["test"],
        },
        {
            "id": "forecastbench::q1",
            "variant_index": 1,
            "split": "train",
            "question": "Does X happen?",
            "outcome": 1,
            "freeze_date": "2025-08-01",
            "resolved_at": "2025-09-01",
            "source": "forecastbench",
            "base_rate": {"value": 0.4, "n_reference_events": 10, "reference_window_years": 5},
            "news_snapshot": [{"date": "2025-07-01", "headline": "A", "summary": "B"}],
            "categories": ["test"],
        },
    ]


def _rewrites():
    return {
        "forecastbench::q1": {
            "question_id": "forecastbench::q1",
            "source_news_hash": "abc",
            "rewrites": [
                {"style": "neutral", "stance": "neutral", "strength": 0, "update_text": "neutral update"},
                {"style": "weak_yes", "stance": "yes", "strength": 1, "update_text": "weak yes update"},
                {"style": "strong_yes", "stance": "yes", "strength": 2, "update_text": "strong yes update"},
                {"style": "weak_no", "stance": "no", "strength": 1, "update_text": "weak no update"},
                {"style": "strong_no", "stance": "no", "strength": 2, "update_text": "strong no update"},
            ],
        }
    }


def test_assemble_v2_groups_by_question_paraphrase_and_update_variant():
    rows, stats = assemble_v2_rows(
        training_rows=_training_rows(),
        questions_by_id={"forecastbench::q1": _question()},
        rewrites_by_id=_rewrites(),
        k_updates=5,
    )
    assert len(rows) == 10
    assert stats["groups_emitted"] == 2
    ids = sorted({r["id"] for r in rows})
    assert ids == ["forecastbench::q1::qv0", "forecastbench::q1::qv1"]
    for row in rows:
        assert row["variant_index"] == row["update_variant_index"]
        assert row["v2_5_format"] == "prior_update_posterior"
        assert len(row["messages"]) == 4
        assert row["messages"][2]["content"] == "Prior probability: 0.400"

    audit = audit_rows(rows, stats, k_updates=5)
    assert audit["bad_group_count"] == 0
    assert audit["group_count"] == 2


def test_assemble_v2_can_use_original_question_only():
    rows, stats = assemble_v2_rows(
        training_rows=_training_rows(),
        questions_by_id={"forecastbench::q1": _question()},
        rewrites_by_id=_rewrites(),
        k_updates=5,
        question_variants="original-only",
    )
    assert len(rows) == 5
    assert {r["id"] for r in rows} == {"forecastbench::q1::qv0"}
    assert stats["skipped_non_original_question_variant"] == 1
