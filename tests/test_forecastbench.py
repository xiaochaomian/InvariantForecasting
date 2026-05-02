"""Tests for the offline ForecastBench puller.

Uses a tmp_path-scoped fake datasets/ tree so we don't depend on the in-tree
snapshots. The most important test is the combo-question filter: ForecastBench
includes paired-question rows whose native id is a list, and we must skip them.
"""

from __future__ import annotations

import json
from pathlib import Path

from frame_invariance.data.forecastbench import iter_resolved_rows, render_question_dates


def test_render_question_dates_substitutes_both():
    text = "Will X happen between {forecast_due_date} and {resolution_date}?"
    out = render_question_dates(text, freeze_date="2025-08-01", resolved_at="2025-09-15")
    assert out == "Will X happen between 2025-08-01 and 2025-09-15?"


def test_render_question_dates_passthrough_when_no_placeholder():
    text = "Will Tesla launch FSD by October 31, 2025?"
    out = render_question_dates(text, freeze_date="2025-08-01", resolved_at="2025-10-31")
    assert out == text  # untouched


def test_render_question_dates_substitutes_only_resolution():
    text = "Will X happen by {resolution_date}?"
    out = render_question_dates(text, freeze_date="2025-08-01", resolved_at="2025-09-15")
    assert out == "Will X happen by 2025-09-15?"


def _write_qs(qs_dir: Path, name: str, questions: list[dict]) -> None:
    qs_dir.mkdir(parents=True, exist_ok=True)
    (qs_dir / name).write_text(
        json.dumps({"forecast_due_date": "2025-08-01",
                    "question_set": name,
                    "questions": questions})
    )


def _write_res(res_dir: Path, name: str, qs_name: str, resolutions: list[dict]) -> None:
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / name).write_text(
        json.dumps({
            "forecast_due_date": "2025-08-01",
            "question_set": qs_name,
            "resolutions": resolutions,
        })
    )


def test_pulls_resolved_binary_post_cutoff(tmp_path: Path):
    qs_dir = tmp_path / "question_sets"
    res_dir = tmp_path / "resolution_sets"

    _write_qs(qs_dir, "2025-08-01-llm.json", [
        {
            "id": "q1",
            "source": "polymarket",
            "question": "Will X happen by 2025-09-01?",
            "background": "X background",
            "resolution_criteria": "X resolves YES if...",
            "freeze_datetime": "2025-08-01T00:00:00Z",
            "url": "https://example.com/q1",
        },
    ])
    _write_res(res_dir, "2025-09-15_resolution_set.json", "2025-08-01-llm.json", [
        {
            "id": "q1",
            "source": "polymarket",
            "resolution_date": "2025-09-15",
            "resolved_to": 1.0,
            "resolved": True,
        },
    ])

    rows, skipped = iter_resolved_rows(datasets_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0].id == "forecastbench::q1"
    assert rows[0].outcome == 1
    assert rows[0].freeze_date == "2025-08-01"
    assert rows[0].resolved_at == "2025-09-15"


def test_skips_combo_question_with_list_id(tmp_path: Path):
    """The most important test: paired-question rows must be filtered out."""

    qs_dir = tmp_path / "question_sets"
    res_dir = tmp_path / "resolution_sets"

    _write_qs(qs_dir, "2025-08-01-llm.json", [
        {
            "id": "q1",
            "source": "polymarket",
            "question": "Will X happen by 2025-09-01?",
            "background": "X background",
            "freeze_datetime": "2025-08-01T00:00:00Z",
            "url": "https://example.com/q1",
        },
        {
            "id": "q2",
            "source": "polymarket",
            "question": "Will Y happen by 2025-09-01?",
            "background": "Y background",
            "freeze_datetime": "2025-08-01T00:00:00Z",
            "url": "https://example.com/q2",
        },
    ])
    _write_res(res_dir, "2025-09-15_resolution_set.json", "2025-08-01-llm.json", [
        {
            "id": "q1",
            "source": "polymarket",
            "resolution_date": "2025-09-15",
            "resolved_to": 1.0,
            "resolved": True,
        },
        # The combo row: id is a *list* of two ids
        {
            "id": ["q1", "q2"],
            "source": "polymarket",
            "resolution_date": "2025-09-15",
            "resolved_to": 1.0,
            "resolved": True,
        },
    ])

    rows, skipped = iter_resolved_rows(datasets_dir=tmp_path)
    assert len(rows) == 1
    assert rows[0].id == "forecastbench::q1"
    assert skipped["combo_question"] == 1


def test_skips_pre_cutoff(tmp_path: Path):
    qs_dir = tmp_path / "question_sets"
    res_dir = tmp_path / "resolution_sets"

    _write_qs(qs_dir, "2024-01-01-llm.json", [
        {
            "id": "old",
            "source": "polymarket",
            "question": "Will old thing happen?",
            "freeze_datetime": "2024-01-01T00:00:00Z",
            "url": "https://example.com/old",
        },
    ])
    _write_res(res_dir, "2024-02-15_resolution_set.json", "2024-01-01-llm.json", [
        {
            "id": "old",
            "resolution_date": "2024-02-15",
            "resolved_to": 1.0,
            "resolved": True,
        },
    ])

    rows, skipped = iter_resolved_rows(datasets_dir=tmp_path)
    assert len(rows) == 0
    assert skipped["pre_cutoff"] == 1


def test_pulls_render_question_dates(tmp_path: Path):
    """Templated FB questions should have dates substituted by the puller."""

    qs_dir = tmp_path / "question_sets"
    res_dir = tmp_path / "resolution_sets"

    _write_qs(qs_dir, "2025-08-01-llm.json", [{
        "id": "templ",
        "source": "acled",
        "question": "Will X happen between {forecast_due_date} and {resolution_date}?",
        "freeze_datetime": "2025-08-01T00:00:00Z",
        "url": "https://example.com/templ",
    }])
    _write_res(res_dir, "2025-09-15_resolution_set.json", "2025-08-01-llm.json", [{
        "id": "templ",
        "resolution_date": "2025-09-15",
        "resolved_to": 1.0,
        "resolved": True,
    }])

    rows, _ = iter_resolved_rows(datasets_dir=tmp_path)
    assert len(rows) == 1
    assert "{resolution_date}" not in rows[0].question
    assert "{forecast_due_date}" not in rows[0].question
    assert "2025-08-01" in rows[0].question
    assert "2025-09-15" in rows[0].question


def test_skips_unresolved(tmp_path: Path):
    qs_dir = tmp_path / "question_sets"
    res_dir = tmp_path / "resolution_sets"

    _write_qs(qs_dir, "2025-08-01-llm.json", [{
        "id": "q1",
        "question": "Will X?",
        "freeze_datetime": "2025-08-01T00:00:00Z",
    }])
    _write_res(res_dir, "2025-09-15_resolution_set.json", "2025-08-01-llm.json", [{
        "id": "q1",
        "resolution_date": "2025-09-15",
        "resolved_to": 1.0,
        "resolved": False,  # not resolved
    }])

    rows, skipped = iter_resolved_rows(datasets_dir=tmp_path)
    assert len(rows) == 0
    assert skipped["not_resolved"] == 1
