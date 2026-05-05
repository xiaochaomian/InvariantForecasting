"""Tests for strong evidence-update rewrite parsing and validation."""

from __future__ import annotations

from frame_invariance.data.update_rewrites_llm import (
    build_prompt,
    normalize_rewrites,
    parse_rewrite_response,
    sanitize_update_text,
    validate_rewrite_row,
)
from frame_invariance.data.schema import Question


def _question() -> Question:
    return Question(
        id="forecastbench::q1",
        question="Will X happen?",
        outcome=1,
        freeze_date="2025-08-01",
        resolved_at="2025-09-01",
        source="forecastbench",
        background="background",
        resolution_criteria="criteria",
    )


def _context():
    return {
        "question_id": "forecastbench::q1",
        "base_rate": {"value": 0.4},
        "news_snapshot": [
            {"date": "2025-07-01", "headline": "A happened", "summary": "First fact."},
            {"date": "2025-07-15", "headline": "B happened", "summary": "Second fact."},
        ],
    }


def _rewrite_text(prefix: str) -> str:
    return (
        f"{prefix}: [2025-07-01] A happened and First fact. "
        "[2025-07-15] B happened and Second fact. "
        "The wording changes emphasis but keeps the evidence bounded to these dated items."
    )


def test_build_prompt_includes_framing_contract():
    prompt = build_prompt(_question(), _context())
    assert "weak_yes" in prompt
    assert "strong_no" in prompt
    assert "Do not output a probability" in prompt
    assert "Do not discuss the" in prompt
    assert "30 days" in prompt
    assert "2025-07-01" in prompt


def test_parse_and_normalize_rewrites():
    raw = {
        "rewrites": [
            {"style": "neutral", "update_text": _rewrite_text("Neutral")},
            {"style": "weak_yes", "update_text": _rewrite_text("Weak yes")},
            {"style": "strong_yes", "update_text": _rewrite_text("Strong yes")},
            {"style": "weak_no", "update_text": _rewrite_text("Weak no")},
            {"style": "strong_no", "update_text": _rewrite_text("Strong no")},
        ]
    }
    parsed = parse_rewrite_response(__import__("json").dumps(raw))
    normalized = normalize_rewrites(parsed)
    assert [r["update_variant_index"] for r in normalized] == [0, 1, 2, 3, 4]
    assert [r["stance"] for r in normalized] == ["neutral", "yes", "yes", "no", "no"]
    assert [r["strength"] for r in normalized] == [0, 1, 2, 1, 2]


def test_validate_rewrite_row_rejects_probability_and_post_freeze_date():
    rewrites = normalize_rewrites(
        [
            {"style": "neutral", "update_text": _rewrite_text("Neutral")},
            {"style": "weak_yes", "update_text": _rewrite_text("Weak yes")},
            {"style": "strong_yes", "update_text": _rewrite_text("Strong yes")},
            {"style": "weak_no", "update_text": _rewrite_text("Weak no")},
            {
                "style": "strong_no",
                "update_text": _rewrite_text("Strong no")
                + " Probability: 0.2. A later item appeared on 2025-08-03.",
            },
        ]
    )
    result = validate_rewrite_row(
        {"rewrites": rewrites},
        context=_context(),
        freeze_date="2025-08-01",
    )
    assert not result.ok
    assert any("forbidden leakage" in e for e in result.errors)
    assert any("post-freeze date" in e for e in result.errors)
    assert any("non-evidence date" in e for e in result.errors)


def test_sanitize_update_text_removes_boilerplate_mechanics():
    text = (
        "The evidence does not directly enumerate event counts for the specific "
        "30-day window in question."
    )
    sanitized = sanitize_update_text(text)
    assert "30-day" not in sanitized
    assert "specific period in question" in sanitized
