"""Tests for the training-set assembler."""

from __future__ import annotations

import json
from pathlib import Path

from frame_invariance.data.build_training_set import (
    SplitConfig,
    assemble_rows,
    render_user_prompt,
    split_ids,
    stratify_key,
)
from frame_invariance.data.schema import Question


def _question(qid: str, *, freeze="2025-08-01", resolved="2025-09-15", outcome=1) -> Question:
    full_id = qid if "::" in qid else f"forecastbench::{qid}"
    return Question(
        id=full_id,
        question=f"Will {qid} happen?",
        outcome=outcome,
        freeze_date=freeze,
        resolved_at=resolved,
        source="forecastbench",
        url=None,
        background="bg",
        resolution_criteria="rc",
    )


def _context(qid: str, *, news_n: int = 2):
    full_id = qid if "::" in qid else f"forecastbench::{qid}"
    return {
        "question_id": full_id,
        "base_rate": {
            "value": 0.3,
            "n_reference_events": 10,
            "reference_window_years": 5,
            "explanation": "test reference class",
        },
        "news_snapshot": [
            {"date": "2025-07-15", "headline": f"h{i}", "summary": f"s{i}"}
            for i in range(news_n)
        ],
        "leakage_filtered_count": 0,
    }


def _paraphrase_row(qid: str, *, k: int = 5):
    full_id = qid if "::" in qid else f"forecastbench::{qid}"
    paras = [{"variant_index": 0, "text": f"Will {qid} happen?", "is_original": True}]
    for i in range(1, k):
        paras.append({"variant_index": i, "text": f"Does {qid} happen variant {i}?",
                      "is_original": False})
    return {"question_id": full_id, "paraphrases": paras, "rejected_count": 0, "rounds": 1}


def test_render_user_prompt_includes_all_fields():
    prompt = render_user_prompt(
        "Will X happen?",
        freeze_date="2025-08-01",
        resolution_date="2025-09-15",
        source="polymarket",
        background="some bg",
        resolution_criteria="rc text",
        base_rate={"value": 0.18, "n_reference_events": 23,
                   "reference_window_years": 10, "explanation": "test"},
        news_snapshot=[{"date": "2025-07-01", "headline": "H", "summary": "S"}],
    )
    assert "Will X happen?" in prompt
    assert "2025-08-01" in prompt
    assert "2025-09-15" in prompt
    assert "polymarket" in prompt
    assert "Probability:" in prompt  # the output instruction
    assert "0.180" in prompt  # base rate value
    assert "[2025-07-01]" in prompt
    assert "rc text" in prompt


def test_stratify_key_by_month():
    q = _question("q1", resolved="2025-09-15", freeze="2025-09-01")
    assert stratify_key(q, "resolved_month") == "2025-09"
    assert stratify_key(q, "none") == "all"


def test_split_basic_proportions():
    questions = [_question(f"q{i}", resolved=f"2025-{9 if i < 50 else 10:02d}-15",
                           freeze=f"2025-{9 if i < 50 else 10:02d}-01")
                 for i in range(100)]
    cfg = SplitConfig(train_frac=0.8, val_frac=0.1)
    splits = split_ids(questions, cfg)
    # Approximate proportions (rounding can move 1-2 ids between buckets)
    assert 75 <= len(splits["train"]) <= 85
    assert 5 <= len(splits["validation"]) <= 15
    assert 5 <= len(splits["test"]) <= 15
    # No overlap
    assert not (splits["train"] & splits["validation"])
    assert not (splits["train"] & splits["test"])


def test_split_fixed_test_ids_reserved():
    questions = [_question(f"q{i}",
                           resolved=f"2025-{(i % 3) + 9:02d}-15",
                           freeze=f"2025-{(i % 3) + 9:02d}-01")
                 for i in range(30)]
    fixed = {"forecastbench::q0", "forecastbench::q5", "forecastbench::q10"}
    cfg = SplitConfig(train_frac=0.8, val_frac=0.1)
    splits = split_ids(questions, cfg, fixed_test_ids=fixed)
    assert fixed.issubset(splits["test"])
    assert not (fixed & splits["train"])
    assert not (fixed & splits["validation"])


def test_assemble_rows_emits_k_per_question():
    questions = [_question("q1"), _question("q2")]
    contexts = {"forecastbench::q1": _context("q1"),
                "forecastbench::q2": _context("q2")}
    paras = {"forecastbench::q1": _paraphrase_row("q1"),
             "forecastbench::q2": _paraphrase_row("q2")}
    splits = {"train": {"forecastbench::q1"}, "validation": {"forecastbench::q2"}, "test": set()}

    rows, stats = assemble_rows(questions, contexts, paras, k=5, splits=splits)
    assert len(rows) == 10  # 2 questions × K=5
    assert stats["groups_emitted"] == 2
    # Each group has variant_index 0..4 and same id
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    for qid, group in by_id.items():
        idxs = sorted(r["variant_index"] for r in group)
        assert idxs == [0, 1, 2, 3, 4]
        assert all(r["outcome"] == group[0]["outcome"] for r in group)
        # variant 0 is the original
        v0 = next(r for r in group if r["variant_index"] == 0)
        assert v0["is_original"]
        # native id was "q1" or "q2"
        native = qid.split("::", 1)[1]
        assert v0["question"] == f"Will {native} happen?"


def test_assemble_truncates_oversized_paraphrase_groups_to_k():
    """Old K=8 rows should be trimmed when target K=5."""

    questions = [_question("q1")]
    contexts = {"forecastbench::q1": _context("q1")}
    paras = {"forecastbench::q1": _paraphrase_row("q1", k=8)}  # 8 paraphrases
    splits = {"train": {"forecastbench::q1"}, "validation": set(), "test": set()}

    rows, stats = assemble_rows(questions, contexts, paras, k=5, splits=splits)
    assert len(rows) == 5
    idxs = sorted(r["variant_index"] for r in rows)
    assert idxs == [0, 1, 2, 3, 4]


def test_assemble_drops_questions_with_too_few_paraphrases():
    questions = [_question("q1"), _question("q2")]
    contexts = {"forecastbench::q1": _context("q1"),
                "forecastbench::q2": _context("q2")}
    paras = {
        "forecastbench::q1": _paraphrase_row("q1", k=5),
        "forecastbench::q2": _paraphrase_row("q2", k=3),  # too few
    }
    splits = {"train": {"forecastbench::q1", "forecastbench::q2"},
              "validation": set(), "test": set()}

    rows, stats = assemble_rows(questions, contexts, paras, k=5, splits=splits)
    assert len(rows) == 5  # only q1
    assert stats["paraphrase_group_too_small"] == 1


def test_assemble_drops_questions_missing_context():
    questions = [_question("q1"), _question("q2")]
    contexts = {"forecastbench::q1": _context("q1")}  # q2 missing
    paras = {"forecastbench::q1": _paraphrase_row("q1"),
             "forecastbench::q2": _paraphrase_row("q2")}
    splits = {"train": {"forecastbench::q1", "forecastbench::q2"},
              "validation": set(), "test": set()}

    rows, stats = assemble_rows(questions, contexts, paras, k=5, splits=splits)
    assert len(rows) == 5
    assert stats["missing_context"] == 1


def test_assemble_emits_split_label_per_row():
    questions = [_question("q1"), _question("q2"), _question("q3")]
    contexts = {q.id: _context(q.id.split("::", 1)[1]) for q in questions}
    paras = {q.id: _paraphrase_row(q.id.split("::", 1)[1]) for q in questions}
    splits = {"train": {"forecastbench::q1"},
              "validation": {"forecastbench::q2"},
              "test": {"forecastbench::q3"}}

    rows, _ = assemble_rows(questions, contexts, paras, k=5, splits=splits)
    by_id = {r["id"]: r["split"] for r in rows}
    assert by_id["forecastbench::q1"] == "train"
    assert by_id["forecastbench::q2"] == "validation"
    assert by_id["forecastbench::q3"] == "test"
