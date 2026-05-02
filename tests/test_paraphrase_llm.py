"""Tests for the LLM paraphraser. No network."""

from __future__ import annotations

import json
from pathlib import Path

from frame_invariance.data.paraphrase_llm import (
    ParaphraseParseError,
    extract_dates,
    extract_numbers,
    extract_proper_nouns,
    generate_for_question,
    parse_paraphrase_list,
    validate_paraphrase,
)
from frame_invariance.data.schema import Question
from frame_invariance.llm.client import ClaudeClient


def _q(question_text: str) -> Question:
    return Question(
        id="forecastbench::abc",
        question=question_text,
        outcome=1,
        freeze_date="2025-08-01",
        resolved_at="2025-09-15",
        source="forecastbench",
    )


# ----------------------------- entity extraction ---------------------------
def test_extract_dates_iso_and_natural():
    text = "Will X happen by 2025-09-01 or before September 15, 2025?"
    dates = extract_dates(text)
    assert any("2025-09-01" in d for d in dates)
    assert any("September 15" in d for d in dates)


def test_extract_numbers_with_units():
    text = "Will TSLA reach $250 with a 25% drop by 2025-09-01?"
    nums = extract_numbers(text)
    assert "$250" in nums or "250" in nums
    assert any("25" in n for n in nums)


def test_extract_proper_nouns_skips_sentence_initial_will():
    text = "Will Tesla announce FSD by December 31?"
    propers = extract_proper_nouns(text)
    assert "Tesla" in propers
    # FSD is a single capitalized token; we should pick it up
    assert "FSD" in propers
    # Will at the start should NOT be flagged
    assert "Will" not in propers


def test_extract_proper_nouns_multi_word():
    text = "Will the United States and Venezuela enter a military conflict?"
    propers = extract_proper_nouns(text)
    assert any("United States" in p for p in propers)
    assert "Venezuela" in propers


# ---------------------------- validate_paraphrase --------------------------
def test_validate_accepts_clean_paraphrase():
    orig = "Will Tesla launch unsupervised FSD by October 31, 2025?"
    cand = "By October 31, 2025, will Tesla release its unsupervised FSD?"
    ok, reason = validate_paraphrase(orig, cand)
    assert ok, reason


def test_validate_rejects_dropped_date():
    orig = "Will Tesla launch FSD by October 31, 2025?"
    cand = "Will Tesla release FSD?"
    ok, reason = validate_paraphrase(orig, cand)
    assert not ok
    assert "missing" in reason.lower()


def test_validate_rejects_dropped_proper_noun():
    orig = "Will Tesla launch FSD by October 31, 2025?"
    cand = "Will the company launch driver assistance by October 31, 2025?"
    ok, reason = validate_paraphrase(orig, cand)
    assert not ok
    assert "missing" in reason.lower()


def test_validate_rejects_dropped_number():
    orig = "Will TSLA fall to $250 by October 31, 2025?"
    cand = "Will Tesla drop to a low price by October 31, 2025?"
    ok, reason = validate_paraphrase(orig, cand)
    assert not ok
    # could be flagged on number OR on proper noun "TSLA", both are valid
    assert "missing" in reason.lower()


def test_validate_rejects_identical():
    orig = "Will X happen?"
    cand = "Will X happen?"
    ok, reason = validate_paraphrase(orig, cand)
    assert not ok


def test_validate_rejects_empty():
    ok, _ = validate_paraphrase("Will X happen?", "")
    assert not ok


# --------------------------- response parsing ------------------------------
def test_parse_paraphrase_list_basic():
    text = '["a", "b", "c"]'
    out = parse_paraphrase_list(text)
    assert out == ["a", "b", "c"]


def test_parse_paraphrase_list_strips_fences():
    text = '```json\n["a", "b"]\n```'
    out = parse_paraphrase_list(text)
    assert out == ["a", "b"]


def test_parse_paraphrase_list_finds_embedded_list():
    text = "Here are the paraphrases:\n[\"a\", \"b\"]\nThanks!"
    out = parse_paraphrase_list(text)
    assert out == ["a", "b"]


def test_parse_paraphrase_list_handles_dicts():
    text = '[{"text": "first"}, {"text": "second"}]'
    out = parse_paraphrase_list(text)
    assert out == ["first", "second"]


def test_parse_paraphrase_list_rejects_non_list():
    text = '{"foo": 1}'
    try:
        parse_paraphrase_list(text)
        raise AssertionError("expected ParaphraseParseError")
    except ParaphraseParseError:
        pass


# ----------------------- end-to-end with mocked client ---------------------
def test_generate_for_question_full_success(tmp_path: Path, monkeypatch):
    q = _q("Will Tesla launch unsupervised FSD by October 31, 2025?")

    # Build a fake set of K-1=7 valid paraphrases for the first response.
    mocked = json.dumps(
        [
            "Does Tesla launch unsupervised FSD on or before October 31, 2025?",
            "By October 31, 2025, will Tesla release its unsupervised FSD?",
            "On or before October 31, 2025, does Tesla make unsupervised FSD generally available?",
            "Will Tesla's unsupervised FSD ship before October 31, 2025?",
            "Will the launch of Tesla unsupervised FSD occur by October 31, 2025?",
            "Will Tesla deploy unsupervised FSD no later than October 31, 2025?",
            "By October 31, 2025, has Tesla rolled out unsupervised FSD?",
        ]
    )

    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)

    from frame_invariance.llm.client import ClaudeResponse
    monkeypatch.setattr(client, "_sdk_client", None)
    monkeypatch.setattr(
        client,
        "_send_via_urllib",
        lambda r: ClaudeResponse(text=mocked),
    )

    result = generate_for_question(q, client, k=8)
    assert len(result.paraphrases) == 8
    assert result.paraphrases[0].is_original
    assert result.paraphrases[0].text == q.question
    assert all(not p.is_original for p in result.paraphrases[1:])
    # All paraphrases should pass validation
    for p in result.paraphrases[1:]:
        ok, _ = validate_paraphrase(q.question, p.text)
        assert ok, f"paraphrase failed validation: {p.text!r}"


def test_generate_for_question_retries_when_paraphrases_invalid(tmp_path: Path, monkeypatch):
    q = _q("Will Tesla launch FSD by October 31, 2025?")

    bad_first_round = json.dumps(
        [
            "Will the company launch driver assistance soon?",  # drops Tesla, FSD, date
            "Tesla will launch FSD",                              # drops date
            "Tesla launches FSD by October 31, 2025?",            # ok!
        ]
    )
    good_second_round = json.dumps(
        [
            "By October 31, 2025, will Tesla release FSD?",
            "Does Tesla launch FSD on or before October 31, 2025?",
            "Will Tesla deploy FSD before October 31, 2025?",
            "On or before October 31, 2025, does Tesla make FSD available?",
            "Will the rollout of Tesla FSD happen by October 31, 2025?",
            "Has Tesla launched FSD by October 31, 2025?",
        ]
    )

    responses = iter([bad_first_round, good_second_round])
    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)
    from frame_invariance.llm.client import ClaudeResponse
    monkeypatch.setattr(client, "_sdk_client", None)
    monkeypatch.setattr(
        client,
        "_send_via_urllib",
        lambda r: ClaudeResponse(text=next(responses)),
    )

    result = generate_for_question(q, client, k=8)
    assert len(result.paraphrases) == 8
    assert result.rounds == 2
    # The bad ones should have been recorded with reasons
    assert len(result.rejected) >= 2
