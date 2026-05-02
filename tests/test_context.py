"""Tests for the context generator. No network."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from frame_invariance.data.context import (
    ContextParseError,
    build_request,
    generate_all,
    generate_for_question,
    parse_context_response,
)
from frame_invariance.data.schema import Question
from frame_invariance.llm.client import ClaudeClient, ClaudeResponse


def _question(**overrides):
    base = dict(
        id="forecastbench::abc",
        question="Will X happen by 2025-09-01?",
        outcome=1,
        freeze_date="2025-08-01",
        resolved_at="2025-09-15",
        source="forecastbench",
        url="https://example.com",
        background="X is a thing.",
        resolution_criteria="Resolves YES if X happens.",
    )
    base.update(overrides)
    return Question(**base)


def _good_response_json(freeze_date="2025-08-01"):
    return json.dumps(
        {
            "base_rate": {
                "value": 0.18,
                "n_reference_events": 23,
                "reference_window_years": 10,
                "explanation": "Reference class: similar X events. ~4/23 resolved YES.",
            },
            "news_snapshot": [
                {"date": "2025-07-01", "headline": "thing happened", "summary": "..."},
                {"date": "2025-07-20", "headline": "another thing", "summary": "..."},
            ],
        }
    )


def test_build_request_includes_freeze_date_and_question():
    q = _question()
    req = build_request(q)
    assert q.question in req.messages[0][1]
    assert q.freeze_date in req.messages[0][1]
    assert req.system  # non-empty
    # cache key should differ for different questions
    other = build_request(_question(question="Will Y happen by 2025-09-01?"))
    assert req.cache_key() != other.cache_key()


def test_parse_basic_response():
    base, news, n_filtered = parse_context_response(
        _good_response_json(), freeze_date="2025-08-01"
    )
    assert base.value == 0.18
    assert base.n_reference_events == 23
    assert len(news) == 2
    assert news[0].date == "2025-07-01"
    assert n_filtered == 0


def test_post_freeze_news_filtered():
    payload = json.dumps(
        {
            "base_rate": {
                "value": 0.5,
                "n_reference_events": 10,
                "reference_window_years": 5,
                "explanation": "...",
            },
            "news_snapshot": [
                {"date": "2025-07-15", "headline": "valid", "summary": "..."},
                {"date": "2025-08-15", "headline": "AFTER FREEZE", "summary": "leak"},
                {"date": "2025-08-01", "headline": "ON FREEZE", "summary": "leak"},
            ],
        }
    )
    base, news, n_filtered = parse_context_response(payload, freeze_date="2025-08-01")
    assert len(news) == 1
    assert news[0].headline == "valid"
    assert n_filtered == 2


def test_unparseable_news_dates_filtered():
    payload = json.dumps(
        {
            "base_rate": {
                "value": 0.3,
                "n_reference_events": 10,
                "reference_window_years": 5,
                "explanation": "x",
            },
            "news_snapshot": [
                {"date": "not a date", "headline": "h", "summary": "s"},
                {"date": "2025-07-10", "headline": "h", "summary": "s"},
                {"date": None, "headline": "h", "summary": "s"},
            ],
        }
    )
    base, news, n_filtered = parse_context_response(payload, freeze_date="2025-08-01")
    assert len(news) == 1
    assert n_filtered == 2


def test_strips_markdown_fences():
    text = "```json\n" + _good_response_json() + "\n```"
    base, news, n = parse_context_response(text, freeze_date="2025-08-01")
    assert base.value == 0.18
    assert len(news) == 2


def test_parses_python_dict_style_response():
    """Claude sometimes returns single-quoted Python-dict literals; we should
    parse those rather than throwing ContextParseError."""

    payload = (
        "{'base_rate': {'value': 0.25, 'n_reference_events': 12, "
        "'reference_window_years': 5, 'explanation': 'reference class explanation'}, "
        "'news_snapshot': [{'date': '2025-07-15', 'headline': 'h1', 'summary': 's1'}]}"
    )
    base, news, n = parse_context_response(payload, freeze_date="2025-08-01")
    assert base.value == 0.25
    assert base.n_reference_events == 12
    assert len(news) == 1
    assert news[0].headline == "h1"


def test_parses_dict_with_prose_before_and_after():
    payload = (
        "Here is the context:\n\n"
        "{'base_rate': {'value': 0.30, 'n_reference_events': 10, "
        "'reference_window_years': 5, 'explanation': 'x'}, "
        "'news_snapshot': []}\n\n"
        "Hope that helps."
    )
    base, news, n = parse_context_response(payload, freeze_date="2025-08-01")
    assert base.value == 0.30


def test_unparseable_garbage_raises():
    try:
        parse_context_response("not json or python dict at all", freeze_date="2025-08-01")
        raise AssertionError("expected ContextParseError")
    except ContextParseError as e:
        # The error message should include enough to debug
        assert "first 120 chars" in str(e)


def test_value_out_of_range_raises():
    payload = json.dumps(
        {
            "base_rate": {"value": 1.5, "n_reference_events": 1, "reference_window_years": 1, "explanation": ""},
            "news_snapshot": [],
        }
    )
    try:
        parse_context_response(payload, freeze_date="2025-08-01")
        raise AssertionError("expected ContextParseError")
    except ContextParseError:
        pass


def test_missing_base_rate_raises():
    payload = json.dumps({"news_snapshot": []})
    try:
        parse_context_response(payload, freeze_date="2025-08-01")
        raise AssertionError("expected ContextParseError")
    except ContextParseError:
        pass


def test_generate_for_question_uses_cache(tmp_path: Path, monkeypatch):
    q = _question()
    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)

    # Pre-seed the cache with a valid response keyed off the actual request.
    req = build_request(q)
    cache_path = tmp_path / f"{req.cache_key()}.json"
    cache_path.write_text(json.dumps({"text": _good_response_json(), "usage": {}}))

    # Disable transport so we know we're hitting cache
    monkeypatch.setattr(client, "_send_via_sdk", lambda r: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(client, "_send_via_urllib", lambda r: (_ for _ in ()).throw(AssertionError()))

    result = generate_for_question(q, client)
    assert result.cache_hit
    assert result.base_rate.value == 0.18
    assert len(result.news_snapshot) == 2


def test_generate_all_resumes_from_existing_output(tmp_path: Path, monkeypatch):
    q1 = _question(id="forecastbench::a")
    q2 = _question(id="forecastbench::b")

    output = tmp_path / "contexts.jsonl"
    # Pretend q1 was already done in a prior run.
    output.write_text(
        json.dumps(
            {"question_id": "forecastbench::a", "base_rate": {}, "news_snapshot": []}
        )
        + "\n"
    )

    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path / "cache")

    # Mock the per-question driver to a deterministic result.
    from frame_invariance.data import context as ctx_mod

    def _mock(q, *_args, **_kwargs):
        from frame_invariance.data.context import (
            BaseRate,
            ContextResult,
            NewsItem,
        )

        return ContextResult(
            question_id=q.id,
            base_rate=BaseRate(0.5, 1, 1, "test"),
            news_snapshot=[NewsItem("2025-07-01", "h", "s")],
        )

    monkeypatch.setattr(ctx_mod, "generate_for_question", _mock)

    stats = generate_all([q1, q2], client, output_path=output)
    assert stats["skipped_already_done"] == 1
    assert stats["ok"] == 1  # only q2 processed

    lines = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    ids = sorted(l["question_id"] for l in lines)
    assert ids == ["forecastbench::a", "forecastbench::b"]
