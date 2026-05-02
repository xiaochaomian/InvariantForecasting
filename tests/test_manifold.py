"""Unit tests for the Manifold puller. Hand-built fixtures; no network."""

from __future__ import annotations

from frame_invariance.data.manifold import market_to_question, _flatten_richtext


def _fixture(**overrides):
    base = {
        "id": "abc123",
        "slug": "will-x-happen",
        "creatorUsername": "alice",
        "question": "Will X happen by Sept 2025?",
        "textDescription": "Resolves YES if X happens.",
        "outcomeType": "BINARY",
        "isResolved": True,
        "resolution": "YES",
        "volume": 5000.0,
        "closeTime": 1756857600000,  # 2025-09-03 UTC ms
        "resolutionTime": 1757462400000,  # 2025-09-10 UTC ms
        "groupSlugs": ["politics", "tech"],
    }
    base.update(overrides)
    return base


def test_yes_resolved():
    q = market_to_question(_fixture())
    assert q is not None
    assert q.outcome == 1
    assert q.id == "manifold::abc123"
    assert q.url == "https://manifold.markets/alice/will-x-happen"
    assert q.freeze_date == "2025-09-03"
    assert q.resolved_at == "2025-09-10"
    assert "politics" in q.categories


def test_no_resolved():
    q = market_to_question(_fixture(resolution="NO"))
    assert q is not None and q.outcome == 0


def test_mkt_resolution_filtered():
    assert market_to_question(_fixture(resolution="MKT")) is None


def test_cancel_resolution_filtered():
    assert market_to_question(_fixture(resolution="CANCEL")) is None


def test_low_volume_filtered():
    assert market_to_question(_fixture(volume=10)) is None


def test_unresolved_filtered():
    assert market_to_question(_fixture(isResolved=False)) is None


def test_non_binary_filtered():
    assert market_to_question(_fixture(outcomeType="MULTIPLE_CHOICE")) is None


def test_pre_cutoff_filtered():
    # 2024-01 close
    assert (
        market_to_question(
            _fixture(closeTime=1704067200000, resolutionTime=1706745600000)
        )
        is None
    )


def test_richtext_description_flattens():
    rich = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "Hello "}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "world."}]},
        ],
    }
    flat = _flatten_richtext(rich)
    assert "Hello" in flat and "world" in flat
