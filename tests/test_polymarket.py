"""Unit tests for the Polymarket puller. Hand-built fixtures; no network."""

from __future__ import annotations

from frame_invariance.data.polymarket import market_to_question


def _fixture(**overrides):
    base = {
        "id": "0x123",
        "conditionId": "0xabc",
        "slug": "will-x-happen-by-2025",
        "question": "Will X happen by Sept 1, 2025?",
        "description": "Resolves YES if X happens before Sept 1, 2025.",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["1.0","0.0"]',
        "closed": True,
        "active": False,
        "volumeNum": 50000.0,
        "endDate": "2025-09-01T00:00:00Z",
        "umaEndDate": "2025-09-05T00:00:00Z",
        "tags": [{"label": "crypto"}, "Politics"],
    }
    base.update(overrides)
    return base


def test_yes_resolved():
    q = market_to_question(_fixture())
    assert q is not None
    assert q.outcome == 1
    assert q.id == "polymarket::0xabc"
    assert q.url == "https://polymarket.com/event/will-x-happen-by-2025"
    assert q.freeze_date == "2025-09-01"
    assert q.resolved_at == "2025-09-05"
    assert "crypto" in q.categories
    assert "Politics" in q.categories


def test_no_resolved():
    q = market_to_question(_fixture(outcomePrices='["0.0","1.0"]'))
    assert q is not None and q.outcome == 0


def test_open_market_filtered():
    assert market_to_question(_fixture(closed=False, active=True)) is None


def test_low_volume_filtered():
    assert market_to_question(_fixture(volumeNum=100.0)) is None


def test_non_binary_outcomes_filtered():
    assert market_to_question(_fixture(outcomes='["A","B","C"]')) is None
    assert market_to_question(_fixture(outcomes='["Trump","Biden"]')) is None


def test_ambiguous_outcome_filtered():
    # 50/50 doesn't count as resolved
    assert market_to_question(_fixture(outcomePrices='["0.5","0.5"]')) is None


def test_pre_cutoff_filtered():
    assert (
        market_to_question(
            _fixture(endDate="2024-03-01T00:00:00Z", umaEndDate="2024-03-15T00:00:00Z")
        )
        is None
    )


def test_min_volume_override():
    q = market_to_question(_fixture(volumeNum=100.0), min_volume=10.0)
    assert q is not None
