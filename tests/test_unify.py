"""Unit tests for the unifier."""

from __future__ import annotations

from frame_invariance.data.schema import Question
from frame_invariance.data.unify import (
    canonical_url,
    deduplicate,
    normalize_text,
    token_jaccard,
)


def _q(source, native, *, question="Will X happen by 2025-09-01?", url=None, outcome=1,
       freeze="2025-08-01", resolved="2025-09-01", background=None):
    return Question(
        id=f"{source}::{native}",
        question=question,
        outcome=outcome,
        freeze_date=freeze,
        resolved_at=resolved,
        source=source,
        url=url,
        background=background,
    )


def test_canonical_url_strips_query_and_slash():
    assert canonical_url("https://example.com/foo/?utm=x") == "https://example.com/foo"
    assert canonical_url("https://EXAMPLE.com/foo/") == "https://example.com/foo"
    assert canonical_url("http://example.com/foo") == "https://example.com/foo"
    assert canonical_url(None) is None


def test_token_jaccard_basic():
    a = normalize_text("Will X happen by 2025-09-01?")
    b = normalize_text("Will X happen by 2025 09 01")
    assert token_jaccard(a, b) >= 0.5


def test_url_dedupe_keeps_higher_priority():
    a = _q("forecastbench", "abc", url="https://polymarket.com/event/will-x-happen")
    b = _q("polymarket", "0xabc", url="https://polymarket.com/event/will-x-happen/")
    kept, stats = deduplicate([a, b])
    assert stats["url_duplicates_dropped"] == 1
    assert len(kept) == 1
    # Polymarket has higher priority than forecastbench
    assert kept[0].source == "polymarket"


def test_text_dedupe_cross_source():
    a = _q("polymarket", "0xabc", question="Will Trump win the 2024 election?",
           freeze="2024-11-01", resolved="2024-11-06")
    b = _q("manifold", "abc123", question="Will Donald Trump win the 2024 election",
           freeze="2024-11-01", resolved="2024-11-06")
    kept, stats = deduplicate([a, b])
    # Same source-priority tier doesn't apply; PM beats Manifold.
    assert stats["text_duplicates_dropped"] == 1
    assert len(kept) == 1
    assert kept[0].source == "polymarket"


def test_text_dedupe_window_excludes_far_apart():
    a = _q("polymarket", "0xabc", question="Will X happen by 2025-09-01?",
           freeze="2025-01-01", resolved="2025-01-15")
    b = _q("manifold", "abc123", question="Will X happen by 2025-09-01?",
           freeze="2025-09-01", resolved="2025-09-15")
    kept, _ = deduplicate([a, b], date_window_days=14)
    # 8-month gap -> not deduplicated even with identical text
    assert len(kept) == 2


def test_no_dedupe_within_same_source():
    a = _q("polymarket", "0xa", question="Same text X", freeze="2025-08-01", resolved="2025-09-01")
    b = _q("polymarket", "0xb", question="Same text X", freeze="2025-08-01", resolved="2025-09-01")
    kept, stats = deduplicate([a, b])
    # Within-source duplicates pass through (we trust source-native ids)
    assert stats["text_duplicates_dropped"] == 0
    assert len(kept) == 2


def test_priority_keeps_mantic_aib():
    a = _q("metaculus", "12345_67890",
           question="Will X happen by 2025-09-01?",
           url="https://www.metaculus.com/questions/12345/")
    b = _q("mantic_q2_aib", "12345_67890",
           question="Will X happen by 2025-09-01?",
           url="https://www.metaculus.com/questions/12345/")
    kept, _ = deduplicate([a, b])
    assert len(kept) == 1
    assert kept[0].source == "mantic_q2_aib"
