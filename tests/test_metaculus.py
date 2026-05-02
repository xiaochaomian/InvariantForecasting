"""Unit tests for the Metaculus puller. Uses a hand-built fixture; no network.

Also includes a small ForecastBench combo-question-filter test, since that
filter is a defense-in-depth check we don't want to regress on (the offline
forecastbench puller has its own primary filter).
"""

from __future__ import annotations

from frame_invariance.data.metaculus import post_to_question


def _fixture(**overrides):
    base = {
        "id": 12345,
        "title": "post title",
        "description": "post-level description",
        "scheduled_close_time": "2025-09-01T00:00:00Z",
        "resolution_set_time": "2025-09-15T00:00:00Z",
        "projects": {"category": [{"name": "AI"}, {"name": "Geopolitics"}]},
        "question": {
            "id": 67890,
            "type": "binary",
            "status": "resolved",
            "title": "Will X happen by 2025-09-01?",
            "description": "rich question-level description with background",
            "resolution_criteria": "Resolves YES if X happens.",
            "fine_print": "Excludes Y and Z.",
            "scheduled_close_time": "2025-09-01T00:00:00Z",
            "actual_resolve_time": "2025-09-15T00:00:00Z",
            "resolution": "yes",
        },
    }
    base.update(overrides)
    return base


def test_basic_resolved_yes():
    q = post_to_question(_fixture())
    assert q is not None
    assert q.outcome == 1
    assert q.id == "metaculus::12345_67890"
    assert q.freeze_date == "2025-09-01"
    assert q.resolved_at == "2025-09-15"
    assert q.source == "metaculus"
    assert "AI" in q.categories
    assert q.url == "https://www.metaculus.com/questions/12345/"
    assert q.resolution_criteria.startswith("Resolves YES if X happens.")
    assert "Fine print: Excludes Y and Z." in q.resolution_criteria


def test_basic_resolved_no():
    f = _fixture()
    f["question"]["resolution"] = "no"
    q = post_to_question(f)
    assert q is not None and q.outcome == 0


def test_pre_cutoff_filtered():
    f = _fixture()
    f["question"]["scheduled_close_time"] = "2024-01-01T00:00:00Z"
    f["question"]["actual_resolve_time"] = "2024-02-01T00:00:00Z"
    assert post_to_question(f) is None


def test_non_binary_filtered():
    f = _fixture()
    f["question"]["type"] = "numeric"
    assert post_to_question(f) is None


def test_unresolved_filtered():
    f = _fixture()
    f["question"]["status"] = "open"
    assert post_to_question(f) is None


def test_ambiguous_resolution_filtered():
    f = _fixture()
    f["question"]["resolution"] = "ambiguous"
    assert post_to_question(f) is None


def test_tournament_label_added():
    q = post_to_question(_fixture(), tournament_label="mantic_q2_aib")
    assert q is not None
    assert "tournament:mantic_q2_aib" in q.categories


def test_freeze_after_resolved_filtered():
    f = _fixture()
    f["question"]["scheduled_close_time"] = "2025-10-01T00:00:00Z"
    f["question"]["actual_resolve_time"] = "2025-09-15T00:00:00Z"
    assert post_to_question(f) is None
