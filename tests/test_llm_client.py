"""Tests for the Claude client wrapper. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frame_invariance.llm.client import (
    ClaudeClient,
    ClaudeRequest,
    ClaudeResponse,
)


def test_cache_key_stable_under_field_order():
    a = ClaudeRequest.make(system="s", user="hello", model="m")
    b = ClaudeRequest.make(model="m", system="s", user="hello")
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_on_temperature():
    a = ClaudeRequest.make(system="s", user="hi", temperature=0.0)
    b = ClaudeRequest.make(system="s", user="hi", temperature=0.5)
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_on_user_text():
    a = ClaudeRequest.make(system="s", user="hi")
    b = ClaudeRequest.make(system="s", user="hello")
    assert a.cache_key() != b.cache_key()


def test_response_roundtrip():
    r = ClaudeResponse(text="abc", usage={"input_tokens": 5}, stop_reason="end_turn")
    back = ClaudeResponse.from_dict(r.to_dict())
    assert back.text == r.text
    assert back.usage == r.usage
    assert back.stop_reason == r.stop_reason
    assert not back.cached  # unmarked unless explicitly set


def test_cache_hit_returns_cached_response(tmp_path: Path, monkeypatch):
    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)
    req = ClaudeRequest.make(system="sys", user="hi")

    # Pre-populate the cache
    fake = ClaudeResponse(text="cached output", usage={"input_tokens": 10})
    cache_path = tmp_path / f"{req.cache_key()}.json"
    cache_path.write_text(json.dumps(fake.to_dict()))

    sent = []

    def _no_send(*_args, **_kwargs):
        sent.append(1)
        raise AssertionError("network should not be hit on cache hit")

    monkeypatch.setattr(client, "_send_via_sdk", _no_send)
    monkeypatch.setattr(client, "_send_via_urllib", _no_send)

    resp = client.send(req)
    assert resp.text == "cached output"
    assert resp.cached is True
    assert sent == []


def test_cache_miss_calls_transport_and_writes_cache(tmp_path: Path, monkeypatch):
    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)
    req = ClaudeRequest.make(system="sys", user="hi")

    monkeypatch.setattr(
        client,
        "_send_via_urllib",
        lambda r: ClaudeResponse(text="fresh", usage={"input_tokens": 1}),
    )
    monkeypatch.setattr(client, "_sdk_client", None)

    resp = client.send(req)
    assert resp.text == "fresh"
    assert not resp.cached

    # Re-send: should hit cache, no transport call
    monkeypatch.setattr(
        client,
        "_send_via_urllib",
        lambda r: (_ for _ in ()).throw(AssertionError("should not call transport")),
    )
    resp2 = client.send(req)
    assert resp2.text == "fresh"
    assert resp2.cached


def test_disable_cache(tmp_path: Path, monkeypatch):
    client = ClaudeClient(api_key="dummy", cache_dir=tmp_path)
    req = ClaudeRequest.make(system="sys", user="hi")

    calls = []

    def _send(_r):
        calls.append(1)
        return ClaudeResponse(text=f"call{len(calls)}")

    monkeypatch.setattr(client, "_send_via_urllib", _send)
    monkeypatch.setattr(client, "_sdk_client", None)

    r1 = client.send(req, use_cache=False)
    r2 = client.send(req, use_cache=False)
    assert r1.text == "call1"
    assert r2.text == "call2"
