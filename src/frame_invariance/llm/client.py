"""Anthropic Claude client wrapper.

Two responsibilities:
  1. **On-disk content-addressed cache.** The cache key is a SHA-256 of
     ``(model, system, messages, max_tokens, temperature)``. Re-running a
     pipeline that's already produced a response is free; partial results from
     an interrupted batch are reusable on the next run.
  2. **Retry with exponential backoff** on rate limits and transient 5xx
     errors. We don't try to be clever about token-bucket rate limiting; the
     retry-on-429 with backoff is enough for a few-thousand-call batch.

The interface intentionally avoids streaming and tool use — both context
generation and paraphrasing are single-turn, plain-text-out (parsed as JSON
later by the caller).

We program against the ``anthropic`` Python SDK if installed, with a
``urllib``-only fallback that hits the Messages REST endpoint directly. The
fallback exists so tests can run in environments without the SDK and so the
wrapper has no hard dependency on an SDK version.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.0  # deterministic where Claude permits
DEFAULT_TIMEOUT_S = 60
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_S = 2.0
DEFAULT_BACKOFF_MAX_S = 60.0
ENV_API_KEY = "ANTHROPIC_API_KEY"
MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"


@dataclass(frozen=True)
class ClaudeRequest:
    """Inputs that go into the cache key. Everything Claude sees lives here."""

    model: str
    system: str
    messages: tuple[tuple[str, str], ...]  # (role, content) pairs, frozen
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def make(
        cls,
        *,
        model: str = DEFAULT_MODEL,
        system: str,
        user: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> "ClaudeRequest":
        return cls(
            model=model,
            system=system,
            messages=(("user", user),),
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": self.system,
            "messages": [{"role": r, "content": c} for r, c in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

    def cache_key(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ClaudeResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, cached: bool = False) -> "ClaudeResponse":
        return cls(
            text=str(payload.get("text", "")),
            usage=dict(payload.get("usage") or {}),
            stop_reason=payload.get("stop_reason"),
            cached=cached,
        )


class ClaudeError(RuntimeError):
    """Non-retryable Claude API error (4xx other than 429, malformed input)."""


class ClaudeClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: Path | str | None = "data/cache/claude",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
        backoff_max_s: float = DEFAULT_BACKOFF_MAX_S,
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key or os.environ.get(ENV_API_KEY)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._backoff_max_s = backoff_max_s
        self._timeout_s = timeout_s

        self._sdk_client: Any | None = None
        try:
            import anthropic  # type: ignore

            if self._api_key:
                self._sdk_client = anthropic.Anthropic(
                    api_key=self._api_key, timeout=self._timeout_s
                )
        except ImportError:
            self._sdk_client = None

    # ------------------------------------------------------------------ cache
    def _cache_path(self, request: ClaudeRequest) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{request.cache_key()}.json"

    def _load_cached(self, request: ClaudeRequest) -> ClaudeResponse | None:
        path = self._cache_path(request)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return ClaudeResponse.from_dict(payload, cached=True)

    def _store_cached(self, request: ClaudeRequest, response: ClaudeResponse) -> None:
        path = self._cache_path(request)
        if path is None:
            return
        # Atomic write; readers may concurrently read.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(response.to_dict(), sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    # ---------------------------------------------------------------- transport
    def _send_via_sdk(self, request: ClaudeRequest) -> ClaudeResponse:
        assert self._sdk_client is not None
        msg = self._sdk_client.messages.create(
            model=request.model,
            system=request.system,
            messages=[{"role": r, "content": c} for r, c in request.messages],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        # SDK returns content blocks; we want concatenated text.
        text_parts: list[str] = []
        for block in msg.content or []:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        usage = {}
        if hasattr(msg, "usage") and msg.usage is not None:
            usage = {
                "input_tokens": getattr(msg.usage, "input_tokens", 0),
                "output_tokens": getattr(msg.usage, "output_tokens", 0),
            }
        return ClaudeResponse(
            text="".join(text_parts),
            usage=usage,
            stop_reason=getattr(msg, "stop_reason", None),
        )

    def _send_via_urllib(self, request: ClaudeRequest) -> ClaudeResponse:
        import urllib.error
        import urllib.request

        if not self._api_key:
            raise ClaudeError(
                f"no ${ENV_API_KEY} and no anthropic SDK; cannot send requests"
            )
        body = json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            MESSAGES_ENDPOINT,
            data=body,
            method="POST",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code == 429:
                raise _RetryableError(f"429 rate limited: {body_text[:200]}")
            if 500 <= e.code < 600:
                raise _RetryableError(f"{e.code} server error: {body_text[:200]}")
            raise ClaudeError(f"{e.code} from Claude: {body_text[:300]}") from e
        except urllib.error.URLError as e:
            raise _RetryableError(f"URLError: {e}") from e

        text_parts: list[str] = []
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        usage = payload.get("usage") or {}
        return ClaudeResponse(
            text="".join(text_parts),
            usage={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
            stop_reason=payload.get("stop_reason"),
        )

    # ------------------------------------------------------------------- send
    def send(self, request: ClaudeRequest, *, use_cache: bool = True) -> ClaudeResponse:
        if use_cache:
            cached = self._load_cached(request)
            if cached is not None:
                return cached

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                if self._sdk_client is not None:
                    response = self._send_via_sdk(request)
                else:
                    response = self._send_via_urllib(request)
                if use_cache:
                    self._store_cached(request, response)
                return response
            except _RetryableError as e:
                last_exc = e
                if attempt >= self._max_retries:
                    break
                sleep = min(
                    self._backoff_max_s,
                    self._backoff_s * (2**attempt) * (0.5 + random.random()),
                )
                time.sleep(sleep)
            except ClaudeError:
                raise
            except Exception as e:
                # SDK exceptions are heterogeneous; treat them as retryable
                # unless the message clearly indicates a 4xx other than 429.
                msg = str(e)
                if "429" in msg or "rate" in msg.lower() or "5" in msg[:3]:
                    last_exc = e
                    if attempt >= self._max_retries:
                        break
                    sleep = min(
                        self._backoff_max_s,
                        self._backoff_s * (2**attempt) * (0.5 + random.random()),
                    )
                    time.sleep(sleep)
                else:
                    raise ClaudeError(str(e)) from e
        raise ClaudeError(
            f"max retries ({self._max_retries}) exceeded; last error: {last_exc}"
        )


class _RetryableError(RuntimeError):
    pass
