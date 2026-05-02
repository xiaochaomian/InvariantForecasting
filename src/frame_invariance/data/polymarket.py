"""Polymarket Gamma API puller.

Polymarket exposes ``https://gamma-api.polymarket.com/markets`` with pagination
via ``offset`` + ``limit`` (limit max 500). Resolved binary markets have
``closed: true`` and a ``umaResolutionStatuses`` history; ``outcomePrices``
holds the final resolution as ``["1.0","0.0"]`` (yes won) or ``["0.0","1.0"]``
(no won).

We filter aggressively for usable judgmental-forecasting questions:
  - binary outcomes (the 'Yes/No' market type, not multi-outcome)
  - closed and resolved (``closed and not active and outcomePrices``)
  - non-trivial volume (volumeNum >= MIN_VOLUME) — drops vanity / private markets
  - post-cutoff freeze and resolution dates

Run on a machine with internet; this sandbox can't reach Polymarket. Cached
JSON pages are written to ``data/raw/polymarket_cache/``.

Schema reference observed: https://docs.polymarket.com/  (the public docs
intermittently lag the live API; treat fields as best-effort).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

from .schema import GPT_OSS_120B_CUTOFF, Question, parse_iso_date, write_jsonl

DEFAULT_GAMMA = "https://gamma-api.polymarket.com"
DEFAULT_PAGE_SIZE = 500
DEFAULT_TIMEOUT = 30
DEFAULT_THROTTLE = 0.2
MIN_VOLUME = 1000.0  # USD-equivalent; drops empty markets


def _http_get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> Any:
    try:
        import requests  # type: ignore

        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "InvariantForecasting/0.2 (research)"},
        )
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        import urllib.request

        req = urllib.request.Request(
            url, headers={"User-Agent": "InvariantForecasting/0.2 (research)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def fetch_page(
    base: str,
    *,
    offset: int,
    limit: int,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    params = {
        "limit": limit,
        "offset": offset,
        "closed": "true",
        "active": "false",
        "order": "endDate",
        "ascending": "false",
    }
    url = f"{base}/markets?{urlencode(params)}"
    if cache_dir is not None:
        key = f"closed_off{offset}_lim{limit}.json"
        cache_path = cache_dir / key
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _http_get_json(url)
    if not isinstance(payload, list):
        # Some Gamma responses wrap in {"markets": [...]} — defensive parse.
        payload = payload.get("markets", []) if isinstance(payload, dict) else []
    if cache_dir is not None:
        cache_path = cache_dir / key
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def iter_markets(
    base: str = DEFAULT_GAMMA,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
    throttle_sec: float = DEFAULT_THROTTLE,
    cache_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    offset = 0
    page_idx = 0
    while True:
        page = fetch_page(base, offset=offset, limit=page_size, cache_dir=cache_dir)
        if not page:
            return
        for m in page:
            yield m
        offset += len(page)
        page_idx += 1
        if max_pages is not None and page_idx >= max_pages:
            return
        if len(page) < page_size:
            return
        time.sleep(throttle_sec)


def _parse_outcome_prices(value: Any) -> tuple[float, float] | None:
    """Polymarket returns outcomePrices as a JSON-encoded string ``'["1.0","0.0"]'``.

    For a closed binary market, exactly one element should be 1.0.
    """

    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        a, b = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return a, b


def _is_yes_no_market(market: dict[str, Any]) -> bool:
    """Binary Yes/No markets have outcomes ``["Yes","No"]`` or ``["No","Yes"]``."""

    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except json.JSONDecodeError:
            return False
    if not isinstance(outcomes, (list, tuple)) or len(outcomes) != 2:
        return False
    lowered = {str(o).strip().lower() for o in outcomes}
    return lowered == {"yes", "no"}


def market_to_question(
    market: dict[str, Any],
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    min_volume: float = MIN_VOLUME,
) -> Question | None:
    if not market.get("closed"):
        return None
    if market.get("active"):
        return None
    if not _is_yes_no_market(market):
        return None

    prices = _parse_outcome_prices(market.get("outcomePrices"))
    if prices is None:
        return None
    yes_price, no_price = prices
    if yes_price not in (0.0, 1.0) or no_price not in (0.0, 1.0):
        return None  # ambiguous resolution; not a clean binary
    if yes_price == no_price:
        return None
    outcome = 1 if yes_price == 1.0 else 0

    try:
        volume = float(market.get("volumeNum") or market.get("volume") or 0)
    except (TypeError, ValueError):
        volume = 0.0
    if volume < min_volume:
        return None

    # Polymarket doesn't have a separate 'freeze' concept; the close (endDate)
    # is when trading ends, which is the natural commit time. Resolution is
    # umaEndDate when present, else endDate.
    freeze_date = parse_iso_date(market.get("endDate"))
    resolved_at = parse_iso_date(
        market.get("umaEndDate") or market.get("endDate")
    )
    if freeze_date is None or resolved_at is None:
        return None
    if freeze_date > resolved_at:
        freeze_date, resolved_at = resolved_at, resolved_at  # collapse degenerate
    if freeze_date <= cutoff_date or resolved_at <= cutoff_date:
        return None

    title = (market.get("question") or market.get("title") or "").strip()
    if not title:
        return None

    description = (market.get("description") or "").strip() or None

    slug = market.get("slug")
    url = f"https://polymarket.com/event/{slug}" if slug else None
    native = str(market.get("conditionId") or market.get("id") or slug)
    if not native or native == "None":
        return None
    unified_id = f"polymarket::{native}"

    categories: list[str] = []
    for tag in market.get("tags") or []:
        if isinstance(tag, dict):
            label = tag.get("label") or tag.get("name")
        else:
            label = str(tag)
        if label:
            categories.append(label)

    try:
        return Question(
            id=unified_id,
            question=title,
            outcome=outcome,
            freeze_date=freeze_date,
            resolved_at=resolved_at,
            source="polymarket",
            url=url,
            background=description,
            resolution_criteria=description,  # PM bundles them together
            categories=categories,
            raw={
                "polymarket_condition_id": market.get("conditionId"),
                "polymarket_id": market.get("id"),
                "slug": slug,
                "endDate": market.get("endDate"),
                "umaEndDate": market.get("umaEndDate"),
                "volumeNum": market.get("volumeNum"),
                "outcomePrices": market.get("outcomePrices"),
                "outcomes": market.get("outcomes"),
            },
        )
    except ValueError:
        return None


def pull(
    base: str = DEFAULT_GAMMA,
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    min_volume: float = MIN_VOLUME,
    max_pages: int | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[Question], Counter]:
    rows: list[Question] = []
    skipped: Counter = Counter()
    for market in iter_markets(base, max_pages=max_pages, cache_dir=cache_dir):
        q = market_to_question(market, cutoff_date=cutoff_date, min_volume=min_volume)
        if q is None:
            skipped["filtered"] += 1
            continue
        rows.append(q)
    rows.sort(key=lambda r: (r.resolved_at, r.id))
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull resolved binary Polymarket markets via the Gamma API."
    )
    parser.add_argument("--api-base", default=DEFAULT_GAMMA)
    parser.add_argument("--cutoff", default=GPT_OSS_120B_CUTOFF)
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--cache-dir", default="data/raw/polymarket_cache")
    parser.add_argument("--output", default="data/raw/polymarket.jsonl")
    args = parser.parse_args()

    rows, skipped = pull(
        base=args.api_base,
        cutoff_date=args.cutoff,
        min_volume=args.min_volume,
        max_pages=args.max_pages,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    n = write_jsonl(rows, args.output)
    print(f"polymarket: wrote {n} questions to {args.output}", file=sys.stderr)
    print(f"date range: {rows[0].freeze_date} to {rows[-1].resolved_at}" if rows else "no rows",
          file=sys.stderr)
    print(f"skipped: {dict(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
