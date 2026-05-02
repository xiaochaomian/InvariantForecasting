"""Manifold Markets puller.

Uses the public ``https://api.manifold.markets/v0/markets`` endpoint, which
paginates with ``before=<lastId>`` and returns up to 1000 markets per page.

Filters:
  - binary markets (``outcomeType == "BINARY"``)
  - resolved (``isResolved == true``) with a non-cancelled resolution
    (``resolution`` ∈ {"YES","NO"}; "MKT" / "CANCEL" are not clean labels)
  - liquidity / volume floor (``volume >= MIN_VOLUME``) — drops vanity markets
  - post-cutoff freeze and resolution dates

Manifold has lots of low-engagement / joke markets; the volume filter is the
main quality signal we have. The default 100 floor is intentionally
conservative; many real markets clear ~M$10–100k in volume.
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

DEFAULT_API = "https://api.manifold.markets/v0"
DEFAULT_PAGE_SIZE = 1000
DEFAULT_TIMEOUT = 30
DEFAULT_THROTTLE = 0.2
MIN_VOLUME = 100.0


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
    limit: int,
    before: str | None,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit}
    if before:
        params["before"] = before
    url = f"{base}/markets?{urlencode(params)}"
    if cache_dir is not None:
        key = f"markets_lim{limit}_before{before or 'head'}.json"
        cache_path = cache_dir / key
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _http_get_json(url)
    if not isinstance(payload, list):
        payload = []
    if cache_dir is not None:
        cache_path = cache_dir / key
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def iter_markets(
    base: str = DEFAULT_API,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
    throttle_sec: float = DEFAULT_THROTTLE,
    cache_dir: Path | None = None,
) -> Iterator[dict[str, Any]]:
    before: str | None = None
    page_idx = 0
    while True:
        page = fetch_page(base, limit=page_size, before=before, cache_dir=cache_dir)
        if not page:
            return
        for m in page:
            yield m
        before = page[-1].get("id")
        page_idx += 1
        if max_pages is not None and page_idx >= max_pages:
            return
        if len(page) < page_size:
            return
        time.sleep(throttle_sec)


def market_to_question(
    market: dict[str, Any],
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    min_volume: float = MIN_VOLUME,
) -> Question | None:
    if market.get("outcomeType") != "BINARY":
        return None
    if not market.get("isResolved"):
        return None

    resolution = market.get("resolution")
    if resolution == "YES":
        outcome = 1
    elif resolution == "NO":
        outcome = 0
    else:
        return None  # MKT / CANCEL / probability-resolution; not a clean label

    try:
        volume = float(market.get("volume") or 0)
    except (TypeError, ValueError):
        volume = 0.0
    if volume < min_volume:
        return None

    # Manifold close-time is when forecasting ends; resolutionTime is when it
    # was resolved by the creator. Both are unix-ms.
    freeze_date = parse_iso_date(market.get("closeTime"))
    resolved_at = parse_iso_date(
        market.get("resolutionTime") or market.get("closeTime")
    )
    if freeze_date is None or resolved_at is None:
        return None
    if freeze_date > resolved_at:
        freeze_date = resolved_at
    if freeze_date <= cutoff_date or resolved_at <= cutoff_date:
        return None

    title = (market.get("question") or "").strip()
    if not title:
        return None

    description: str | None = None
    raw_desc = market.get("textDescription") or market.get("description")
    if isinstance(raw_desc, str) and raw_desc.strip():
        description = raw_desc.strip()
    elif isinstance(raw_desc, dict):
        # Manifold sometimes returns rich-text JSON; flatten to plain text.
        description = _flatten_richtext(raw_desc).strip() or None

    slug = market.get("slug")
    creator = market.get("creatorUsername")
    url = (
        f"https://manifold.markets/{creator}/{slug}"
        if (creator and slug)
        else market.get("url")
    )
    native = str(market.get("id") or slug)
    if not native or native == "None":
        return None
    unified_id = f"manifold::{native}"

    categories: list[str] = []
    for tag in market.get("groupSlugs") or []:
        if isinstance(tag, str):
            categories.append(tag)

    try:
        return Question(
            id=unified_id,
            question=title,
            outcome=outcome,
            freeze_date=freeze_date,
            resolved_at=resolved_at,
            source="manifold",
            url=url,
            background=description,
            resolution_criteria=description,
            categories=categories,
            raw={
                "manifold_id": market.get("id"),
                "slug": slug,
                "creatorUsername": creator,
                "closeTime": market.get("closeTime"),
                "resolutionTime": market.get("resolutionTime"),
                "volume": market.get("volume"),
                "uniqueBettorCount": market.get("uniqueBettorCount"),
                "totalLiquidity": market.get("totalLiquidity"),
            },
        )
    except ValueError:
        return None


def _flatten_richtext(node: Any) -> str:
    """Best-effort flatten of TipTap-style rich-text to plain text."""

    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return " ".join(_flatten_richtext(n) for n in node)
    if isinstance(node, dict):
        chunks: list[str] = []
        if isinstance(node.get("text"), str):
            chunks.append(node["text"])
        for k in ("content", "children"):
            if k in node:
                chunks.append(_flatten_richtext(node[k]))
        return " ".join(c for c in chunks if c)
    return ""


def pull(
    base: str = DEFAULT_API,
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
        description="Pull resolved binary markets from Manifold's public API."
    )
    parser.add_argument("--api-base", default=DEFAULT_API)
    parser.add_argument("--cutoff", default=GPT_OSS_120B_CUTOFF)
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--cache-dir", default="data/raw/manifold_cache")
    parser.add_argument("--output", default="data/raw/manifold.jsonl")
    args = parser.parse_args()

    rows, skipped = pull(
        base=args.api_base,
        cutoff_date=args.cutoff,
        min_volume=args.min_volume,
        max_pages=args.max_pages,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    n = write_jsonl(rows, args.output)
    print(f"manifold: wrote {n} questions to {args.output}", file=sys.stderr)
    print(f"date range: {rows[0].freeze_date} to {rows[-1].resolved_at}" if rows else "no rows",
          file=sys.stderr)
    print(f"skipped: {dict(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
