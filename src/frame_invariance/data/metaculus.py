"""Metaculus public-API puller.

Hits the Metaculus REST API (``/api/posts/``) for resolved binary questions and
emits unified ``Question`` rows. Includes a knob for restricting to a specific
tournament (e.g. the Q2 2025 AI Forecasting Benchmark, slug
``aibq2``) so we can carve out a ``mantic_q2_aib`` test split for direct
comparison with Mantic's reported numbers.

Run on a machine with internet; this sandbox can't reach metaculus.com. Cached
JSON is written to ``data/raw/metaculus_cache/`` so re-runs are cheap.

API reference: https://www.metaculus.com/api/  (the v2 endpoints; we use
``/api/posts/`` rather than the older ``/api2/questions/``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

from .schema import GPT_OSS_120B_CUTOFF, Question, parse_iso_date, write_jsonl

DEFAULT_API_BASE = "https://www.metaculus.com/api"
DEFAULT_PAGE_SIZE = 100
DEFAULT_TIMEOUT = 30
DEFAULT_THROTTLE_SEC = 0.25  # be polite; their rate limits are real

# Metaculus's public API gates `/api/posts/` behind an API token. Get yours at
# https://www.metaculus.com/accounts/profile/ (you must be signed in). Export it
# as ``METACULUS_TOKEN`` or pass --token. As of 2026, anonymous requests get
# 403 Forbidden.
ENV_TOKEN = "METACULUS_TOKEN"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 InvariantForecasting/0.2"
)


def _build_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def _http_get_json(
    url: str, *, timeout: int = DEFAULT_TIMEOUT, token: str | None = None
) -> Any:
    """Fetch JSON. Uses ``requests`` if installed; falls back to ``urllib``."""

    headers = _build_headers(token)
    try:
        import requests  # type: ignore

        resp = requests.get(url, timeout=timeout, headers=headers)
        if resp.status_code in (401, 403):
            hint = (
                "Metaculus requires an API token. Sign in, copy your token from "
                "https://www.metaculus.com/accounts/profile/, then export "
                f"{ENV_TOKEN}=<token> or pass --token."
            )
            raise PermissionError(f"{resp.status_code} on {url}\n{hint}")
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        import urllib.request

        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def fetch_posts_page(
    base: str,
    *,
    page_size: int,
    offset: int,
    statuses: tuple[str, ...] = ("resolved",),
    forecast_type: str = "binary",
    tournaments: str | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch one page of /api/posts/. Caches the raw response by query."""

    params: dict[str, Any] = {
        "limit": page_size,
        "offset": offset,
        "forecast_type": forecast_type,
        "order_by": "-resolution_set_time",
    }
    if statuses:
        params["statuses"] = ",".join(statuses)
    if tournaments:
        params["tournaments"] = tournaments
    url = f"{base}/posts/?{urlencode(params)}"

    if cache_dir is not None:
        cache_key = f"posts_off{offset}_lim{page_size}_t{tournaments or 'all'}.json"
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        cache_dir.mkdir(parents=True, exist_ok=True)

    payload = _http_get_json(url, token=token)

    if cache_dir is not None:
        cache_path = cache_dir / cache_key
        cache_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def iter_posts(
    base: str = DEFAULT_API_BASE,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    tournaments: str | None = None,
    max_pages: int | None = None,
    throttle_sec: float = DEFAULT_THROTTLE_SEC,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield individual post dicts across pagination."""

    offset = 0
    page_idx = 0
    while True:
        payload = fetch_posts_page(
            base,
            page_size=page_size,
            offset=offset,
            tournaments=tournaments,
            cache_dir=cache_dir,
            token=token,
        )
        results = payload.get("results", [])
        if not results:
            return
        for post in results:
            yield post
        offset += len(results)
        page_idx += 1
        if max_pages is not None and page_idx >= max_pages:
            return
        if not payload.get("next"):
            return
        time.sleep(throttle_sec)


def post_to_question(
    post: dict[str, Any],
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    tournament_label: str | None = None,
) -> Question | None:
    """Convert one Metaculus post to a unified ``Question``, or None on reject."""

    question_block = post.get("question") or {}
    if question_block.get("type") != "binary":
        return None
    if question_block.get("status") != "resolved":
        # Metaculus marks ambiguous/annulled with their own statuses.
        return None
    resolution_value = question_block.get("resolution")
    outcome: int | None
    if resolution_value in ("yes", True, 1, "1"):
        outcome = 1
    elif resolution_value in ("no", False, 0, "0"):
        outcome = 0
    else:
        return None  # ambiguous / annulled / numeric drift

    # Freeze date: Metaculus's "scheduled_close_time" is when forecasting ends.
    # Resolution: "actual_resolve_time".
    freeze_date = parse_iso_date(
        question_block.get("scheduled_close_time")
        or question_block.get("close_time")
        or post.get("scheduled_close_time")
    )
    resolved_at = parse_iso_date(
        question_block.get("actual_resolve_time")
        or question_block.get("resolution_set_time")
        or post.get("resolution_set_time")
    )
    if freeze_date is None or resolved_at is None:
        return None
    if freeze_date > resolved_at:
        return None
    if freeze_date <= cutoff_date or resolved_at <= cutoff_date:
        return None

    title = (
        question_block.get("title")
        or post.get("title")
        or ""
    )
    if not title.strip():
        return None

    background = (
        question_block.get("description")
        or post.get("description")
        or None
    )
    resolution_criteria = question_block.get("resolution_criteria") or None
    fine_print = question_block.get("fine_print")
    if resolution_criteria and fine_print:
        resolution_criteria = f"{resolution_criteria}\n\nFine print: {fine_print}"
    elif fine_print and not resolution_criteria:
        resolution_criteria = f"Fine print: {fine_print}"

    pid = post.get("id")
    qid = question_block.get("id")
    native = f"{pid}_{qid}" if (pid and qid) else str(pid or qid)
    if not native or native == "None":
        return None
    unified_id = f"metaculus::{native}"

    categories: list[str] = []
    for proj in (post.get("projects") or {}).get("category", []) or []:
        name = proj.get("name") if isinstance(proj, dict) else None
        if name:
            categories.append(name)
    if tournament_label:
        categories.append(f"tournament:{tournament_label}")

    url = (
        f"https://www.metaculus.com/questions/{pid}/"
        if pid
        else None
    )

    try:
        return Question(
            id=unified_id,
            question=title,
            outcome=outcome,
            freeze_date=freeze_date,
            resolved_at=resolved_at,
            source="metaculus",
            url=url,
            background=background,
            resolution_criteria=resolution_criteria,
            categories=categories,
            raw={
                "metaculus_post_id": pid,
                "metaculus_question_id": qid,
                "tournament_label": tournament_label,
                "open_time": question_block.get("open_time"),
                "scheduled_close_time": question_block.get("scheduled_close_time"),
                "actual_resolve_time": question_block.get("actual_resolve_time"),
                "resolution_set_time": question_block.get("resolution_set_time"),
            },
        )
    except ValueError:
        return None


def pull(
    base: str = DEFAULT_API_BASE,
    *,
    cutoff_date: str = GPT_OSS_120B_CUTOFF,
    tournaments: str | None = None,
    tournament_label: str | None = None,
    max_pages: int | None = None,
    cache_dir: Path | None = None,
    token: str | None = None,
) -> tuple[list[Question], Counter]:
    rows: list[Question] = []
    skipped: Counter = Counter()
    for post in iter_posts(
        base,
        tournaments=tournaments,
        max_pages=max_pages,
        cache_dir=cache_dir,
        token=token,
    ):
        q = post_to_question(post, cutoff_date=cutoff_date, tournament_label=tournament_label)
        if q is None:
            skipped["filtered"] += 1
            continue
        rows.append(q)
    rows.sort(key=lambda r: (r.resolved_at, r.id))
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull resolved binary questions from the Metaculus public API."
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--cutoff", default=GPT_OSS_120B_CUTOFF)
    parser.add_argument(
        "--tournaments",
        default=None,
        help="Comma-separated tournament slugs (e.g. 'aibq2'). Default: all.",
    )
    parser.add_argument(
        "--tournament-label",
        default=None,
        help="Tag added to categories (e.g. 'mantic_q2_aib') for downstream filtering.",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        default="data/raw/metaculus_cache",
        help="Cache fetched API pages here so re-runs are free.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=f"Metaculus API token; falls back to ${ENV_TOKEN} env var.",
    )
    parser.add_argument("--output", default="data/raw/metaculus.jsonl")
    args = parser.parse_args()

    token = args.token or os.environ.get(ENV_TOKEN)
    if not token:
        print(
            f"warning: no Metaculus token (set ${ENV_TOKEN} or pass --token). "
            "Anonymous /api/posts/ requests are 403'd by Metaculus.",
            file=sys.stderr,
        )

    rows, skipped = pull(
        base=args.api_base,
        cutoff_date=args.cutoff,
        tournaments=args.tournaments,
        tournament_label=args.tournament_label,
        max_pages=args.max_pages,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        token=token,
    )
    n = write_jsonl(rows, args.output)
    print(f"metaculus: wrote {n} questions to {args.output}", file=sys.stderr)
    print(f"date range: {rows[0].freeze_date} to {rows[-1].resolved_at}" if rows else "no rows",
          file=sys.stderr)
    print(f"skipped: {dict(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
