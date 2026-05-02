"""Merge per-source JSONLs into a single unified question pool.

Cross-source dedupe strategy:
  1. **URL match.** If two rows have the same canonical URL (after stripping
     trailing slashes and query strings), keep one. This catches the same
     Polymarket market appearing in both ForecastBench and Polymarket pulls.
  2. **Fuzzy text match within ±14 days of each other on resolution date.**
     Normalize text (lowercase, strip punctuation, collapse whitespace), then
     use a token-Jaccard similarity threshold. Catches cross-platform
     duplicates (a Polymarket market mirrored on Manifold).

Source priority (kept) on duplicates: ``mantic_q2_aib`` > ``metaculus`` >
``polymarket`` > ``manifold`` > ``forecastbench``. Rationale: real-platform
sources have richer metadata than ForecastBench's curated copies; the AIB
test-set tag should never be lost.

The audit report includes: source counts, post-merge count, dedupe count,
yes-rate per source, freeze/resolution date histograms, and category
distribution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from .schema import Question, read_jsonl, write_jsonl


SOURCE_PRIORITY = {
    "mantic_q2_aib": 0,
    "metaculus": 1,
    "polymarket": 2,
    "manifold": 3,
    "forecastbench": 4,
}

URL_QUERY_STRIP = re.compile(r"\?.*$")
URL_TRAIL_SLASH = re.compile(r"/+$")


def canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip().lower()
    u = URL_QUERY_STRIP.sub("", u)
    u = URL_TRAIL_SLASH.sub("", u)
    if u.startswith("http://"):
        u = "https://" + u[len("http://") :]
    return u or None


_PUNCT = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    t = text.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return t


def token_jaccard(a: str, b: str, *, min_len: int = 3) -> float:
    sa = {t for t in a.split() if len(t) >= min_len}
    sb = {t for t in b.split() if len(t) >= min_len}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _date_to_dt(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _better(a: Question, b: Question) -> Question:
    """Pick the higher-priority of two duplicates."""

    pa = SOURCE_PRIORITY.get(a.source, 99)
    pb = SOURCE_PRIORITY.get(b.source, 99)
    if pa != pb:
        return a if pa < pb else b
    # Same source priority -> prefer the one with more metadata
    score_a = (len(a.background or ""), len(a.resolution_criteria or ""), len(a.categories))
    score_b = (len(b.background or ""), len(b.resolution_criteria or ""), len(b.categories))
    return a if score_a >= score_b else b


def deduplicate(
    rows: list[Question],
    *,
    text_jaccard_threshold: float = 0.7,
    date_window_days: int = 14,
) -> tuple[list[Question], dict[str, int]]:
    """Return (kept rows, stats). Order-stable on the kept rows."""

    # Pass 1: URL dedupe — but only ACROSS sources. Within a source, the same
    # URL can legitimately appear on many distinct questions (e.g. different
    # forecast horizons of the same yfinance/Wikipedia entity). The
    # source-native id is the within-source uniqueness contract.
    #
    # Strategy: group rows by canonical URL. For each URL group, partition by
    # source. If only one source touches the URL, keep every row from that
    # source (no dedupe). If multiple sources touch the URL, keep the highest-
    # priority source's rows and drop the others.
    by_url_group: dict[str | None, list[Question]] = defaultdict(list)
    for r in rows:
        by_url_group[canonical_url(r.url)].append(r)

    pass1: list[Question] = []
    url_dups = 0
    for cu, group in by_url_group.items():
        if cu is None:
            pass1.extend(group)
            continue
        sources_in_group = {r.source for r in group}
        if len(sources_in_group) == 1:
            pass1.extend(group)
            continue
        # Cross-source URL collision: keep highest-priority source's rows.
        best_src = min(sources_in_group, key=lambda s: SOURCE_PRIORITY.get(s, 99))
        kept_from_url = [r for r in group if r.source == best_src]
        pass1.extend(kept_from_url)
        url_dups += len(group) - len(kept_from_url)

    # Pass 2: fuzzy text dedupe within a ±date_window_days resolution-date window.
    # We bucket by month for efficiency, then compare within neighbouring buckets.
    norm: dict[str, str] = {r.id: normalize_text(r.question) for r in pass1}
    res_dt: dict[str, datetime] = {r.id: _date_to_dt(r.resolved_at) for r in pass1}
    by_month: dict[str, list[Question]] = defaultdict(list)
    for r in pass1:
        by_month[r.resolved_at[:7]].append(r)

    dropped: set[str] = set()
    text_dups = 0

    months = sorted(by_month)
    for i, mk in enumerate(months):
        # Compare within month and the next month (covers the 14-day window across boundaries).
        cohort = list(by_month[mk])
        if i + 1 < len(months):
            cohort = cohort + by_month[months[i + 1]]
        for j in range(len(cohort)):
            a = cohort[j]
            if a.id in dropped:
                continue
            for k in range(j + 1, len(cohort)):
                b = cohort[k]
                if b.id in dropped:
                    continue
                if a.source == b.source:
                    continue  # same source already de-duped by id
                if abs((res_dt[a.id] - res_dt[b.id]).days) > date_window_days:
                    continue
                sim = token_jaccard(norm[a.id], norm[b.id])
                if sim < text_jaccard_threshold:
                    continue
                winner = _better(a, b)
                loser = b if winner.id == a.id else a
                dropped.add(loser.id)
                text_dups += 1

    kept = [r for r in pass1 if r.id not in dropped]
    kept.sort(key=lambda r: (r.resolved_at, r.id))
    return kept, {
        "input": len(rows),
        "url_duplicates_dropped": url_dups,
        "text_duplicates_dropped": text_dups,
        "kept": len(kept),
    }


def audit_report(rows: list[Question]) -> dict[str, object]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    yes_rate = sum(r.outcome for r in rows) / n
    by_source = Counter(r.source for r in rows)
    yes_by_source: dict[str, float] = {}
    for src in by_source:
        sub = [r for r in rows if r.source == src]
        yes_by_source[src] = round(sum(r.outcome for r in sub) / len(sub), 3)
    months = Counter(r.resolved_at[:7] for r in rows)
    cats = Counter()
    for r in rows:
        cats.update(r.categories)
    return {
        "n": n,
        "yes_rate": round(yes_rate, 3),
        "by_source": dict(by_source.most_common()),
        "yes_rate_by_source": yes_by_source,
        "freeze_range": (rows[0].freeze_date, rows[-1].freeze_date),
        "resolved_range": (rows[0].resolved_at, rows[-1].resolved_at),
        "resolved_months": dict(sorted(months.items())),
        "top_categories": dict(cats.most_common(20)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge per-source JSONLs into one deduplicated unified pool."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Per-source JSONL paths (forecastbench.jsonl, metaculus.jsonl, ...).",
    )
    parser.add_argument("--output", default="data/processed/unified.jsonl")
    parser.add_argument("--audit-output", default="data/processed/unified_audit.json")
    parser.add_argument("--text-jaccard-threshold", type=float, default=0.7)
    parser.add_argument("--date-window-days", type=int, default=14)
    args = parser.parse_args()

    all_rows: list[Question] = []
    for path in args.inputs:
        try:
            rows = read_jsonl(path)
        except FileNotFoundError:
            print(f"warning: {path} not found, skipping", file=sys.stderr)
            continue
        all_rows.extend(rows)
        print(f"  + {len(rows)} from {path}", file=sys.stderr)

    kept, dedupe_stats = deduplicate(
        all_rows,
        text_jaccard_threshold=args.text_jaccard_threshold,
        date_window_days=args.date_window_days,
    )
    n_written = write_jsonl(kept, args.output)
    audit = {"dedupe": dedupe_stats, **audit_report(kept)}
    Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_output).write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"unified: wrote {n_written} rows to {args.output}", file=sys.stderr)
    print(f"audit:   wrote summary to {args.audit_output}", file=sys.stderr)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
