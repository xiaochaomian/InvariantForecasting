"""Live status: progress, throughput, ETA, and cost projection for Day-2 runs.

Reads three sources:

  1. ``data/processed/contexts.jsonl`` — one line per completed context call.
  2. ``data/processed/paraphrases.jsonl`` — one line per completed paraphrase call.
  3. ``data/cache/claude/*.json`` — every Claude response we've ever cached
     (includes ``usage.input_tokens`` and ``usage.output_tokens``).

Outputs:

  - Progress vs. expected total (default 3,471 questions).
  - Throughput in the last 1 / 5 / 15 minutes (windowed by cache-file mtime).
  - ETA for each script, computed at the windowed rate.
  - Spend so far, computed from cached usage at Sonnet 4.6 list pricing.
  - Projected total spend (linear extrapolation by question).

Run ad-hoc, or wrap in a shell loop for live monitoring:

    while true; do clear; python scripts/estimate_progress.py; sleep 30; done
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Sonnet 4.6 list pricing per Anthropic docs as of May 2026. Override via flags
# if the prices change or you want to model a different SKU.
DEFAULT_INPUT_PRICE_PER_M = 3.0
DEFAULT_OUTPUT_PRICE_PER_M = 15.0
DEFAULT_TOTAL_QUESTIONS = 3471


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                n += 1
    return n


def classify_cache_text(text: str) -> str:
    """Cheap classification of a cached response: 'context', 'paraphrase', or '?'.

    Context responses are JSON objects with a ``base_rate`` key; paraphrase
    responses are JSON lists of strings. We sniff the leading non-whitespace
    char and (for objects) look for ``base_rate``.
    """

    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    if s.startswith("["):
        return "paraphrase"
    if s.startswith("{"):
        return "context" if "base_rate" in s[:300] else "?"
    return "?"


def walk_cache(cache_dir: Path) -> tuple[Counter, Counter, list[float]]:
    """Return (tokens_by_kind, calls_by_kind, mtimes) over every cache file.

    tokens_by_kind: keys ``input_<kind>`` and ``output_<kind>`` for kind in
    {context, paraphrase, ?}.
    calls_by_kind: kind -> count.
    mtimes: float-seconds since epoch for every cache file (used for windowed
    throughput).
    """

    tokens: Counter = Counter()
    calls: Counter = Counter()
    mtimes: list[float] = []
    if not cache_dir.exists():
        return tokens, calls, mtimes

    for path in cache_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        usage = payload.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        kind = classify_cache_text(payload.get("text", ""))
        calls[kind] += 1
        tokens[f"input_{kind}"] += in_tok
        tokens[f"output_{kind}"] += out_tok
        mtimes.append(path.stat().st_mtime)
    return tokens, calls, mtimes


def windowed_rate(mtimes: list[float], window_s: float) -> float:
    """Calls per minute over the last ``window_s`` seconds of cache writes."""

    if not mtimes:
        return 0.0
    now = time.time()
    n = sum(1 for m in mtimes if now - m <= window_s)
    return n / (window_s / 60.0)


def fmt_eta(remaining: int, rate_per_min: float) -> str:
    if rate_per_min <= 0:
        return "—"
    minutes = remaining / rate_per_min
    if minutes < 60:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.2f} h ({minutes:.0f} min)"


def fmt_cost(in_tokens: int, out_tokens: int, in_price: float, out_price: float) -> str:
    cost = (in_tokens / 1e6) * in_price + (out_tokens / 1e6) * out_price
    return f"${cost:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate progress + cost + ETA.")
    parser.add_argument("--contexts", default="data/processed/contexts.jsonl")
    parser.add_argument("--paraphrases", default="data/processed/paraphrases.jsonl")
    parser.add_argument("--cache-dir", default="data/cache/claude")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_QUESTIONS,
                        help="Expected questions per script.")
    parser.add_argument("--input-price", type=float, default=DEFAULT_INPUT_PRICE_PER_M,
                        help="USD per 1M input tokens.")
    parser.add_argument("--output-price", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M,
                        help="USD per 1M output tokens.")
    args = parser.parse_args()

    contexts_path = Path(args.contexts)
    paraphrases_path = Path(args.paraphrases)
    cache_dir = Path(args.cache_dir)
    total = args.total

    n_ctx = count_lines(contexts_path)
    n_par = count_lines(paraphrases_path)
    tokens, calls, mtimes = walk_cache(cache_dir)

    now = datetime.now(timezone.utc).astimezone()
    print(f"=== {now.strftime('%Y-%m-%d %H:%M:%S %Z')} ===\n")

    # Progress
    print("Progress:")
    print(f"  contexts.jsonl:    {n_ctx:5d} / {total} ({100 * n_ctx / total:5.1f}%)")
    print(f"  paraphrases.jsonl: {n_par:5d} / {total} ({100 * n_par / total:5.1f}%)")
    print()

    # Throughput windows (combined cache, both scripts share)
    r1 = windowed_rate(mtimes, 60)
    r5 = windowed_rate(mtimes, 5 * 60)
    r15 = windowed_rate(mtimes, 15 * 60)
    print("Throughput (combined Claude calls/min):")
    print(f"  last  1 min: {r1:6.1f}")
    print(f"  last  5 min: {r5:6.1f}")
    print(f"  last 15 min: {r15:6.1f}")
    print()

    # Per-script rate inferred by classification ratio
    ctx_calls = calls.get("context", 0)
    par_calls = calls.get("paraphrase", 0)
    total_known = max(ctx_calls + par_calls, 1)
    ctx_share = ctx_calls / total_known
    par_share = par_calls / total_known
    rate_window = r5 if r5 > 0 else r1
    rate_ctx = rate_window * ctx_share
    rate_par = rate_window * par_share
    print("Per-script rate (calls/min, inferred from cache mix):")
    print(f"  context:    {rate_ctx:6.1f}")
    print(f"  paraphrase: {rate_par:6.1f}")
    print()

    # ETAs
    ctx_remaining = max(total - n_ctx, 0)
    par_remaining = max(total - n_par, 0)
    print("ETA at last-5-min rate:")
    print(f"  context complete:    {fmt_eta(ctx_remaining, rate_ctx)}")
    print(f"  paraphrase complete: {fmt_eta(par_remaining, rate_par)}")
    if rate_ctx > 0 and rate_par > 0:
        finish_min = max(ctx_remaining / rate_ctx, par_remaining / rate_par)
        eta_dt = datetime.now() + timedelta(minutes=finish_min)
        print(f"  both done by:        {eta_dt.strftime('%H:%M:%S')} "
              f"(~{finish_min:.0f} min from now)")
    print()

    # Spend so far
    in_ctx = tokens.get("input_context", 0)
    out_ctx = tokens.get("output_context", 0)
    in_par = tokens.get("input_paraphrase", 0)
    out_par = tokens.get("output_paraphrase", 0)
    in_unk = tokens.get("input_?", 0)
    out_unk = tokens.get("output_?", 0)
    in_total = in_ctx + in_par + in_unk
    out_total = out_ctx + out_par + out_unk
    print("Tokens used (from Claude cache):")
    print(f"  context     in/out: {in_ctx:>10,} / {out_ctx:>10,}  "
          f"({fmt_cost(in_ctx, out_ctx, args.input_price, args.output_price)})")
    print(f"  paraphrase  in/out: {in_par:>10,} / {out_par:>10,}  "
          f"({fmt_cost(in_par, out_par, args.input_price, args.output_price)})")
    if in_unk or out_unk:
        print(f"  unclassified in/out:{in_unk:>10,} / {out_unk:>10,}  "
              f"({fmt_cost(in_unk, out_unk, args.input_price, args.output_price)})")
    print(f"  TOTAL       in/out: {in_total:>10,} / {out_total:>10,}  "
          f"({fmt_cost(in_total, out_total, args.input_price, args.output_price)})")
    print()

    # Per-call averages and projection
    print("Projected total cost (linear extrapolation):")
    if ctx_calls > 0:
        ctx_avg_in = in_ctx / ctx_calls
        ctx_avg_out = out_ctx / ctx_calls
        ctx_proj_in = ctx_avg_in * total
        ctx_proj_out = ctx_avg_out * total
        print(f"  context     × {total:4d} q: ~{fmt_cost(ctx_proj_in, ctx_proj_out, args.input_price, args.output_price)}"
              f"   (avg ~{ctx_avg_in:.0f} in / {ctx_avg_out:.0f} out per call)")
    if par_calls > 0:
        par_avg_in = in_par / par_calls
        par_avg_out = out_par / par_calls
        # Paraphrase has retries: actual calls per question can be > 1.
        par_calls_per_q = par_calls / max(n_par, 1) if n_par > 0 else 1.0
        par_proj_in = par_avg_in * total * par_calls_per_q
        par_proj_out = par_avg_out * total * par_calls_per_q
        print(f"  paraphrase  × {total:4d} q: ~{fmt_cost(par_proj_in, par_proj_out, args.input_price, args.output_price)}"
              f"   (avg ~{par_avg_in:.0f} in / {par_avg_out:.0f} out per call, ~{par_calls_per_q:.2f} calls/q)")
    if ctx_calls > 0 and par_calls > 0:
        proj_total = (
            (ctx_avg_in * total + par_avg_in * total * par_calls_per_q) / 1e6 * args.input_price
            + (ctx_avg_out * total + par_avg_out * total * par_calls_per_q) / 1e6 * args.output_price
        )
        print(f"  TOTAL projected:        ~${proj_total:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
