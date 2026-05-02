"""LLM-generated forecast context: base rate + dated news snapshot.

For each ``Question`` in the unified pool, we ask Claude Sonnet to output two
artifacts that a forecaster at the question's ``freeze_date`` would plausibly
have access to:

  - ``base_rate``: a reference-class probability with sample-count and one-line
    explanation. Format: ``{value, n_reference_events, reference_window_years,
    explanation}``.
  - ``news_snapshot``: a list of dated bullet points (≤ 8) summarizing relevant
    information available *strictly before* ``freeze_date``. Format:
    ``[{date, headline, summary}, ...]``.

The same context is used for every paraphrase of a question, which is what
isolates question-frame variance from research variance (cf. Mantic's deep-
research approach which produces *non*-paraphrase-stable contexts).

Leakage controls:
  - System prompt forbids referencing the actual resolution.
  - Output is parsed strictly as JSON; any news item dated ≥ ``freeze_date``
    is *dropped* in post-processing (the prompt also forbids it, but we
    enforce defense-in-depth).
  - The Claude knowledge cutoff is later than gpt-oss-120b's. We mitigate by
    instructing Claude to imagine itself as a forecaster at the freeze date,
    but we do *not* claim this fully eliminates Claude's hindsight bias —
    that's a well-known limitation we'll discuss in the paper.

The CLI processes one ``unified.jsonl`` and writes one ``contexts.jsonl`` with
one row per question id. Re-runs are free thanks to the Claude client's
content-addressed cache.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..llm.client import ClaudeClient, ClaudeError, ClaudeRequest, DEFAULT_MODEL
from .schema import Question, parse_iso_date, read_jsonl


SYSTEM_PROMPT = """You construct static forecasting context packets. For a given binary forecasting question with a freeze date F, you will output:

  1. A reference-class base rate: how often events of this type historically resolved YES in the years before F.
  2. A list of relevant news items, each strictly dated BEFORE F.

You are simulating the information a competent human forecaster would have AT THE FREEZE DATE. You must NOT:
  - mention how the question actually resolved (you do not know),
  - include any news dated on or after F,
  - speculate about what will happen after F,
  - hint at the answer through selection of news.

Your tone is neutral and journalistic. You favor verifiable facts over opinion. If you are uncertain about a date, omit the item rather than guess.

Output ONLY a JSON object matching the schema in the user message. No prose before or after the JSON. No markdown fences."""


USER_TEMPLATE = """Question: {question}

Freeze date (the forecaster commits a probability strictly BEFORE this date): {freeze_date}

Background (provided by the source platform):
{background}

Resolution criteria:
{resolution_criteria}

Output a JSON object with exactly two top-level fields:

{{
  "base_rate": {{
    "value": <float in [0, 1], your reference-class estimate>,
    "n_reference_events": <integer; how many similar past events you considered>,
    "reference_window_years": <integer; how many years back you looked>,
    "explanation": <one or two sentences naming the reference class and the historical fraction>
  }},
  "news_snapshot": [
    {{
      "date": "YYYY-MM-DD",
      "headline": <short factual headline>,
      "summary": <one or two sentences of relevant context>
    }},
    ... up to {max_news_items} items, each with date STRICTLY BEFORE {freeze_date}
  ]
}}

Important:
  - Every news item MUST have date < {freeze_date}.
  - Do not mention the actual outcome or any post-{freeze_date} development.
  - If you have low confidence on the base rate, set value near 0.5 and explain.
  - Keep the JSON compact; it will be parsed programmatically."""


DEFAULT_MAX_NEWS_ITEMS = 6
DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.0


@dataclass(frozen=True)
class BaseRate:
    value: float
    n_reference_events: int
    reference_window_years: int
    explanation: str


@dataclass(frozen=True)
class NewsItem:
    date: str
    headline: str
    summary: str


@dataclass
class ContextResult:
    """Generated context for a single question."""

    question_id: str
    base_rate: BaseRate
    news_snapshot: list[NewsItem]
    raw_response: str = ""
    leakage_filtered_count: int = 0  # news items dropped because date >= freeze_date
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "base_rate": asdict(self.base_rate),
            "news_snapshot": [asdict(n) for n in self.news_snapshot],
            "leakage_filtered_count": self.leakage_filtered_count,
            "cache_hit": self.cache_hit,
        }


class ContextParseError(ValueError):
    pass


# ---------------------------------------------------------- prompt construction
def build_request(
    question: Question,
    *,
    model: str = DEFAULT_MODEL,
    max_news_items: int = DEFAULT_MAX_NEWS_ITEMS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ClaudeRequest:
    user = USER_TEMPLATE.format(
        question=question.question.strip(),
        freeze_date=question.freeze_date,
        background=(question.background or "(no background provided)").strip(),
        resolution_criteria=(
            question.resolution_criteria or "(no explicit criteria provided)"
        ).strip(),
        max_news_items=max_news_items,
    )
    return ClaudeRequest.make(
        model=model,
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ---------------------------------------------------------- response parsing
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _try_loose_dict(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a Python-dict-style or relaxed-JSON object.

    Claude occasionally returns responses with single-quoted keys/values
    (Python dict literal syntax) instead of strict JSON, especially on
    templated questions. ``ast.literal_eval`` parses those cleanly and is
    safe for tuples/lists/dicts/strings/numbers/None/True/False — it does
    NOT execute code, only evaluates literals.

    Returns None if the text isn't a dict literal, so the caller can fall
    through to other strategies.
    """

    try:
        result = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(result, dict):
        return result
    return None


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first top-level JSON object out of the response.

    Tries strict JSON, then strict JSON on the largest ``{...}`` substring,
    then falls back to ``ast.literal_eval`` for Python-dict-style responses.
    """

    text = text.strip()
    # Strip optional markdown fences if Claude ignored the no-fences instruction.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("`\n ")

    # 1) Strict JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Strict JSON on the largest object-shaped substring (handles prose
    # before/after the JSON).
    match = _JSON_BLOCK.search(text)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 3) Python-dict literal (single-quoted keys/values, True/False/None).
    loose = _try_loose_dict(candidate)
    if loose is not None:
        return loose
    loose = _try_loose_dict(text)
    if loose is not None:
        return loose

    raise ContextParseError(
        "could not parse response as JSON or Python dict literal "
        f"(first 120 chars: {text[:120]!r})"
    )


def parse_context_response(
    text: str,
    *,
    freeze_date: str,
) -> tuple[BaseRate, list[NewsItem], int]:
    """Parse Claude's response into ``(base_rate, news_snapshot, n_filtered)``.

    News items whose date is missing, unparseable, or >= freeze_date are
    dropped silently and counted as ``n_filtered`` (defense in depth against
    the prompt instruction).
    """

    payload = _extract_json(text)
    raw_br = payload.get("base_rate")
    if not isinstance(raw_br, dict):
        raise ContextParseError("missing or non-object 'base_rate'")
    try:
        value = float(raw_br.get("value"))
    except (TypeError, ValueError) as exc:
        raise ContextParseError("base_rate.value not a number") from exc
    if not (0.0 <= value <= 1.0):
        raise ContextParseError(f"base_rate.value out of [0,1]: {value}")
    base = BaseRate(
        value=value,
        n_reference_events=int(raw_br.get("n_reference_events", 0)),
        reference_window_years=int(raw_br.get("reference_window_years", 0)),
        explanation=str(raw_br.get("explanation", "")).strip(),
    )

    raw_news = payload.get("news_snapshot") or []
    if not isinstance(raw_news, list):
        raise ContextParseError("'news_snapshot' is not a list")

    items: list[NewsItem] = []
    n_filtered = 0
    for item in raw_news:
        if not isinstance(item, dict):
            n_filtered += 1
            continue
        date = parse_iso_date(item.get("date"))
        if date is None or date >= freeze_date:
            n_filtered += 1
            continue
        headline = str(item.get("headline", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not headline and not summary:
            n_filtered += 1
            continue
        items.append(NewsItem(date=date, headline=headline, summary=summary))

    return base, items, n_filtered


# ---------------------------------------------------------- single-call driver
def generate_for_question(
    question: Question,
    client: ClaudeClient,
    *,
    model: str = DEFAULT_MODEL,
    max_news_items: int = DEFAULT_MAX_NEWS_ITEMS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ContextResult:
    request = build_request(
        question,
        model=model,
        max_news_items=max_news_items,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = client.send(request)
    base, news, n_filtered = parse_context_response(
        response.text, freeze_date=question.freeze_date
    )
    return ContextResult(
        question_id=question.id,
        base_rate=base,
        news_snapshot=news,
        raw_response=response.text,
        leakage_filtered_count=n_filtered,
        cache_hit=response.cached,
    )


# ---------------------------------------------------------- batch runner
def generate_all(
    questions: Iterable[Question],
    client: ClaudeClient,
    *,
    output_path: Path,
    max_workers: int = 8,
    on_error: str = "skip",  # "skip" | "raise"
    model: str = DEFAULT_MODEL,
    max_news_items: int = DEFAULT_MAX_NEWS_ITEMS,
) -> dict[str, int]:
    """Process every question concurrently, append-writing successful results.

    Already-cached responses are returned synchronously; the thread-pool only
    spreads the actual API calls across workers.

    Returns a stats dict ``{ok, errors, skipped, leakage_filtered_total}``.
    """

    questions = list(questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip ids that are already in the output.
    done_ids: set[str] = set()
    if output_path.exists():
        for line in output_path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                done_ids.add(json.loads(line)["question_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [q for q in questions if q.id not in done_ids]

    stats = {
        "ok": 0,
        "errors": 0,
        "skipped_already_done": len(done_ids),
        "leakage_filtered_total": 0,
        "cache_hits": 0,
    }

    if not todo:
        return stats

    handle = output_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    generate_for_question,
                    q,
                    client,
                    model=model,
                    max_news_items=max_news_items,
                ): q
                for q in todo
            }
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    result = fut.result()
                except ContextParseError as e:
                    stats["errors"] += 1
                    print(f"  parse error on {q.id}: {e}", file=sys.stderr)
                    if on_error == "raise":
                        raise
                    continue
                except ClaudeError as e:
                    stats["errors"] += 1
                    print(f"  api error on {q.id}: {e}", file=sys.stderr)
                    if on_error == "raise":
                        raise
                    continue
                handle.write(
                    json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
                handle.flush()
                stats["ok"] += 1
                stats["leakage_filtered_total"] += result.leakage_filtered_count
                if result.cache_hit:
                    stats["cache_hits"] += 1
    finally:
        handle.close()
    return stats


# ---------------------------------------------------------- CLI
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate base-rate + dated news context per question via Claude Sonnet."
    )
    parser.add_argument(
        "--input",
        default="data/processed/unified.jsonl",
        help="Path to unified question JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/contexts.jsonl",
        help="Where to write context JSONL (one row per question).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-news-items", type=int, default=DEFAULT_MAX_NEWS_ITEMS)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        default="data/cache/claude",
        help="On-disk Claude response cache (content-addressed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many input questions (0 = all). Useful for smoke tests.",
    )
    parser.add_argument(
        "--on-error",
        choices=("skip", "raise"),
        default="skip",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "warning: $ANTHROPIC_API_KEY not set; only cached responses will succeed",
            file=sys.stderr,
        )

    questions = read_jsonl(args.input)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"input: {len(questions)} questions from {args.input}", file=sys.stderr)

    client = ClaudeClient(api_key=api_key, cache_dir=args.cache_dir)
    stats = generate_all(
        questions,
        client,
        output_path=Path(args.output),
        max_workers=args.max_workers,
        on_error=args.on_error,
        model=args.model,
        max_news_items=args.max_news_items,
    )
    print(json.dumps(stats, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
