"""Claude-based paraphrase generator for forecasting questions.

For each ``Question``, ask Claude Sonnet to produce K-1 paraphrases that
preserve dates, numbers, and named entities exactly. The original is included
as ``variant_index=0``; paraphrases are ``1..K-1``. The default K=8 matches
Mantic's recipe (group size 8 in GRPO).

We post-validate every Claude paraphrase by extracting dates/numbers/proper
nouns from the original and rejecting any paraphrase that drops or mutates
them. If too few paraphrases survive, we *re-prompt* with the original plus
a list of bad paraphrases until we have K-1, or give up after 3 rounds.

This is the cheapest defense against silent corruption — Claude occasionally
mutates "Israel-Iran" into "the Israel-Iran ceasefire" or strips the year
from a deadline. Those would be data-set-breaking bugs in training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..llm.client import ClaudeClient, ClaudeError, ClaudeRequest, DEFAULT_MODEL
from .schema import Question, read_jsonl


SYSTEM_PROMPT = """You paraphrase forecasting questions while preserving meaning.

Your output is parsed programmatically as a JSON list of strings. Output ONLY the JSON list. No prose before or after. No markdown fences.

You MUST preserve EVERY:
  - date (YYYY-MM-DD, "January 1, 2025", "Jan 2025", etc.) verbatim,
  - number (including percentages, dollar amounts, multipliers) verbatim,
  - proper noun (people, organizations, places, products, acronyms) verbatim,
  - quantitative threshold ("at least 5", "more than 10%", "by end of 2025") verbatim,
  - resolution-relevant negation ("will NOT", "fails to", "does not").

You MAY change:
  - voice (active/passive),
  - clause order,
  - syntactic frame (interrogative vs declarative-interrogative),
  - synonym choice for COMMON verbs and CONNECTIVES (will/does, before/by, happens/occurs),
  - punctuation.

You MAY NOT change:
  - the semantic question (what would resolve YES must be unchanged),
  - any specific date, number, or named entity,
  - the implicit reference class.

If two paraphrases would only differ in punctuation, generate something more substantively different instead."""


USER_TEMPLATE = """Original forecasting question:

{question}

Generate exactly {k} paraphrases. Each paraphrase must be semantically equivalent to the original, asking the same yes/no question with the same resolution semantics.

Output a JSON list of exactly {k} strings, each one a paraphrase. No keys, no metadata, just the strings.

Example output format (illustrative, do not copy):
["paraphrase 1", "paraphrase 2", ...]"""


RETRY_USER_TEMPLATE = """The previous attempt produced paraphrases that violated the entity/date preservation requirements:

Bad paraphrases:
{bad_list}

Original:
{question}

Already-validated paraphrases ({n_have} of {k_target}):
{good_list}

Generate {k_remaining} more paraphrases. Same rules: preserve all entities, dates, numbers, and quantitative thresholds verbatim. Output a JSON list of exactly {k_remaining} strings."""


DEFAULT_K = 8                # group size 8 = original + 7 paraphrases (Mantic recipe)
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.7    # we WANT diversity in surface form
DEFAULT_RETRY_ROUNDS = 3


@dataclass(frozen=True)
class Paraphrase:
    variant_index: int
    text: str
    is_original: bool


@dataclass
class ParaphraseResult:
    question_id: str
    paraphrases: list[Paraphrase]
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (text, reason)
    rounds: int = 1
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "paraphrases": [asdict(p) for p in self.paraphrases],
            "rejected_count": len(self.rejected),
            "rejected_samples": self.rejected[:5],  # keep a few for debugging
            "rounds": self.rounds,
            "cache_hit": self.cache_hit,
        }


class ParaphraseParseError(ValueError):
    pass


# ----------------------------------------------------- entity-preservation
# Matches dates in many formats, numbers (including %, $, decimals), and
# capitalized proper-noun-like spans. Inexact but high recall — false
# positives just mean we sometimes reject valid paraphrases, which is the
# safer failure mode.

_DATE_PATTERNS = [
    re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b"),                           # 2025-09-01
    re.compile(r"\b(20\d{2}/\d{2}/\d{2})\b"),                           # 2025/09/01
    re.compile(
        r"\b("
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
        r")\s+\d{1,2}(?:,\s*20\d{2})?\b"
    ),                                                                   # September 1, 2025
    re.compile(
        r"\b("
        r"January|February|March|April|May|June|July|August|"
        r"September|October|November|December"
        r")\s+20\d{2}\b"
    ),                                                                   # September 2025
    re.compile(r"\b(20\d{2})\b"),                                        # bare 2025
]

_NUMBER = re.compile(
    r"(?<![\w$])"
    r"(\$?\d+(?:[,.]\d+)?(?:\s*%)?(?:[kKmMbB])?)"
    r"(?![\w])"
)
_PROPER_RUN = re.compile(
    r"\b((?:[A-Z][\w'\-]*\s+)*[A-Z][\w'\-]*)"  # one or more capitalized tokens
)
# Stop words we don't want to flag as proper nouns just because they appear at
# sentence start.
_SENTENCE_INITIAL_OK = {
    "Will", "Does", "Is", "By", "On", "At", "After", "Before",
    "What", "When", "Which", "Who", "If", "The", "A", "An", "In",
    "How", "Are", "As", "Of", "For", "From", "To", "And", "Or",
    "But", "It", "This", "That", "These", "Those",
}


def extract_dates(text: str) -> set[str]:
    out: set[str] = set()
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            out.add(m.group(0).strip())
    return out


def extract_numbers(text: str) -> set[str]:
    return {m.group(1) for m in _NUMBER.finditer(text)}


def extract_proper_nouns(text: str) -> set[str]:
    out: set[str] = set()
    # Only look at tokens that are NOT at the start of a sentence (heuristic:
    # not preceded by sentence-final punctuation or position 0).
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        stripped = sentence.lstrip()
        first_word = stripped.split(" ", 1)[0] if stripped else ""
        # If the first word is a capitalized common-question-starter, drop it
        # before scanning so we don't penalize sentence-initial capitalization.
        if first_word in _SENTENCE_INITIAL_OK:
            stripped = stripped[len(first_word):].lstrip()
        for m in _PROPER_RUN.finditer(stripped):
            span = m.group(1).strip()
            # Skip pure dates (we've already extracted those).
            if span in _SENTENCE_INITIAL_OK:
                continue
            if extract_dates(span):
                continue
            # Skip single-letter spans (e.g. stray "I").
            if len(span) <= 1:
                continue
            out.add(span)
    return out


def validate_paraphrase(original: str, candidate: str) -> tuple[bool, str]:
    """Return ``(is_valid, reason_if_invalid)``."""

    orig_dates = extract_dates(original)
    cand_dates = extract_dates(candidate)
    missing_dates = orig_dates - cand_dates
    if missing_dates:
        return False, f"missing date(s): {sorted(missing_dates)}"

    orig_numbers = extract_numbers(original)
    cand_numbers = extract_numbers(candidate)
    # Year tokens (2024, 2025, 2026) get caught by both date and number
    # patterns; treat the date set as authoritative for year-strings.
    orig_numbers_real = {n for n in orig_numbers if not re.fullmatch(r"20\d{2}", n)}
    cand_numbers_real = {n for n in cand_numbers if not re.fullmatch(r"20\d{2}", n)}
    missing_numbers = orig_numbers_real - cand_numbers_real
    if missing_numbers:
        return False, f"missing number(s): {sorted(missing_numbers)}"

    orig_propers = extract_proper_nouns(original)
    cand_propers = extract_proper_nouns(candidate)
    # We allow proper-noun re-casing inside the string (lowercase common-noun
    # use of an acronym, e.g. "Spotify"->"the spotify chart"), but we require
    # every original proper noun to appear *somewhere* in the candidate as a
    # case-insensitive substring.
    cand_lower = candidate.lower()
    missing_propers = {p for p in orig_propers if p.lower() not in cand_lower}
    if missing_propers:
        return False, f"missing proper noun(s): {sorted(missing_propers)}"

    # Reject empty / whitespace-only / near-identical paraphrases.
    if not candidate.strip():
        return False, "empty"
    if candidate.strip() == original.strip():
        return False, "identical to original"

    return True, ""


# ----------------------------------------------------- response parsing
def parse_paraphrase_list(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("`\n ")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Try to locate the first list literal.
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            raise ParaphraseParseError("no JSON list found in response")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ParaphraseParseError(f"malformed JSON list: {exc}") from exc
    if not isinstance(payload, list):
        raise ParaphraseParseError(f"expected a list, got {type(payload).__name__}")
    out: list[str] = []
    for item in payload:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict) and "text" in item:
            out.append(str(item["text"]).strip())
    return [s for s in out if s]


# ----------------------------------------------------- single-call driver
def _initial_request(
    question_text: str,
    *,
    k: int,
    model: str,
    max_tokens: int,
    temperature: float,
) -> ClaudeRequest:
    user = USER_TEMPLATE.format(question=question_text.strip(), k=k - 1)
    return ClaudeRequest.make(
        model=model,
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _retry_request(
    question_text: str,
    *,
    good: list[str],
    bad: list[tuple[str, str]],
    k: int,
    k_remaining: int,
    model: str,
    max_tokens: int,
    temperature: float,
) -> ClaudeRequest:
    bad_list = "\n".join(f"  - {t} ({reason})" for t, reason in bad[-5:])
    good_list = "\n".join(f"  - {t}" for t in good)
    user = RETRY_USER_TEMPLATE.format(
        bad_list=bad_list or "  (none)",
        question=question_text.strip(),
        good_list=good_list or "  (none)",
        n_have=len(good),
        k_target=k - 1,
        k_remaining=k_remaining,
    )
    return ClaudeRequest.make(
        model=model,
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def generate_for_question(
    question: Question,
    client: ClaudeClient,
    *,
    k: int = DEFAULT_K,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    retry_rounds: int = DEFAULT_RETRY_ROUNDS,
) -> ParaphraseResult:
    if k < 2:
        raise ValueError("k must be >= 2 (original + at least one paraphrase)")

    good: list[str] = []
    bad: list[tuple[str, str]] = []
    cache_hit_any = False

    request = _initial_request(
        question.question,
        k=k,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    rounds_used = 0
    for round_idx in range(retry_rounds):
        rounds_used = round_idx + 1
        response = client.send(request)
        cache_hit_any = cache_hit_any or response.cached
        try:
            candidates = parse_paraphrase_list(response.text)
        except ParaphraseParseError as exc:
            bad.append(("<unparseable response>", str(exc)))
            candidates = []

        for candidate in candidates:
            if len(good) >= k - 1:
                break
            ok, reason = validate_paraphrase(question.question, candidate)
            if ok and candidate not in good:
                good.append(candidate)
            else:
                bad.append((candidate, reason or "duplicate"))

        if len(good) >= k - 1:
            break

        # Re-prompt with explicit feedback.
        request = _retry_request(
            question.question,
            good=good,
            bad=bad,
            k=k,
            k_remaining=(k - 1) - len(good),
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    paraphrases = [
        Paraphrase(variant_index=0, text=question.question, is_original=True)
    ]
    for i, text in enumerate(good[: k - 1], start=1):
        paraphrases.append(Paraphrase(variant_index=i, text=text, is_original=False))

    return ParaphraseResult(
        question_id=question.id,
        paraphrases=paraphrases,
        rejected=bad,
        rounds=rounds_used,
        cache_hit=cache_hit_any,
    )


# ----------------------------------------------------- batch runner
def generate_all(
    questions: Iterable[Question],
    client: ClaudeClient,
    *,
    output_path: Path,
    k: int = DEFAULT_K,
    max_workers: int = 8,
    on_error: str = "skip",
    model: str = DEFAULT_MODEL,
) -> dict[str, int]:
    questions = list(questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        "incomplete_groups": 0,  # got fewer than K-1 paraphrases after retries
        "skipped_already_done": len(done_ids),
        "cache_hits": 0,
    }
    if not todo:
        return stats

    handle = output_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(generate_for_question, q, client, k=k, model=model): q
                for q in todo
            }
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    result = fut.result()
                except (ParaphraseParseError, ClaudeError) as e:
                    stats["errors"] += 1
                    print(f"  paraphrase error on {q.id}: {e}", file=sys.stderr)
                    if on_error == "raise":
                        raise
                    continue
                handle.write(
                    json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False)
                    + "\n"
                )
                handle.flush()
                stats["ok"] += 1
                if len(result.paraphrases) < k:
                    stats["incomplete_groups"] += 1
                if result.cache_hit:
                    stats["cache_hits"] += 1
    finally:
        handle.close()
    return stats


# ----------------------------------------------------- CLI
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate K-1 paraphrases per forecasting question via Claude Sonnet."
    )
    parser.add_argument("--input", default="data/processed/unified.jsonl")
    parser.add_argument("--output", default="data/processed/paraphrases.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Group size; produces K-1 paraphrases per question.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="data/cache/claude")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--on-error", choices=("skip", "raise"), default="skip")
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
        k=args.k,
        max_workers=args.max_workers,
        on_error=args.on_error,
        model=args.model,
    )
    print(json.dumps(stats, indent=2, sort_keys=True), file=sys.stderr)
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
