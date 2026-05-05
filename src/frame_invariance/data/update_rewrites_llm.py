"""Generate and validate strongly framed evidence-update rewrites.

This file creates the new V2.5 evidence packet:

``data/processed/update_rewrites_strong.jsonl``

Each row contains five fact-preserving rewrites of the same dated news packet:

  0. neutral
  1. weak_yes
  2. strong_yes
  3. weak_no
  4. strong_no

The "yes/no" labels are *framing pressures*, not predictions and not labels.
The rewrite is allowed to emphasize, hedge, order, and word the same facts
differently. It is not allowed to add facts, reveal outcomes, or use post-freeze
information.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ..llm.client import ClaudeClient, ClaudeRequest, DEFAULT_MODEL
from .build_training_set import index_by_question_id, read_jsonl
from .schema import Question, read_jsonl as read_questions


DEFAULT_OUTPUT = "data/processed/update_rewrites_strong.jsonl"
DEFAULT_PROMPT_OUTPUT = "data/processed/update_rewrite_prompts.jsonl"
DEFAULT_K = 5
EXPECTED_VARIANTS = [
    ("neutral", 0),
    ("yes", 1),
    ("yes", 2),
    ("no", 1),
    ("no", 2),
]
STYLE_NAMES = [
    "neutral",
    "weak_yes",
    "strong_yes",
    "weak_no",
    "strong_no",
]
SANITIZE_REPLACEMENTS = [
    (re.compile(r"\bspecific 30[- ]day window in question\b", re.I), "specific period in question"),
    (re.compile(r"\b30[- ]day window in question\b", re.I), "period in question"),
    (re.compile(r"\bspecific 30[- ]day window\b", re.I), "specific period"),
    (re.compile(r"\b30[- ]day window\b", re.I), "period"),
    (re.compile(r"\b30[- ]day window in question\b", re.I), "period in question"),
    (re.compile(r"\bpreceding 360 days\b", re.I), "historical reference period"),
    (re.compile(r"\b360 days\b", re.I), "historical period"),
    (re.compile(r"\bresolution window\b", re.I), "stated range"),
    (re.compile(r"\bten[- ]times\b", re.I), "large relative"),
    (re.compile(r"\b10\s*x\b", re.I), "large relative"),
    (re.compile(r"\bdenominator\b", re.I), "comparison baseline"),
    (re.compile(r"\bbaseline condition\b", re.I), "comparison"),
]


SYSTEM_PROMPT = """You rewrite evidence packets for a forecasting dataset.

Your job is NOT to forecast. Your job is to produce fact-preserving versions of
the evidence packet with different implicit update pressure.

Output valid JSON only. No markdown fences. No prose outside JSON."""


USER_TEMPLATE = """Create five evidence-update rewrites from the same facts.

Hard constraints:
- Use only the evidence provided below.
- Do not add new facts, hidden assumptions, or causal claims.
- Do not remove any important evidence item.
- Include every evidence item and every bracketed evidence date in every rewrite.
- Preserve all dates, named entities, places, quantities, and uncertainty.
- Do not use information after the forecast date.
- The question, forecast date, resolution date, and base-rate prior are for
  relevance only. Do not copy any date from those fields into update_text unless
  that date also appears inside the Evidence packet.
- Every date in update_text must be one of the bracketed dates shown in the
  Evidence packet.
- update_text must rewrite only the Evidence packet. Do not discuss the
  base-rate prior, the resolution criteria, the threshold, the forecast window,
  the "30 days" / "360 days" mechanics, or what would make the question resolve.
- Never write the phrases "prior probability", "resolution window",
  "denominator", "ten-times", "ten times", "preceding 360", or
  "baseline condition".
  For numeric-range questions, describe the evidence as above, below, inside,
  or outside the relevant range without naming the resolution machinery.
- Before returning JSON, check each individual rewrite against the Required
  evidence dates checklist. Every rewrite must literally contain every date in
  that checklist.
- Do not reveal, imply, or discuss the true outcome.
- Do not output a probability.
- Framing bias may come only from emphasis, ordering, hedging, and word choice.
- If the facts cannot honestly support one side, make that side skeptical or
  uncertainty-focused rather than false.

Return JSON in exactly this shape:
{{
  "rewrites": [
    {{"style": "neutral", "stance": "neutral", "strength": 0, "update_text": "..."}},
    {{"style": "weak_yes", "stance": "yes", "strength": 1, "update_text": "..."}},
    {{"style": "strong_yes", "stance": "yes", "strength": 2, "update_text": "..."}},
    {{"style": "weak_no", "stance": "no", "strength": 1, "update_text": "..."}},
    {{"style": "strong_no", "stance": "no", "strength": 2, "update_text": "..."}}
  ]
}}

Question:
{question}

Forecast date:
{freeze_date}

Resolution date:
{resolved_at}

Base-rate prior:
{base_rate}

Evidence packet:
{evidence_packet}

Required evidence dates checklist:
{required_dates}
"""


FORBIDDEN_LEAKAGE_PATTERNS = [
    re.compile(r"\btrue outcome\b", re.I),
    re.compile(r"\bactual outcome\b", re.I),
    re.compile(r"\bresolved (?:as|to)\b", re.I),
    re.compile(r"\bresolves? (?:positively|negatively)\b", re.I),
    re.compile(r"\bresolution result\b", re.I),
    re.compile(r"\bresolution criteria\b", re.I),
    re.compile(r"\bresolution window\b", re.I),
    re.compile(r"\banswer is\b", re.I),
    re.compile(r"\bprobability\s*:", re.I),
    re.compile(r"\bbase[- ]rate prior\b", re.I),
    re.compile(r"\bbase ?rate prior\b", re.I),
    re.compile(r"\bprior probability\b", re.I),
    re.compile(r"\b30[- ]day window in question\b", re.I),
    re.compile(r"\b360 days\b", re.I),
    re.compile(r"\bpreceding 360\b", re.I),
    re.compile(r"\bdenominator\b", re.I),
    re.compile(r"\bten[- ]times\b", re.I),
    re.compile(r"\b10\s*x\b", re.I),
    re.compile(r"\bbaseline condition\b", re.I),
]
ISO_DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


@dataclass(frozen=True)
class RewriteValidation:
    ok: bool
    errors: list[str]


def news_hash(news_snapshot: list[dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(news_snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_evidence_packet(news_snapshot: list[dict[str, Any]]) -> str:
    if not news_snapshot:
        return "(no dated evidence items were provided)"
    lines: list[str] = []
    for item in news_snapshot:
        date = str(item.get("date", "")).strip()
        headline = str(item.get("headline", "")).strip()
        summary = str(item.get("summary", "")).strip()
        line = f"[{date}] {headline}"
        if summary:
            line += f" -- {summary}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(question: Question, context: dict[str, Any]) -> str:
    news_snapshot = context.get("news_snapshot") or []
    required_dates = [
        str(item.get("date", "")).strip()
        for item in news_snapshot
        if str(item.get("date", "")).strip()
    ]
    return USER_TEMPLATE.format(
        question=question.question.strip(),
        freeze_date=question.freeze_date,
        resolved_at=question.resolved_at,
        base_rate=json.dumps(context.get("base_rate") or {}, ensure_ascii=False, sort_keys=True),
        evidence_packet=format_evidence_packet(news_snapshot),
        required_dates=", ".join(required_dates) if required_dates else "(none)",
    )


def parse_rewrite_response(text: str) -> list[dict[str, Any]]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("no JSON object found in rewrite response")
        payload = json.loads(match.group(0))
    rewrites = payload.get("rewrites") if isinstance(payload, dict) else None
    if not isinstance(rewrites, list):
        raise ValueError("rewrite response must contain a 'rewrites' list")
    return [dict(item) for item in rewrites if isinstance(item, dict)]


def normalize_rewrites(rewrites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, item in enumerate(rewrites):
        style = str(item.get("style") or (STYLE_NAMES[i] if i < len(STYLE_NAMES) else "")).strip()
        stance, strength = EXPECTED_VARIANTS[i] if i < len(EXPECTED_VARIANTS) else ("", -1)
        update_text = sanitize_update_text(str(item.get("update_text") or "").strip())
        out.append(
            {
                "update_variant_index": i,
                "style": style,
                "stance": str(item.get("stance") or stance).strip(),
                "strength": int(item.get("strength", strength)),
                "update_text": update_text,
            }
        )
    return out


def sanitize_update_text(text: str) -> str:
    """Remove boilerplate forecasting mechanics while preserving evidence text."""
    sanitized = text
    for pattern, replacement in SANITIZE_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized.strip()


def validate_rewrite_row(row: dict[str, Any], *, context: dict[str, Any], freeze_date: str) -> RewriteValidation:
    errors: list[str] = []
    rewrites = row.get("rewrites")
    if not isinstance(rewrites, list):
        return RewriteValidation(False, ["missing rewrites list"])
    if len(rewrites) != DEFAULT_K:
        errors.append(f"expected {DEFAULT_K} rewrites, got {len(rewrites)}")

    news_snapshot = context.get("news_snapshot") or []
    evidence_text = format_evidence_packet(news_snapshot)
    allowed_dates = set(ISO_DATE.findall(evidence_text))
    required_dates = {
        str(item.get("date", "")).strip()
        for item in news_snapshot
        if str(item.get("date", "")).strip()
    }

    for i, rewrite in enumerate(rewrites):
        if not isinstance(rewrite, dict):
            errors.append(f"rewrite {i} is not an object")
            continue
        expected_stance, expected_strength = (
            EXPECTED_VARIANTS[i] if i < len(EXPECTED_VARIANTS) else ("", -1)
        )
        text = str(rewrite.get("update_text") or "").strip()
        if rewrite.get("update_variant_index") != i:
            errors.append(f"rewrite {i} has wrong update_variant_index")
        if rewrite.get("stance") != expected_stance:
            errors.append(f"rewrite {i} has stance {rewrite.get('stance')!r}, expected {expected_stance!r}")
        if int(rewrite.get("strength", -1)) != expected_strength:
            errors.append(
                f"rewrite {i} has strength {rewrite.get('strength')!r}, expected {expected_strength}"
            )
        if len(text) < 80:
            errors.append(f"rewrite {i} is too short")
        if len(text) > 5000:
            errors.append(f"rewrite {i} is too long")
        missing_dates = sorted(d for d in required_dates if d and d not in text)
        if missing_dates:
            errors.append(f"rewrite {i} is missing news date(s): {missing_dates[:5]}")
        for pattern in FORBIDDEN_LEAKAGE_PATTERNS:
            if pattern.search(text):
                errors.append(f"rewrite {i} contains forbidden leakage phrase: {pattern.pattern}")
        for date_text in ISO_DATE.findall(text):
            if allowed_dates and date_text not in allowed_dates:
                errors.append(f"rewrite {i} contains non-evidence date {date_text}")
            if _date_after(date_text, freeze_date):
                errors.append(f"rewrite {i} contains post-freeze date {date_text}")

    return RewriteValidation(not errors, errors)


def _date_after(date_text: str, freeze_date: str) -> bool:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d") > datetime.strptime(freeze_date, "%Y-%m-%d")
    except ValueError:
        return False


def make_prompt_rows(
    *,
    questions: Iterable[Question],
    contexts_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in questions:
        ctx = contexts_by_id.get(q.id)
        if ctx is None:
            continue
        news_snapshot = ctx.get("news_snapshot") or []
        rows.append(
            {
                "question_id": q.id,
                "source_news_hash": news_hash(news_snapshot),
                "system": SYSTEM_PROMPT,
                "prompt": build_prompt(q, ctx),
                "expected_styles": STYLE_NAMES,
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if row.get("question_id") and row.get("valid") is True:
            ids.add(str(row["question_id"]))
    return ids


def generate_one(
    q: Question,
    ctx: dict[str, Any],
    *,
    client: ClaudeClient,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    request = ClaudeRequest.make(
        model=model,
        system=SYSTEM_PROMPT,
        user=build_prompt(q, ctx),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = client.send(request)
    rewrites = normalize_rewrites(parse_rewrite_response(response.text))
    row = {
        "question_id": q.id,
        "source_news_hash": news_hash(ctx.get("news_snapshot") or []),
        "rewrites": rewrites,
        "cache_hit": response.cached,
        "usage": response.usage,
    }
    validation = validate_rewrite_row(row, context=ctx, freeze_date=q.freeze_date)
    row["valid"] = validation.ok
    row["validation_errors"] = validation.errors
    return row


def validate_file(path: Path, *, questions_by_id: dict[str, Question], contexts_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        qid = str(raw.get("question_id", ""))
        if qid:
            rows_by_id[qid] = raw
    total = 0
    valid = 0
    errors: list[dict[str, Any]] = []
    for raw in rows_by_id.values():
        total += 1
        qid = str(raw.get("question_id", ""))
        q = questions_by_id.get(qid)
        ctx = contexts_by_id.get(qid)
        if q is None or ctx is None:
            errors.append({"question_id": qid, "errors": ["missing question or context"]})
            continue
        if raw.get("valid") is False and raw.get("validation_errors") and not raw.get("rewrites"):
            errors.append({"question_id": qid, "errors": list(raw["validation_errors"])})
            continue
        normalized = dict(raw)
        normalized["rewrites"] = normalize_rewrites(list(raw.get("rewrites") or []))
        result = validate_rewrite_row(normalized, context=ctx, freeze_date=q.freeze_date)
        if result.ok:
            valid += 1
        else:
            errors.append({"question_id": qid, "errors": result.errors})
    return {"rows": total, "valid_rows": valid, "invalid_rows": total - valid, "errors": errors[:20]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate/export/validate strong update rewrites.")
    parser.add_argument("--unified", default="data/processed/unified.jsonl")
    parser.add_argument("--contexts", default="data/processed/contexts.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt-output", default=DEFAULT_PROMPT_OUTPUT)
    parser.add_argument("--mode", choices=("export-prompts", "generate", "validate"), default="export-prompts")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file before generation. Useful after auth errors or prompt changes.",
    )
    args = parser.parse_args(argv)

    questions = read_questions(args.unified)
    if args.limit is not None:
        questions = questions[: args.limit]
    contexts_by_id = index_by_question_id(read_jsonl(Path(args.contexts)))
    questions_by_id = {q.id: q for q in questions}

    if args.mode == "export-prompts":
        rows = make_prompt_rows(questions=questions, contexts_by_id=contexts_by_id)
        n = write_jsonl(Path(args.prompt_output), rows)
        print(json.dumps({"prompt_rows": n, "prompt_output": args.prompt_output}, indent=2))
        return 0

    if args.mode == "validate":
        summary = validate_file(
            Path(args.output), questions_by_id=questions_by_id, contexts_by_id=contexts_by_id
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["invalid_rows"] == 0 else 1

    output = Path(args.output)
    if args.overwrite and output.exists():
        output.unlink()
    done = read_existing_ids(output)
    todo = [q for q in questions if q.id not in done and q.id in contexts_by_id]
    client = ClaudeClient()
    print(f"generating {len(todo)} rewrite rows; already done {len(done)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                generate_one,
                q,
                contexts_by_id[q.id],
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ): q
            for q in todo
        }
        for i, future in enumerate(as_completed(futures), start=1):
            q = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve batch progress
                row = {
                    "question_id": q.id,
                    "source_news_hash": news_hash(contexts_by_id[q.id].get("news_snapshot") or []),
                    "rewrites": [],
                    "valid": False,
                    "validation_errors": [f"{type(exc).__name__}: {exc}"],
                }
            append_jsonl(output, row)
            if i % 25 == 0 or i == len(todo):
                print(f"wrote {i}/{len(todo)} rewrite rows", file=sys.stderr)

    summary = validate_file(output, questions_by_id=questions_by_id, contexts_by_id=contexts_by_id)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["invalid_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
