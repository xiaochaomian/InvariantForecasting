"""Prompt templates for probabilistic forecasting."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You are a careful, calibrated forecasting model. Your task is to forecast "
    "the probability that a binary question resolves Yes. Use only information "
    "that would have been available at the forecast date."
)

MAX_BACKGROUND_CHARS = 2400
MAX_RESOLUTION_CRITERIA_CHARS = 1000
MAX_MARKET_CRITERIA_CHARS = 800

USER_TEMPLATE = """Forecast date: {forecast_date}
Resolution date: {resolution_date}
Source: {source}

Question:
{question}

Background:
{background}

Resolution criteria:
{resolution_criteria}

Market timing:
{market_timing}

Put your forecast on the first line exactly as:
Probability: <number between 0 and 1>

Then add a short rationale on the following lines.
"""


def _clean_text(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return default
    return " ".join(text.split())


def _date_part(value: Any) -> str:
    text = _clean_text(value)
    if text == "N/A":
        return text
    return text[:10]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated} ..."


def build_user_prompt(row: dict[str, Any]) -> str:
    background = _truncate(_clean_text(row.get("background")), MAX_BACKGROUND_CHARS)
    resolution_criteria = _truncate(
        _clean_text(row.get("resolution_criteria")),
        MAX_RESOLUTION_CRITERIA_CHARS,
    )
    market_resolution_criteria = _truncate(
        _clean_text(row.get("market_info_resolution_criteria")),
        MAX_MARKET_CRITERIA_CHARS,
    )
    timing_parts = [
        f"market opened: {_date_part(row.get('market_info_open_datetime'))}",
        f"market closed: {_date_part(row.get('market_info_close_datetime'))}",
    ]
    if market_resolution_criteria != "N/A":
        timing_parts.append(f"market criteria: {market_resolution_criteria}")

    return USER_TEMPLATE.format(
        forecast_date=_date_part(row.get("freeze_datetime")),
        resolution_date=_date_part(row.get("resolved_at")),
        source=_clean_text(row.get("source")),
        question=_clean_text(row.get("question")),
        background=background,
        resolution_criteria=resolution_criteria,
        market_timing="; ".join(timing_parts),
    ).strip()


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row)},
    ]


def build_prompt(row_or_question: dict[str, Any] | str) -> str:
    if isinstance(row_or_question, dict):
        user_prompt = build_user_prompt(row_or_question)
    else:
        user_prompt = USER_TEMPLATE.format(
            forecast_date="N/A",
            resolution_date="N/A",
            source="N/A",
            question=str(row_or_question).strip(),
            background="N/A",
            resolution_criteria="N/A",
            market_timing="market opened: N/A; market closed: N/A",
        ).strip()
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}"


def apply_chat_template(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)
