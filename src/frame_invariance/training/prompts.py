"""Prompt templates for probabilistic forecasting."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful, calibrated forecasting model. Your task is to forecast "
    "the probability that a binary question resolves Yes."
)

USER_TEMPLATE = """Question:
{question}

Return a short rationale, then put your final answer on its own line exactly as:
Probability: <number between 0 and 1>
"""


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(question=question.strip())}"
