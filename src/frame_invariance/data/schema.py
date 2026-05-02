"""Unified question schema.

Every data puller in this package emits rows that conform to ``Question``.
Downstream code (context generation, paraphrasing, training, eval) reads only
this schema, so the source of a question doesn't leak into training paths.

The schema is deliberately minimal at this stage. Later stages add more fields
(``base_rate``, ``news_snapshot``, ``paraphrases``, ...) but those rows are
written to different files; this schema is the canonical *raw question* shape.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# gpt-oss-120b knowledge cutoff is June 2024 per OpenAI model card; we use the
# conservative end-of-month bound. Everything else in the project keys off this.
GPT_OSS_120B_CUTOFF = "2024-06-30"


@dataclass
class Question:
    """One resolved binary forecasting question, normalized across sources.

    Required fields are present for every source; optional fields may be ``None``
    when a particular source doesn't expose them.

    Field semantics:
      - ``id``: globally unique within the unified dataset; use
        ``"{source}::{native_id}"`` as the convention.
      - ``question``: human-readable question text. Should be a complete
        binary question; do not include answer hints.
      - ``outcome``: 0 (no) or 1 (yes), as an int.
      - ``freeze_date``: ISO date (YYYY-MM-DD) before which the model must
        commit a forecast. Forecasting is only meaningful with
        ``freeze_date <= resolved_at``.
      - ``resolved_at``: ISO date (YYYY-MM-DD) at which the question resolved.
      - ``source``: short label (``"forecastbench"``, ``"metaculus"``,
        ``"polymarket"``, ``"manifold"``, ``"mantic_q2_aib"``).
      - ``url``: canonical URL for the resolved market/question, if available.
      - ``background``: free-text context provided by the source (Metaculus
        description, ForecastBench background, Polymarket description, etc.).
      - ``resolution_criteria``: free-text criteria for what counts as YES.
      - ``categories``: optional list of source-supplied tags.
      - ``raw``: opaque source-native payload for debugging and re-ingestion.
    """

    id: str
    question: str
    outcome: int
    freeze_date: str
    resolved_at: str
    source: str
    url: str | None = None
    background: str | None = None
    resolution_criteria: str | None = None
    categories: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1; got {self.outcome!r}")
        if not _is_iso_date(self.freeze_date):
            raise ValueError(f"freeze_date must be YYYY-MM-DD; got {self.freeze_date!r}")
        if not _is_iso_date(self.resolved_at):
            raise ValueError(f"resolved_at must be YYYY-MM-DD; got {self.resolved_at!r}")
        if self.freeze_date > self.resolved_at:
            raise ValueError(
                f"freeze_date {self.freeze_date} must be <= resolved_at {self.resolved_at}"
            )
        if not self.id or "::" not in self.id:
            raise ValueError(f"id must be 'source::native_id'; got {self.id!r}")
        if not self.question.strip():
            raise ValueError("question is empty")

    def is_post_cutoff(self, cutoff_date: str = GPT_OSS_120B_CUTOFF) -> bool:
        """True iff both freeze and resolution are strictly after the cutoff."""

        return self.freeze_date > cutoff_date and self.resolved_at > cutoff_date

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_iso_date(value: str) -> bool:
    if not isinstance(value, str) or len(value) != 10:
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def parse_iso_date(value: Any) -> str | None:
    """Best-effort parse of various date encodings into ``YYYY-MM-DD``.

    Accepts ``str``, ``int`` (unix seconds or millis), and full ISO timestamps.
    Returns ``None`` for missing/invalid inputs rather than raising; callers
    are expected to filter out ``None`` rows.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        # heuristic: > 10^12 looks like millis, otherwise seconds
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date().isoformat()


def write_jsonl(rows: Iterable[Question], path: str | Path) -> int:
    """Write questions to a JSONL file. Returns count written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Question]:
    rows: list[Question] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(Question(**payload))
    return rows
