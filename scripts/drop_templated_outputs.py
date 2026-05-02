"""Remove stale rows from contexts.jsonl and paraphrases.jsonl whose original
unified.jsonl row contained an unrendered ``{resolution_date}`` /
``{forecast_due_date}`` placeholder.

The fix in forecastbench.py renders those placeholders. But the cache and
output files still contain rows generated against the templated question
text. This script re-reads the OLD contexts and paraphrases files, identifies
rows whose paraphrase or context text contains the literal placeholder, and
removes them. The resume logic in the context/paraphrase generators will
then re-fire just those questions on the next run.

Run:
    python scripts/drop_templated_outputs.py

Writes filtered files in place; backs up the originals to *.bak.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

PLACEHOLDERS = ("{resolution_date}", "{forecast_due_date}")


def has_placeholder(obj) -> bool:
    if isinstance(obj, str):
        return any(p in obj for p in PLACEHOLDERS)
    if isinstance(obj, dict):
        return any(has_placeholder(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_placeholder(v) for v in obj)
    return False


def filter_jsonl(path: Path) -> tuple[int, int]:
    if not path.exists():
        print(f"  (skip) {path} does not exist")
        return 0, 0
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  backed up to {backup.name}")

    kept: list[str] = []
    dropped = 0
    total = 0
    with backup.open(encoding="utf-8") as h:
        for line in h:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if has_placeholder(obj):
                dropped += 1
            else:
                kept.append(line)

    with path.open("w", encoding="utf-8") as h:
        h.write("\n".join(kept))
        if kept:
            h.write("\n")
    return total, dropped


def main() -> int:
    for name in ("data/processed/contexts.jsonl", "data/processed/paraphrases.jsonl"):
        path = Path(name)
        print(f"filtering {path}")
        total, dropped = filter_jsonl(path)
        print(f"  total={total}  dropped={dropped}  kept={total - dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
