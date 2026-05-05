"""Post-hoc diagnostics for metacognitive update runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from frame_invariance.eval.metacognitive_update import (
    has_exact_zero_or_one,
    summarize_extremity_bins,
    summarize_prediction_quality_core,
    worst_shift_errors,
)


def read_parseable_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (
                row.get("prior_parseable") == "True"
                and row.get("hypothetical_parseable") == "True"
                and row.get("actual_parseable") == "True"
            ):
                continue
            if not has_required_numbers(row):
                continue
            rows.append(row)
    return rows


def has_required_numbers(row: dict[str, Any]) -> bool:
    required = [
        "prior_prob",
        "hypothetical_prob",
        "actual_prob",
        "hypothetical_logodds_shift",
        "actual_logodds_shift",
        "logodds_shift_error",
        "abs_logodds_shift_error",
    ]
    for field in required:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze metacognitive update rows CSV.")
    parser.add_argument("--rows", required=True, help="Path to metacognitive_update_rows.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostics. Defaults to a diagnostics/ folder next to --rows.",
    )
    parser.add_argument("--worst-limit", type=int, default=25)
    args = parser.parse_args(argv)

    rows_path = Path(args.rows)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else rows_path.parent / "diagnostics"
    )
    rows = read_parseable_rows(rows_path)
    exact_filtered = [
        row
        for row in rows
        if not has_exact_zero_or_one(row, fields=("hypothetical_prob", "actual_prob"))
    ]
    shift_filtered = [
        row for row in rows if abs(float(row["actual_logodds_shift"])) < 5.0
    ]

    summary = {
        "rows_path": str(rows_path),
        "n_parseable_rows": len(rows),
        "all": summarize_prediction_quality_core(rows),
        "excluding_exact_0_1_posteriors": {
            "n": len(exact_filtered),
            **summarize_prediction_quality_core(exact_filtered),
        },
        "abs_actual_shift_lt_5": {
            "n": len(shift_filtered),
            **summarize_prediction_quality_core(shift_filtered),
        },
        "actual_shift_extremity_bins": summarize_extremity_bins(rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_diagnostics.json"
    bins_path = output_dir / "extremity_bins.csv"
    worst_path = output_dir / "worst_cases.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(bins_path, summarize_extremity_bins(rows))
    write_csv(worst_path, worst_shift_errors(rows, limit=args.worst_limit))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"bins:    {bins_path}")
    print(f"worst:   {worst_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
