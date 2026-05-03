"""Metrics for grouped paraphrase forecasting evaluations."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


PROBABILITY_RE = re.compile(
    r"(?im)(?:^|assistantfinal|\b)\s*(?:final\s+)?probability\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?)(?:\s|$)"
)
LOOSE_NUMBER_RE = re.compile(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*%?)")


def parse_probability(text: str, *, allow_loose: bool = False) -> float | None:
    """Parse a model forecast probability from text.

    Preferred format is a first-line ``Probability: <number between 0 and 1>``.
    Percentages are accepted, so ``Probability: 63%`` becomes ``0.63``.
    ``allow_loose`` is intentionally off by default because permissive parsing
    can hide prompt-following failures.
    """

    match = PROBABILITY_RE.search(text or "")
    if match is None and allow_loose:
        match = LOOSE_NUMBER_RE.search(text or "")
    if match is None:
        return None
    raw = match.group(1).strip()
    is_percent = raw.endswith("%")
    if is_percent:
        raw = raw[:-1].strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    if is_percent:
        value /= 100.0
    if not math.isfinite(value):
        return None
    if 0.0 <= value <= 1.0:
        return value
    return None


def brier(probability: float, outcome: int) -> float:
    return (probability - float(outcome)) ** 2


def log_loss(probability: float, outcome: int, *, eps: float = 1e-6) -> float:
    p = min(max(probability, eps), 1.0 - eps)
    if outcome == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def population_std(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


@dataclass(frozen=True)
class MetricBundle:
    summary: dict[str, Any]
    group_rows: list[dict[str, Any]]


def compute_metrics(prediction_rows: Iterable[dict[str, Any]]) -> MetricBundle:
    """Compute variant- and group-level metrics from prediction rows.

    Expected fields on each row: ``id``, ``variant_index``, ``outcome``,
    ``prediction`` (float or None), and ``parseable`` (bool).
    """

    rows = list(prediction_rows)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row["id"])].append(row)

    parseable_rows = [
        r for r in rows if r.get("parseable") and r.get("prediction") is not None
    ]
    total_variants = len(rows)
    parseable_variants = len(parseable_rows)

    variant_briers = [brier(float(r["prediction"]), int(r["outcome"])) for r in parseable_rows]
    variant_log_losses = [
        log_loss(float(r["prediction"]), int(r["outcome"])) for r in parseable_rows
    ]

    group_rows: list[dict[str, Any]] = []
    full_groups = 0
    parseable_groups = 0
    group_parasds: list[float] = []
    group_briers: list[float] = []

    for qid, group in sorted(by_id.items()):
        group = sorted(group, key=lambda r: int(r.get("variant_index", 0)))
        outcome = int(group[0]["outcome"])
        preds = [
            float(r["prediction"])
            for r in group
            if r.get("parseable") and r.get("prediction") is not None
        ]
        n_variants = len(group)
        n_parseable = len(preds)
        coverage = n_parseable / n_variants if n_variants else 0.0
        if n_parseable:
            parseable_groups += 1
        if n_parseable == n_variants and n_variants > 0:
            full_groups += 1
        mean_prob = sum(preds) / len(preds) if preds else math.nan
        parasd = population_std(preds) if preds else math.nan
        group_brier = (
            sum(brier(p, outcome) for p in preds) / len(preds) if preds else math.nan
        )
        if math.isfinite(parasd):
            group_parasds.append(parasd)
        if math.isfinite(group_brier):
            group_briers.append(group_brier)
        group_rows.append(
            {
                "id": qid,
                "split": group[0].get("split", ""),
                "outcome": outcome,
                "n_variants": n_variants,
                "n_parseable": n_parseable,
                "coverage": coverage,
                "mean_prob": mean_prob,
                "parasd": parasd,
                "mean_brier": group_brier,
            }
        )

    mean_brier = _mean(variant_briers)
    mean_parasd = _mean(group_parasds)
    summary = {
        "n_groups": len(by_id),
        "n_parseable_groups": parseable_groups,
        "n_variant_predictions": total_variants,
        "n_parseable_variant_predictions": parseable_variants,
        "variant_coverage": parseable_variants / total_variants if total_variants else 0.0,
        "group_full_coverage": full_groups / len(by_id) if by_id else 0.0,
        "mean_brier": mean_brier,
        "mean_brier_variant": mean_brier,
        "mean_brier_group": _mean(group_briers),
        "mean_log_loss": _mean(variant_log_losses),
        "mean_parasd": mean_parasd,
        "frie_lambda_0": mean_brier,
        "frie_lambda_1": _sum_if_finite(mean_brier, mean_parasd),
        "frie_lambda_5": _sum_if_finite(mean_brier, _scale_if_finite(mean_parasd, 5.0)),
        "ece_10": expected_calibration_error(parseable_rows, n_bins=10),
        "yes_rate": _mean([int(g[0]["outcome"]) for g in by_id.values()]),
        "mean_prob_outcome_0": _mean(
            [float(r["prediction"]) for r in parseable_rows if int(r["outcome"]) == 0]
        ),
        "mean_prob_outcome_1": _mean(
            [float(r["prediction"]) for r in parseable_rows if int(r["outcome"]) == 1]
        ),
    }
    return MetricBundle(summary=summary, group_rows=group_rows)


def expected_calibration_error(
    rows: list[dict[str, Any]], *, n_bins: int = 10
) -> float | None:
    if not rows:
        return None
    total = len(rows)
    ece = 0.0
    for bin_index in range(n_bins):
        lo = bin_index / n_bins
        hi = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            in_bin = [
                r for r in rows if lo <= float(r["prediction"]) <= hi
            ]
        else:
            in_bin = [
                r for r in rows if lo <= float(r["prediction"]) < hi
            ]
        if not in_bin:
            continue
        conf = _mean([float(r["prediction"]) for r in in_bin])
        acc = _mean([int(r["outcome"]) for r in in_bin])
        assert conf is not None and acc is not None
        ece += len(in_bin) / total * abs(conf - acc)
    return ece


def _mean(values: Iterable[float | int]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _sum_if_finite(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a + b


def _scale_if_finite(value: float | None, scale: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return scale * value
