"""Bayesian-coherence benchmark for forecast prediction files.

This is an offline evaluator for an existing ``variant_predictions.csv``. It
joins model probabilities to the base-rate prior in ``training.jsonl`` and asks:

  - did the posterior forecast improve Brier/log-loss relative to the prior?
  - did the model move probability in the correct direction, in hindsight?
  - are log-odds updates stable across paraphrases of the same question?

It is intentionally a benchmark, not a proof of Bayesian correctness: the
dataset gives us base-rate priors and outcomes, but not a formal likelihood
model for the news/context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .metrics import brier, log_loss, population_std


EPS = 1e-6


def read_training_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["id"]), int(row.get("variant_index", 0)))
            index[key] = row
    return index


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_rate_from_row(row: dict[str, Any]) -> float | None:
    raw = (row.get("base_rate") or {}).get("value")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return min(max(value, EPS), 1.0 - EPS)


def prediction_from_row(row: dict[str, Any]) -> float | None:
    if row.get("parseable") not in {True, "True", "true", "1", 1}:
        return None
    try:
        value = float(row.get("prediction"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return min(max(value, EPS), 1.0 - EPS)


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def compute_coherence(
    *,
    prediction_rows: list[dict[str, Any]],
    training_index: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joined: list[dict[str, Any]] = []
    missing_training = 0
    for pred in prediction_rows:
        key = (str(pred.get("id")), int(pred.get("variant_index", 0)))
        train = training_index.get(key)
        if train is None:
            missing_training += 1
            continue
        prior = base_rate_from_row(train)
        posterior = prediction_from_row(pred)
        if prior is None or posterior is None:
            continue
        outcome = int(train["outcome"])
        update = logit(posterior) - logit(prior)
        joined.append(
            {
                "id": key[0],
                "variant_index": key[1],
                "split": train.get("split", ""),
                "outcome": outcome,
                "prior": prior,
                "posterior": posterior,
                "log_odds_update": update,
                "prior_brier": brier(prior, outcome),
                "posterior_brier": brier(posterior, outcome),
                "prior_log_loss": log_loss(prior, outcome),
                "posterior_log_loss": log_loss(posterior, outcome),
                "update_direction_correct": (
                    (outcome == 1 and posterior >= prior)
                    or (outcome == 0 and posterior <= prior)
                ),
            }
        )

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_id[row["id"]].append(row)

    group_rows: list[dict[str, Any]] = []
    for qid, group in sorted(by_id.items()):
        updates = [float(r["log_odds_update"]) for r in group]
        posterior_probs = [float(r["posterior"]) for r in group]
        prior_probs = [float(r["prior"]) for r in group]
        outcome = int(group[0]["outcome"])
        group_rows.append(
            {
                "id": qid,
                "split": group[0]["split"],
                "outcome": outcome,
                "n_parseable": len(group),
                "prior_mean": _mean(prior_probs),
                "posterior_mean": _mean(posterior_probs),
                "mean_log_odds_update": _mean(updates),
                "log_odds_update_sd": population_std(updates),
                "posterior_sd": population_std(posterior_probs),
                "posterior_brier": _mean([float(r["posterior_brier"]) for r in group]),
                "prior_brier": _mean([float(r["prior_brier"]) for r in group]),
                "update_direction_correct_rate": _mean(
                    [1.0 if r["update_direction_correct"] else 0.0 for r in group]
                ),
            }
        )

    prior_briers = [float(r["prior_brier"]) for r in joined]
    posterior_briers = [float(r["posterior_brier"]) for r in joined]
    prior_log_losses = [float(r["prior_log_loss"]) for r in joined]
    posterior_log_losses = [float(r["posterior_log_loss"]) for r in joined]
    updates = [float(r["log_odds_update"]) for r in joined]
    outcome_centered = [float(r["outcome"]) - float(r["prior"]) for r in joined]

    summary = {
        "n_variant_predictions": len(prediction_rows),
        "n_joined_parseable": len(joined),
        "n_groups": len(by_id),
        "missing_training_rows": missing_training,
        "coverage": len(joined) / len(prediction_rows) if prediction_rows else 0.0,
        "mean_prior_brier": _mean(prior_briers),
        "mean_posterior_brier": _mean(posterior_briers),
        "brier_improvement_vs_prior": _mean(prior_briers) - _mean(posterior_briers)
        if prior_briers and posterior_briers
        else None,
        "mean_prior_log_loss": _mean(prior_log_losses),
        "mean_posterior_log_loss": _mean(posterior_log_losses),
        "log_loss_improvement_vs_prior": _mean(prior_log_losses) - _mean(posterior_log_losses)
        if prior_log_losses and posterior_log_losses
        else None,
        "update_direction_accuracy": _mean(
            [1.0 if r["update_direction_correct"] else 0.0 for r in joined]
        ),
        "mean_abs_log_odds_update": _mean([abs(u) for u in updates]),
        "mean_log_odds_update": _mean(updates),
        "mean_group_log_odds_update_sd": _mean(
            [float(g["log_odds_update_sd"]) for g in group_rows]
        ),
        "mean_group_posterior_sd": _mean([float(g["posterior_sd"]) for g in group_rows]),
        "update_outcome_correlation": pearson(updates, outcome_centered),
    }
    return summary, group_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float | int]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return None
    return num / (denx * deny)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bayesian-coherence benchmark.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--training", default="data/processed/training.jsonl")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prediction_path = Path(args.predictions)
    out_dir = Path(args.output_dir) if args.output_dir else prediction_path.parent / "coherence"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, group_rows = compute_coherence(
        prediction_rows=read_prediction_rows(prediction_path),
        training_index=read_training_index(Path(args.training)),
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(out_dir / "group_coherence.csv", group_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"group coherence: {out_dir / 'group_coherence.csv'}")
    print(f"summary:         {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
