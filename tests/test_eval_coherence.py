"""Offline tests for Bayesian-coherence benchmark."""

from __future__ import annotations

from frame_invariance.eval.coherence import compute_coherence


def test_compute_coherence_joined_prediction_improvement():
    training = {
        ("q1", 0): {"id": "q1", "variant_index": 0, "split": "validation",
                    "outcome": 1, "base_rate": {"value": 0.2}},
        ("q1", 1): {"id": "q1", "variant_index": 1, "split": "validation",
                    "outcome": 1, "base_rate": {"value": 0.2}},
        ("q2", 0): {"id": "q2", "variant_index": 0, "split": "validation",
                    "outcome": 0, "base_rate": {"value": 0.8}},
    }
    predictions = [
        {"id": "q1", "variant_index": "0", "parseable": "True", "prediction": "0.7"},
        {"id": "q1", "variant_index": "1", "parseable": "True", "prediction": "0.6"},
        {"id": "q2", "variant_index": "0", "parseable": "True", "prediction": "0.3"},
    ]
    summary, group_rows = compute_coherence(
        prediction_rows=predictions,
        training_index=training,
    )
    assert summary["coverage"] == 1.0
    assert summary["brier_improvement_vs_prior"] > 0
    assert summary["log_loss_improvement_vs_prior"] > 0
    assert summary["update_direction_accuracy"] == 1.0
    assert len(group_rows) == 2
