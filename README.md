# InvariantForecasting

data-prepping so far

## ForecastBench/current data

https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets

Post-August resolution dates: 598 unique resolved binary questions.

```bash
python3 scripts/fetch_forecastbench_current.py --since 2025-08-31 --max-questions 2000
```

So far this is limited to 598 rows; we can look for other data sources.

Outputs:
- `data/raw/forecastbench_current_after_2025-08-31.jsonl`
- `data/processed/forecastbench_current_after_2025-08-31.csv`

## ForecastBench/current paraphrases

```bash
python3 scripts/make_forecastbench_paraphrases.py
```

Output:
- `data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl`
- `data/processed/forecastbench_current_after_2025-08-31_5x_paraphrases.csv`

These are long-form files with 2,990 rows: 598 source questions * 5 variants.
Every five consecutive rows are one question group. `variant_index=0` is the
original question; `variant_index=1..4` are deterministic semantic paraphrases.
The file also keeps `source_question_index`, `id`, `variant_id`, `source`,
`outcome`, and `resolved_at` for safer downstream joins.

Big note: I emailed Metaculus, so we may be able to use their data later.

Open the processed CSV with a spreadsheet app/importer, or use the raw JSONL for programmatic work.

## GRPO post-training scaffold

The training scaffold follows the paper proposal: per-paraphrase Brier reward plus an explicit variance penalty across semantically equivalent question frames.

Reward:

```text
R_i = -(p_i - Y)^2 - lambda * (p_i - mean_group_probability)^2
```

Dry-run the data split and reward wiring locally:

```bash
PYTHONPATH=src python3 -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench.yaml \
  --dry-run
```

Run the actual GRPO job in a GPU environment after installing dependencies:

```bash
pip install -e .
PYTHONPATH=src python3 -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench.yaml
```

Current split from the 5x paraphrase file:

- train: 478 question groups / 2,390 paraphrase rows
- validation: 60 question groups / 300 paraphrase rows
- test: 60 question groups / 300 paraphrase rows

`lambda_invariance` in `configs/training/grpo_forecastbench.yaml` controls the explicit paraphrase-variance penalty. Set it to `0.0` for the outcome-only Brier RL baseline and positive values for the proposed frame-invariant objective.

## Getting actual results

After a model is available on a GPU node, evaluate the base model on the held-out test split:

```bash
PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench.yaml \
  --model Qwen/Qwen2.5-7B-Instruct \
  --run-name base_qwen7b \
  --split test \
  --batch-size 4
```

This writes:

- `results/base_qwen7b/test/summary.json`
- `results/base_qwen7b/test/group_metrics.csv`
- `results/base_qwen7b/test/variant_predictions.csv`

Then train/evaluate the outcome-only baseline by setting `lambda_invariance: 0.0` and a lambda0 output directory in the config, running GRPO, and evaluating the trained checkpoint:

```bash
PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench.yaml \
  --model outputs/grpo_forecastbench_qwen7b_lambda0 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --run-name grpo_lambda0 \
  --split test \
  --batch-size 4
```

Then train/evaluate the proposed invariant objective with `lambda_invariance: 1.0`:

```bash
PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench.yaml \
  --model outputs/grpo_forecastbench_qwen7b_lambda1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --run-name grpo_lambda1 \
  --split test \
  --batch-size 4
```

The paper table comes from each `summary.json`:

- `mean_brier`: forecasting accuracy, lower is better
- `mean_parasd`: paraphrase standard deviation, lower is better
- `variant_coverage`: fraction of paraphrase forecasts with parseable probabilities
