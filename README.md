# InvariantForecasting

data-prepping so far

## First run results

Full documentation for the first end-to-end run is in [`firstrundocumentation.tex`](firstrundocumentation.tex).

Held-out test split: 60 ForecastBench/current question groups, 5 paraphrases per group, 300 variant predictions total.

| Run | Objective | Mean Brier (lower) | Mean ParaSD (lower) | Variant Coverage | Full-Group Coverage |
|---|---|---:|---:|---:|---:|
| Base Qwen2.5-7B | No RL, strict parser | **0.2436** | 0.1229 | 0.6667 | 0.2833 |
| GRPO lambda=0 | Brier reward only | 0.2508 | 0.0894 | **1.0000** | **1.0000** |
| GRPO lambda=1 | Brier + paraphrase variance penalty | 0.2495 | **0.0800** | **1.0000** | **1.0000** |

Main first-run finding: `lambda=1` reduced paraphrase standard deviation by 34.9% vs. base and 10.5% vs. `lambda=0`, without an additional Brier cost relative to outcome-only GRPO. Base still has the best Brier, but with substantially lower parse coverage.

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
  --run-name base_qwen7b_strict \
  --split test \
  --batch-size 4
```

This writes:

- `results/base_qwen7b_strict/test/summary.json`
- `results/base_qwen7b_strict/test/group_metrics.csv`
- `results/base_qwen7b_strict/test/variant_predictions.csv`

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
