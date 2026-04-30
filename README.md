# InvariantForecasting

Frame-invariant RL post-training for binary LLM forecasting.

This repo is set up for the current ForecastBench pilot:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Data: ForecastBench/current questions resolved after `2025-08-31`
- Paraphrases: 5 variants per question, deterministic rule-based pack
- Splits: 80/10/10 by question group, stratified by resolution month
- Main runs: base eval, GRPO `lambda_invariance=0`, GRPO `lambda_invariance=1`

The full project briefing in `reference/briefing.html` describes the paper-target
protocol: Metaculus scale data, `K=8`, API paraphrasers, NLI filtering, and held-out
paraphraser evaluation. This repo now implements the smaller, reproducible
ForecastBench pilot. Do not describe it as the full paper protocol until those
data/paraphraser pieces are added.

## Data

Fetch ForecastBench/current rows:

```bash
PYTHONPATH=src python3 scripts/fetch_forecastbench_current.py \
  --since 2025-08-31 \
  --max-questions 2000
```

Generate the 5x paraphrase pack:

```bash
PYTHONPATH=src python3 scripts/make_forecastbench_paraphrases.py
```

Outputs:

- `data/raw/forecastbench_current_after_2025-08-31.jsonl`
- `data/processed/forecastbench_current_after_2025-08-31.csv`
- `data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl`
- `data/processed/forecastbench_current_after_2025-08-31_5x_paraphrases.csv`
- `human/questions_and_paraphrases.txt`

Before spending GPU time, run:

```bash
PYTHONPATH=src python3 scripts/audit_forecastbench.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected current audit shape:

- 598 groups / 2,990 rows
- Train: 478 groups
- Validation: 60 groups
- Test: 60 groups
- Forecast dates: `2025-10-16` to `2025-11-27`
- Resolution dates: `2025-10-27` to `2025-12-31`

The audit uses a conservative `2024-06-30` model-cutoff guard. All forecast and
resolution dates are after that, so the current questions are post-cutoff for Qwen2.5
under that guard.

## RunPod Setup

```bash
cd /workspace
git clone https://github.com/xiaochaomian/InvariantForecasting.git
cd InvariantForecasting

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

mkdir -p logs
PYTHONPATH=src python3 scripts/audit_forecastbench.py
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

## Base Eval

```bash
PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench_lambda1.yaml \
  --model Qwen/Qwen2.5-7B-Instruct \
  --run-name base_qwen7b_clean \
  --split test \
  --batch-size 4 \
  2>&1 | tee logs/base_qwen7b_clean_$(date +%Y%m%d_%H%M%S).log
```

This writes `results/base_qwen7b_clean/test/{summary.json,group_metrics.csv,variant_predictions.csv}`.

## GRPO Lambda 0

```bash
PYTHONPATH=src python -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench_lambda0.yaml \
  --preflight

PYTHONPATH=src python -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench_lambda0.yaml \
  2>&1 | tee logs/grpo_lambda0_$(date +%Y%m%d_%H%M%S).log

PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench_lambda0.yaml \
  --model outputs/grpo_forecastbench_qwen7b_lambda0 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --run-name grpo_lambda0 \
  --split test \
  --batch-size 4 \
  2>&1 | tee logs/eval_grpo_lambda0_$(date +%Y%m%d_%H%M%S).log
```

## GRPO Lambda 1

```bash
PYTHONPATH=src python -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench_lambda1.yaml \
  --preflight

PYTHONPATH=src python -m frame_invariance.training.train_grpo \
  --config configs/training/grpo_forecastbench_lambda1.yaml \
  2>&1 | tee logs/grpo_lambda1_$(date +%Y%m%d_%H%M%S).log

PYTHONPATH=src python -m frame_invariance.training.evaluate \
  --config configs/training/grpo_forecastbench_lambda1.yaml \
  --model outputs/grpo_forecastbench_qwen7b_lambda1 \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --run-name grpo_lambda1 \
  --split test \
  --batch-size 4 \
  2>&1 | tee logs/eval_grpo_lambda1_$(date +%Y%m%d_%H%M%S).log
```

## Metrics

Each evaluation summary reports:

- `mean_brier`: Brier on group consensus probability
- `mean_parasd`: probability standard deviation across paraphrases
- `mean_brier_base_rate` and `mean_brier_constant_0_5`: sanity baselines
- `mean_brier_if_labels_inverted`: label-direction diagnostic
- `mean_brier_variant`: per-variant Brier before consensus
- `mean_log_loss` and `ece_10`: calibration hygiene
- `frie_lambda_0`, `frie_lambda_1`, `frie_lambda_5`: combined frontier metrics
- `variant_coverage` and `group_full_coverage`: parser coverage

Use the base eval, lambda 0 eval, and lambda 1 eval summaries for the first clean
comparison table. The old exploratory first-run artifacts were removed because they
were generated before the prompt/date/paraphrase fixes.
