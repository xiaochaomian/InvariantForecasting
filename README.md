# InvariantForecasting (v2)

Frame-invariant RL post-training for LLM forecasting on **gpt-oss-120b via
Tinker**, targeting the ICML 2026 Forecasting workshop. This repo is a clean
rewrite; the v1 pilot (Qwen2.5-7B + rule-based paraphrases) is in git history.

## Status

Day 1 (data pulling), Day 2 (context + paraphrasing + assembly), and the
offline evaluation/training harness are implemented. Tinker GRPO is ready for
a tiny remote smoke test before any scaled run.

| Stage | Module | Status |
|---|---|---|
| Unified question schema | `frame_invariance.data.schema` | ✅ |
| ForecastBench puller (offline, with date rendering and combo filter) | `frame_invariance.data.forecastbench` | ✅ |
| Metaculus / Polymarket / Manifold pullers (deferred) | `frame_invariance.data.{metaculus,polymarket,manifold}` | ✅ written, not run |
| Mantic Q2 2025 AIB test set | `frame_invariance.data.mantic_aib` | ✅ |
| Unifier + cross-source dedupe + audit | `frame_invariance.data.unify` | ✅ |
| Claude client wrapper (cache + retry + backoff) | `frame_invariance.llm.client` | ✅ |
| Context generator (base-rate + dated news, leakage filter) | `frame_invariance.data.context` | ✅ |
| Claude paraphraser (K=8 with entity-preservation guard) | `frame_invariance.data.paraphrase_llm` | ✅ |
| Build training set assembler | `frame_invariance.data.build_training_set` | ✅ |
| Tinker GRPO trainer | `frame_invariance.training.train_tinker` | ✅ implemented, remote smoke pending |
| Eval (Brier + BCC headline, ParaSD appendix) | `frame_invariance.eval.*` | ✅ baseline + coherence benchmark |

Core offline tests cover schema, ForecastBench puller, API pullers, unifier,
Claude client, context generation, paraphrasing, training-set assembly,
baseline evaluation, coherence diagnostics, and GRPO reward helpers.

## Dataset summary (current run)

- **3,471 unified post-cutoff binary questions** from ForecastBench's in-tree
  snapshots (datasets/), all freezes and resolutions strictly after gpt-oss-120b's
  June 2024 cutoff.
- **17,240 training rows** = 3,448 question groups × K=5 (1 original + 4
  paraphrases, with 23 questions dropped for incomplete groups or missing context).
- **Splits:** 2,760 train / 343 validation / 345 test, stratified by
  resolution month, deterministic seed 17.
- **Yes-rate:** 0.229 train / 0.216 validation / 0.197 test.
- **Source mix (FB sub-source):** ACLED 1,196 · Polymarket 868 · yfinance 485 ·
  Wikipedia 338 · Manifold 176 · FRED 165 · Metaculus 132 · DBnomics 53 · INFER 35.

Day 2 cost: ~$135 in Anthropic Sonnet 4.6 calls (context generation + LLM
paraphrasing + retries on entity-guard rejections).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Day 1 — Pull every data source

The four pullers all emit the same ``Question`` schema (see
``src/frame_invariance/data/schema.py``). Run them in any order; they're
independent.

```bash
# 1. ForecastBench — uses the in-tree datasets/ snapshots (no network needed).
PYTHONPATH=src python -m frame_invariance.data.forecastbench

# 2. Metaculus — full pull (post-cutoff, resolved binary). Needs internet.
PYTHONPATH=src python -m frame_invariance.data.metaculus

# 3. Polymarket — Gamma API (post-cutoff, resolved binary, volume >= $1k).
PYTHONPATH=src python -m frame_invariance.data.polymarket

# 4. Manifold — public API (post-cutoff, resolved binary, volume >= M$100).
PYTHONPATH=src python -m frame_invariance.data.manifold

# 5. Mantic Q2 2025 AIB test split — pull via Metaculus tournament slug.
PYTHONPATH=src python -m frame_invariance.data.mantic_aib tournament \
    --slug aibq2 --output data/raw/mantic_q2_aib.jsonl
# OR, if you find Mantic's published id list and drop it at data/raw/mantic_aib_ids.txt:
PYTHONPATH=src python -m frame_invariance.data.mantic_aib drop-in \
    --drop-in data/raw/mantic_aib_ids.txt \
    --metaculus-jsonl data/raw/metaculus.jsonl

# 6. Unify, cross-source dedupe, audit.
PYTHONPATH=src python -m frame_invariance.data.unify \
    --inputs data/raw/forecastbench.jsonl \
             data/raw/metaculus.jsonl \
             data/raw/polymarket.jsonl \
             data/raw/manifold.jsonl \
             data/raw/mantic_q2_aib.jsonl \
    --output data/processed/unified.jsonl \
    --audit-output data/processed/unified_audit.json
```

Caches under ``data/raw/<source>_cache/`` are gitignored and re-runnable. The
unifier re-runs in seconds; the API-backed pullers take 5–30 minutes each
depending on rate-limit luck.

### Current ForecastBench-only baseline

After step 1 alone (offline, no APIs hit):

| Source (within FB) | Count |
|---|---:|
| ACLED | 2,002 |
| Polymarket | 1,592 |
| Yahoo Finance | 1,269 |
| Wikipedia | 969 |
| FRED | 943 |
| DBnomics | 590 |
| Manifold | 496 |
| Metaculus | 356 |
| INFER | 112 |
| **Total** | **8,329** |

Date range: 2024-07-12 → 2026-05-31. Yes-rate: 0.184. All post-cutoff for
gpt-oss-120b (cutoff 2024-06-30 per OpenAI model card).

### Tests

```bash
python -m pytest tests/ -v
```

65 tests covering schema, four pullers, unifier, Claude client, context
generator, and paraphraser; all passing.

## Day 2 — Generate context and paraphrases

These two steps both call Claude Sonnet 4.6 via the Anthropic API. Set
`ANTHROPIC_API_KEY` first.

```bash
export ANTHROPIC_API_KEY=<your-key>

# Smoke test on 5 questions first (fast, cheap, surfaces prompt issues early).
PYTHONPATH=src python -m frame_invariance.data.context \
    --input data/processed/unified.jsonl \
    --output data/processed/contexts_smoke.jsonl \
    --limit 5

PYTHONPATH=src python -m frame_invariance.data.paraphrase_llm \
    --input data/processed/unified.jsonl \
    --output data/processed/paraphrases_smoke.jsonl \
    --limit 5
```

Inspect the smoke outputs (especially the `news_snapshot` dates, which must
all be strictly before each question's `freeze_date`, and the paraphrases,
which must all preserve entities). If they look right:

```bash
# Full run on 8,329 questions. ~$200 in Claude API; both runs cached on disk
# under data/cache/claude/, so re-runs are free.
PYTHONPATH=src python -m frame_invariance.data.context \
    --input data/processed/unified.jsonl \
    --output data/processed/contexts.jsonl \
    --max-workers 8

PYTHONPATH=src python -m frame_invariance.data.paraphrase_llm \
    --input data/processed/unified.jsonl \
    --output data/processed/paraphrases.jsonl \
    --max-workers 8 --k 8
```

Resume support: re-running with the same `--output` skips question ids that
already have a row written. The Claude content cache is content-addressed,
so prompts that haven't changed don't re-fire even if you change unrelated
flags.

## Baseline evaluation

Before any RL run, generate a frozen no-posttraining baseline on the assembled
dataset. This establishes the Brier / log-loss / paraphrase-sensitivity numbers
that Tinker GRPO has to beat.

First assemble the grouped dataset if it is not already present:

```bash
PYTHONPATH=src python -m frame_invariance.data.build_training_set \
    --unified data/processed/unified.jsonl \
    --contexts data/processed/contexts.jsonl \
    --paraphrases data/processed/paraphrases.jsonl \
    --output data/processed/training.jsonl \
    --audit-output data/processed/training_audit.json \
    --k 5
```

Run the cheap sanity baselines first:

```bash
PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name baseline_constant_05_validation \
    --mode constant \
    --constant-prob 0.5

PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name baseline_context_base_rate_validation \
    --mode context-base-rate
```

Then run a small model smoke test. For GPT-OSS-120B, prefer Tinker sampling over
local vLLM if serving is unstable. Tinker reads `TINKER_API_KEY` and handles the
remote GPU worker.

```bash
export TINKER_API_KEY=<your-tinker-key>

PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name base_gpt_oss_120b_tinker_smoke \
    --mode tinker \
    --model openai/gpt-oss-120b \
    --limit-groups 5 \
    --max-workers 1 \
    --temperature 0 \
    --max-tokens 1024
```

If the smoke run has `variant_coverage = 1.0`, run full validation and then
test exactly once:

```bash
PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name base_gpt_oss_120b_tinker_validation \
    --mode tinker \
    --model openai/gpt-oss-120b \
    --max-workers 1 \
    --temperature 0 \
    --max-tokens 1024

PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split test \
    --run-name base_gpt_oss_120b_tinker_test \
    --mode tinker \
    --model openai/gpt-oss-120b \
    --max-workers 1 \
    --temperature 0 \
    --max-tokens 1024
```

The evaluator also supports OpenAI-compatible chat completions, so it works with
the OpenAI SDK against either a hosted endpoint or a local/vLLM server via
`OPENAI_BASE_URL`.

```bash
export OPENAI_API_KEY=<your-key-or-local-placeholder>
# Optional for local/vLLM:
# export OPENAI_BASE_URL=http://127.0.0.1:8000/v1

PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name base_gpt_oss_120b_smoke \
    --mode api \
    --model gpt-oss-120b \
    --limit-groups 5 \
    --max-workers 4 \
    --temperature 0 \
    --max-tokens 256
```

If the OpenAI-compatible smoke run has `variant_coverage = 1.0`, run full
validation and then test exactly once:

```bash
PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split validation \
    --run-name base_gpt_oss_120b_validation \
    --mode api \
    --model gpt-oss-120b \
    --max-workers 4 \
    --temperature 0 \
    --max-tokens 256

PYTHONPATH=src python -m frame_invariance.eval.baseline \
    --input data/processed/training.jsonl \
    --split test \
    --run-name base_gpt_oss_120b_test \
    --mode api \
    --model gpt-oss-120b \
    --max-workers 4 \
    --temperature 0 \
    --max-tokens 256
```

Each run writes `variant_predictions.csv`, `group_metrics.csv`, and
`summary.json` under `results/<run-name>/<split>/`. The headline fields are
`mean_brier`, `mean_log_loss`, `mean_parasd`, `frie_lambda_1`, and coverage.

### Bayesian-coherence benchmark

After a model baseline finishes, run the offline Bayesian-coherence benchmark.
It compares the model posterior to the supplied base-rate prior and reports
whether the model improved Brier/log-loss over that prior, whether updates moved
in the correct direction in hindsight, and how stable log-odds updates were
across paraphrases.

```bash
PYTHONPATH=src python -m frame_invariance.eval.coherence \
    --training data/processed/training.jsonl \
    --predictions results/base_gpt_oss_120b_tinker_validation/validation/variant_predictions.csv
```

This writes `coherence/summary.json` and `coherence/group_coherence.csv` next
to the prediction file. Treat this as a diagnostic benchmark, not as a
replacement for Brier/log-loss or ParaSD.

## Tinker GRPO training

The Tinker trainer is in `frame_invariance.training.train_tinker`. It implements
a conservative grouped GRPO-style loop:

- sample one completion for each paraphrase variant in a question group;
- reward each completion with `-(Brier + lambda * paraphrase_variance_penalty)`;
- normalize rewards within the paraphrase group as GRPO advantages;
- train sampled tokens through Tinker's PPO loss;
- stop early on low parse rate, punctuation loops, or all-zero reward variance.

Always preflight first:

```bash
PYTHONPATH=src python -m frame_invariance.training.train_tinker \
    --config configs/training/tinker_grpo_lambda0.yaml \
    --preflight
```

Then run only the tiny lambda=0 smoke config:

```bash
PYTHONPATH=src python -m frame_invariance.training.train_tinker \
    --config configs/training/tinker_grpo_lambda0.yaml
```

If lambda=0 parses cleanly and writes a checkpoint, run the lambda=1 smoke:

```bash
PYTHONPATH=src python -m frame_invariance.training.train_tinker \
    --config configs/training/tinker_grpo_lambda1.yaml
```

Do not scale `max_steps` or `groups_per_step` until the smoke run metrics show
high `reward_parse_rate`, zero `reward_punctuation_loop_rate`, and no safety
stop. Checkpoints are saved as Tinker sampler weights and recorded in
`results/<run-name>/train/checkpoints.jsonl`.

## What survived the v2 cleanup

`datasets/` (ForecastBench in-tree snapshots, 26 resolution sets and matching
question sets) and `.git`. Everything else (code, configs, paraphrases,
reports) is from the v1 pilot and is no longer referenced.

## License

MIT. See `LICENSE`.
