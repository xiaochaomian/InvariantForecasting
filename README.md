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
