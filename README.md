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

Big note: I emailed Metaculus, so we may be able to use their data later.

Open the processed CSV with a spreadsheet app/importer, or use the raw JSONL for programmatic work.
