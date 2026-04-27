# InvariantForecasting
data-prepping so far

## ForecastBench/current data
https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets
post August resolution dates (598 unique)

```bash
python3 scripts/fetch_forecastbench_current.py --since 2025-08-31 --max-questions 2000
```
so far limited to 598 (we can look for other data)

Outputs:
- `data/raw/forecastbench_current_after_2025-08-31.jsonl`
- `data/processed/forecastbench_current_after_2025-08-31.csv`


BIG NOTES: I EMAILED METACULUS WE CAN GET THEIR DATA INSTEAD
Open the processed CSV with a spreadsheet app/importer, or use the raw JSONL for programmatic work.


