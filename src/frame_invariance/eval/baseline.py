"""Baseline evaluator for the v2 grouped forecasting dataset.

The evaluator reads ``data/processed/training.jsonl`` rows produced by
``frame_invariance.data.build_training_set`` and writes three artifacts:

  - ``variant_predictions.csv``: one row per paraphrase variant
  - ``group_metrics.csv``: one row per underlying question group
  - ``summary.json``: aggregate Brier, log-loss, ParaSD, FRIE, and coverage

It supports cheap non-model baselines (constant, train-base-rate, context-base-rate),
an OpenAI-compatible chat-completions mode, and a Tinker SamplingClient mode for
base model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import compute_metrics, parse_probability


DEFAULT_INPUT = "data/processed/training.jsonl"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_CACHE_DIR = "data/cache/baseline"
DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_TINKER_BASE_MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0


@dataclass(frozen=True)
class EvalConfig:
    input_path: Path
    split: str
    run_name: str
    results_dir: Path
    mode: str
    model: str
    constant_prob: float
    limit_groups: int | None
    max_workers: int
    max_tokens: int
    temperature: float
    top_p: float
    allow_loose_parse: bool
    cache_dir: Path | None
    no_cache: bool
    api_key_env: str
    base_url: str | None
    tinker_api_key_env: str
    tinker_base_model: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    limit_groups: int | None = None,
) -> list[dict[str, Any]]:
    selected = [r for r in rows if r.get("split") == split]
    selected.sort(key=lambda r: (str(r.get("id")), int(r.get("variant_index", 0))))
    if limit_groups is None:
        return selected
    keep_ids: list[str] = []
    seen: set[str] = set()
    for row in selected:
        qid = str(row.get("id"))
        if qid in seen:
            continue
        seen.add(qid)
        keep_ids.append(qid)
        if len(keep_ids) >= limit_groups:
            break
    keep = set(keep_ids)
    return [r for r in selected if str(r.get("id")) in keep]


def train_base_rate(rows: list[dict[str, Any]]) -> float:
    train_groups: dict[str, int] = {}
    for row in rows:
        if row.get("split") != "train":
            continue
        train_groups[str(row["id"])] = int(row["outcome"])
    if not train_groups:
        raise ValueError("cannot compute train-base-rate baseline: no train groups found")
    return sum(train_groups.values()) / len(train_groups)


def prediction_for_row(
    row: dict[str, Any], *, mode: str, constant_prob: float
) -> tuple[str, float | None]:
    if mode == "constant":
        return f"Probability: {constant_prob:.6f}", constant_prob
    if mode == "context-base-rate":
        raw = (row.get("base_rate") or {}).get("value")
        try:
            prob = float(raw)
        except (TypeError, ValueError):
            return "", None
        if 0.0 <= prob <= 1.0:
            return f"Probability: {prob:.6f}", prob
        return "", None
    raise ValueError(f"prediction_for_row does not support mode {mode!r}")


class OpenAIChatPredictor:
    """Small OpenAI-compatible chat client with content-addressed caching."""

    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        base_url: str | None,
        cache_dir: Path | None,
        use_cache: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package is not installed; run `pip install -e .`") from exc

        api_key = os.environ.get(api_key_env)
        if not api_key and not base_url:
            raise RuntimeError(
                f"${api_key_env} is not set. For a local OpenAI-compatible server, "
                "pass --base-url and set a dummy key if the server requires one."
            )
        # Some local OpenAI-compatible servers require a non-empty key but ignore it.
        kwargs: dict[str, Any] = {"api_key": api_key or "local"}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def complete(self, messages: list[dict[str, str]]) -> str:
        request = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        cache_path = self._cache_path(request)
        if self.use_cache and cache_path is not None and cache_path.exists():
            try:
                return str(json.loads(cache_path.read_text(encoding="utf-8")).get("text", ""))
            except json.JSONDecodeError:
                pass

        try:
            response = self.client.chat.completions.create(**request)
        except TypeError:
            # Newer OpenAI models may prefer max_completion_tokens.
            request2 = dict(request)
            request2["max_completion_tokens"] = request2.pop("max_tokens")
            response = self.client.chat.completions.create(**request2)
        text = response.choices[0].message.content or ""
        if self.use_cache and cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_path)
        return text

    def _cache_path(self, request: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


class TinkerSamplingPredictor:
    """Tinker base-model sampler with the same completion interface as API mode."""

    def __init__(
        self,
        *,
        model: str,
        base_model: str,
        api_key_env: str,
        cache_dir: Path | None,
        use_cache: bool,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"${api_key_env} is not set; export your Tinker API key first")
        # The Tinker SDK reads TINKER_API_KEY directly. Mirror custom env names into
        # the canonical name so --tinker-api-key-env works without surprising users.
        os.environ.setdefault("TINKER_API_KEY", api_key)

        try:
            import tinker  # type: ignore
            from tinker import types  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "tinker package is not installed; run `pip install -e '.[tinker]'`"
            ) from exc

        self.model = model
        self.base_model = base_model
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.types = types
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        service_client = tinker.ServiceClient()
        if model.startswith("tinker://"):
            self.sampling_client = service_client.create_sampling_client(model_path=model)
            # Tinker sampler metadata for saved weights may expose a tinker URI as
            # the base model. Use the original base-model tokenizer explicitly.
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise RuntimeError("transformers is required for Tinker checkpoint eval") from exc
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, fast=True)
        else:
            self.sampling_client = service_client.create_sampling_client(base_model=model)
            self.tokenizer = self.sampling_client.get_tokenizer()

    def complete(self, messages: list[dict[str, str]]) -> str:
        prompt_text = render_chat_prompt(self.tokenizer, messages)
        request = {
            "provider": "tinker",
            "model": self.model,
            "base_model": self.base_model,
            "prompt": prompt_text,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        cache_path = self._cache_path(request)
        if self.use_cache and cache_path is not None and cache_path.exists():
            try:
                return str(json.loads(cache_path.read_text(encoding="utf-8")).get("text", ""))
            except json.JSONDecodeError:
                pass

        token_ids = self.tokenizer.encode(prompt_text)
        prompt = self.types.ModelInput.from_ints(token_ids)
        params = self.types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        result = self.sampling_client.sample(
            prompt=prompt,
            sampling_params=params,
            num_samples=1,
        ).result()
        sequence = first_tinker_sequence(result)
        text = decode_tokens(self.tokenizer, sequence.tokens)
        if self.use_cache and cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_path)
        return text

    def _cache_path(self, request: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render chat messages for Tinker token sampling."""

    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered)
    chunks: list[str] = []
    for message in messages:
        role = message["role"].strip().upper()
        chunks.append(f"{role}:\n{message['content'].strip()}")
    chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def first_tinker_sequence(result: Any) -> Any:
    sequences = getattr(result, "sequences", None)
    if sequences:
        return sequences[0]
    samples = getattr(result, "samples", None)
    if samples:
        return samples[0]
    raise RuntimeError("Tinker sample response did not contain sequences or samples")


def decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    try:
        return str(tokenizer.decode(tokens, skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(tokens))


def evaluate(config: EvalConfig) -> tuple[dict[str, Any], Path, Path, Path]:
    all_rows = read_jsonl(config.input_path)
    rows = filter_rows(all_rows, split=config.split, limit_groups=config.limit_groups)
    if not rows:
        raise ValueError(
            f"no rows found for split={config.split!r} in {config.input_path}"
        )

    mode = config.mode
    constant_prob = config.constant_prob
    if mode == "train-base-rate":
        constant_prob = train_base_rate(all_rows)
        mode = "constant"

    run_dir = config.results_dir / config.run_name / config.split
    run_dir.mkdir(parents=True, exist_ok=True)
    variant_path = run_dir / "variant_predictions.csv"
    group_path = run_dir / "group_metrics.csv"
    summary_path = run_dir / "summary.json"

    started_at = time.time()
    if mode in {"constant", "context-base-rate"}:
        prediction_rows = []
        for row in rows:
            completion, prediction = prediction_for_row(
                row, mode=mode, constant_prob=constant_prob
            )
            prediction_rows.append(
                make_prediction_row(
                    row,
                    completion=completion,
                    prediction=prediction,
                    mode=config.mode,
                    model=config.model,
                )
            )
    elif mode == "api":
        predictor = OpenAIChatPredictor(
            model=config.model,
            api_key_env=config.api_key_env,
            base_url=config.base_url,
            cache_dir=config.cache_dir,
            use_cache=not config.no_cache,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        prediction_rows = run_api_predictions(
            rows,
            predictor=predictor,
            config=config,
        )
    elif mode == "tinker":
        predictor = TinkerSamplingPredictor(
            model=config.model,
            base_model=config.tinker_base_model,
            api_key_env=config.tinker_api_key_env,
            cache_dir=config.cache_dir,
            use_cache=not config.no_cache,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        prediction_rows = run_api_predictions(
            rows,
            predictor=predictor,
            config=config,
        )
    else:
        raise ValueError(f"unknown mode {config.mode!r}")

    bundle = compute_metrics(prediction_rows)
    summary = dict(bundle.summary)
    summary.update(
        {
            "run_name": config.run_name,
            "split": config.split,
            "mode": config.mode,
            "model": config.model,
            "input": str(config.input_path),
            "elapsed_s": round(time.time() - started_at, 3),
            "limit_groups": config.limit_groups,
        }
    )
    if config.mode == "train-base-rate":
        summary["train_base_rate"] = constant_prob

    write_csv(variant_path, prediction_rows, VARIANT_FIELDS)
    write_csv(group_path, bundle.group_rows, GROUP_FIELDS)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary, variant_path, group_path, summary_path


def run_api_predictions(
    rows: list[dict[str, Any]],
    *,
    predictor: OpenAIChatPredictor | TinkerSamplingPredictor,
    config: EvalConfig,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(predictor.complete, normalize_messages(row.get("messages") or [])): row
            for row in rows
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                completion = future.result()
            except Exception as exc:
                completion = f"ERROR: {type(exc).__name__}: {exc}"
                prediction = None
            else:
                prediction = parse_probability(
                    completion, allow_loose=config.allow_loose_parse
                )
            out.append(
                make_prediction_row(
                    row,
                    completion=completion,
                    prediction=prediction,
                    mode=config.mode,
                    model=config.model,
                )
            )
            if idx % 25 == 0 or idx == len(rows):
                print(f"predicted {idx}/{len(rows)} variants", file=sys.stderr)
    out.sort(key=lambda r: (str(r["id"]), int(r["variant_index"])))
    return out


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip()
        content = str(msg.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            continue
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("row has no usable chat messages")
    return normalized


def make_prediction_row(
    row: dict[str, Any],
    *,
    completion: str,
    prediction: float | None,
    mode: str,
    model: str,
) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "variant_index": int(row.get("variant_index", 0)),
        "split": row.get("split"),
        "outcome": int(row.get("outcome")),
        "prediction": prediction,
        "parseable": prediction is not None,
        "completion": completion,
        "mode": mode,
        "model": model,
        "question": row.get("question", ""),
    }


VARIANT_FIELDS = [
    "id",
    "variant_index",
    "split",
    "outcome",
    "prediction",
    "parseable",
    "mode",
    "model",
    "question",
    "completion",
]
GROUP_FIELDS = [
    "id",
    "split",
    "outcome",
    "n_variants",
    "n_parseable",
    "coverage",
    "mean_prob",
    "parasd",
    "mean_brier",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    parser = argparse.ArgumentParser(description="Run baseline forecasting evaluation.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--mode",
        choices=("api", "tinker", "constant", "train-base-rate", "context-base-rate"),
        default="api",
        help="api/tinker call a base model; other modes are deterministic sanity baselines.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--constant-prob", type=float, default=0.5)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--allow-loose-parse",
        action="store_true",
        help="Fallback to first number if no Probability: line is found.",
    )
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--tinker-api-key-env", default="TINKER_API_KEY")
    parser.add_argument(
        "--tinker-base-model",
        default=DEFAULT_TINKER_BASE_MODEL,
        help="Base model tokenizer to use when --model is a tinker:// checkpoint URI.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible endpoint, e.g. a local/vLLM server.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.constant_prob <= 1.0:
        raise SystemExit("--constant-prob must be in [0, 1]")
    return EvalConfig(
        input_path=Path(args.input),
        split=args.split,
        run_name=args.run_name,
        results_dir=Path(args.results_dir),
        mode=args.mode,
        model=args.model,
        constant_prob=args.constant_prob,
        limit_groups=args.limit_groups,
        max_workers=max(1, args.max_workers),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        allow_loose_parse=args.allow_loose_parse,
        cache_dir=None if args.cache_dir in {"", "none", "None"} else Path(args.cache_dir),
        no_cache=args.no_cache,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        tinker_api_key_env=args.tinker_api_key_env,
        tinker_base_model=args.tinker_base_model,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    summary, variant_path, group_path, summary_path = evaluate(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"variant predictions: {variant_path}")
    print(f"group metrics:       {group_path}")
    print(f"summary:             {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
