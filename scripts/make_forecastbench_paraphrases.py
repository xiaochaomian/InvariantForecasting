#!/usr/bin/env python3
"""Create deterministic paraphrase packs for ForecastBench binary questions.

The output is long-form JSONL: each source question contributes five adjacent
rows, with variant_index=0 holding the original question and 1..4 holding
paraphrases. Keeping the rows adjacent makes line_number // 5 a valid group
lookup while retaining explicit IDs for safer downstream joins.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/raw/forecastbench_current_after_2025-08-31.jsonl"
DEFAULT_OUTPUT = "data/paraphrased/forecastbench_current_after_2025-08-31_5x.jsonl"
DEFAULT_CSV_OUTPUT = "data/processed/forecastbench_current_after_2025-08-31_5x_paraphrases.csv"

RESOLUTION_DATE = "{resolution_date}"
FORECAST_DUE_DATE = "{forecast_due_date}"
CONTEXT_FIELDS = [
    "background",
    "resolution_criteria",
    "url",
    "forecastbench_question_set",
    "forecastbench_resolution_snapshot",
    "freeze_datetime",
    "market_info_open_datetime",
    "market_info_close_datetime",
    "market_info_resolution_criteria",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 5x paraphrase JSONL for ForecastBench rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    return parser.parse_args()


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def date_part(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return text[:10]


def render_question_dates(question: str, row: dict[str, Any]) -> str:
    rendered = question
    resolution_date = date_part(row.get("resolved_at"))
    forecast_due_date = date_part(row.get("freeze_datetime"))
    if RESOLUTION_DATE in rendered:
        if resolution_date is None:
            raise ValueError(f"Missing resolved_at for placeholder question {row.get('id')!r}")
        rendered = rendered.replace(RESOLUTION_DATE, resolution_date)
    if FORECAST_DUE_DATE in rendered:
        if forecast_due_date is None:
            raise ValueError(f"Missing freeze_datetime for placeholder question {row.get('id')!r}")
        rendered = rendered.replace(FORECAST_DUE_DATE, forecast_due_date)
    if "{" in rendered and "}" in rendered:
        raise ValueError(f"Unresolved placeholder in question for {row.get('id')!r}: {rendered!r}")
    return rendered


def normalize_record_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean(value)
    return value


def strip_question_mark(text: str) -> str:
    return clean(text).rstrip("?").strip()


def stock_price_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^Will (.+?)'s market close price on \{resolution_date\} be higher than its market close price on \{forecast_due_date\}\?",
        question,
        flags=re.DOTALL,
    )
    if not match or "Stock splits and reverse splits" not in question:
        return None

    ticker = match.group(1)
    return [
        (
            f"On {RESOLUTION_DATE}, will {ticker}'s split-adjusted market close price exceed "
            f"its market close price on {FORECAST_DUE_DATE}; if the company was delisted through "
            "a merger or bankruptcy, use its final close price?"
        ),
        (
            f"Will {ticker} finish the market day on {RESOLUTION_DATE} above its close on "
            f"{FORECAST_DUE_DATE}, accounting for stock splits, reverse splits, and final-close "
            "resolution for merger or bankruptcy delistings?"
        ),
        (
            f"Comparing adjusted closing prices, is {ticker}'s market close on {RESOLUTION_DATE} "
            f"higher than its market close on {FORECAST_DUE_DATE}, with delisted companies resolved "
            "to their final close price?"
        ),
        (
            f"Does {ticker}'s market close price on {RESOLUTION_DATE}, after any stock split or reverse "
            f"split adjustment, come in above the close on {FORECAST_DUE_DATE}, using the last close "
            "for merger or bankruptcy delistings?"
        ),
    ]


def macro_increase_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^Will (.+) have increased by \{resolution_date\} as compared to its value on \{forecast_due_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    subject = match.group(1)
    return [
        f"As of {RESOLUTION_DATE}, will {subject} be higher than its value on {FORECAST_DUE_DATE}?",
        f"Will {subject} show an increase from {FORECAST_DUE_DATE} to {RESOLUTION_DATE}?",
        f"On {RESOLUTION_DATE}, is {subject} greater than it was on {FORECAST_DUE_DATE}?",
        f"Does {subject} rise between {FORECAST_DUE_DATE} and {RESOLUTION_DATE}?",
    ]


def weather_temperature_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^What is the probability that the daily average temperature at the French weather station at (.+) "
        r"will be higher on \{resolution_date\} than on \{forecast_due_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    station = match.group(1)
    return [
        (
            f"What probability should be assigned to the daily average temperature at the French "
            f"weather station at {station} exceeding its {FORECAST_DUE_DATE} value on {RESOLUTION_DATE}?"
        ),
        (
            f"How likely is it that the French weather station at {station} records a higher daily "
            f"average temperature on {RESOLUTION_DATE} than on {FORECAST_DUE_DATE}?"
        ),
        (
            f"What is the chance that, at the French weather station at {station}, the daily mean "
            f"temperature on {RESOLUTION_DATE} is above the daily mean on {FORECAST_DUE_DATE}?"
        ),
        (
            f"What probability corresponds to {station}'s French weather-station daily average "
            f"temperature being higher on {RESOLUTION_DATE} than on {FORECAST_DUE_DATE}?"
        ),
    ]


def wiki_vaccine_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^According to Wikipedia, will a vaccine have been developed for (.+) by \{resolution_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    disease = match.group(1)
    return [
        f"Using Wikipedia as the source, will there be a developed vaccine for {disease} by {RESOLUTION_DATE}?",
        f"Will Wikipedia document that a vaccine exists for {disease} by {RESOLUTION_DATE}?",
        f"Will Wikipedia indicate that a vaccine has been developed for {disease} no later than {RESOLUTION_DATE}?",
        f"As recorded on Wikipedia, has a vaccine for {disease} been developed by {RESOLUTION_DATE}?",
    ]


def wiki_elo_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^According to Wikipedia, will (.+) have an Elo rating on \{resolution_date\} "
        r"that's at least 1% higher than on \{forecast_due_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    player = match.group(1)
    return [
        (
            f"Using Wikipedia, will {player}'s Elo rating on {RESOLUTION_DATE} be at least 1% above "
            f"their Elo rating on {FORECAST_DUE_DATE}?"
        ),
        (
            f"On Wikipedia, is {player}'s {RESOLUTION_DATE} Elo rating at least 101% of "
            f"their {FORECAST_DUE_DATE} Elo rating?"
        ),
        (
            f"Will Wikipedia show {player}'s {RESOLUTION_DATE} Elo rating as 1% or more above "
            f"the rating listed for {FORECAST_DUE_DATE}?"
        ),
        (
            f"By Wikipedia's record, is {player}'s Elo rating on {RESOLUTION_DATE} no less than "
            f"1% higher than it was on {FORECAST_DUE_DATE}?"
        ),
    ]


def wiki_fide_rank_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^According to Wikipedia, will (.+) have a FIDE ranking on \{resolution_date\} "
        r"as high or higher than their ranking on \{forecast_due_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    player = match.group(1)
    return [
        (
            f"Using Wikipedia as the source, will {player}'s FIDE ranking on {RESOLUTION_DATE} "
            f"be as high as or higher than their ranking on {FORECAST_DUE_DATE}?"
        ),
        (
            f"According to Wikipedia, is {player} ranked at least as highly by FIDE on "
            f"{RESOLUTION_DATE} as they were on {FORECAST_DUE_DATE}?"
        ),
        (
            f"Will Wikipedia show {player} with a FIDE ranking on {RESOLUTION_DATE} that matches "
            f"or exceeds their ranking on {FORECAST_DUE_DATE}?"
        ),
        (
            f"By Wikipedia's record, does {player}'s FIDE ranking on {RESOLUTION_DATE} stand as "
            f"high or higher than the ranking recorded on {FORECAST_DUE_DATE}?"
        ),
    ]


def wiki_record_paraphrases(question: str) -> list[str] | None:
    match = re.match(
        r"^According to Wikipedia, will (.+) still hold the world record for (.+) on \{resolution_date\}\?$",
        clean(question),
    )
    if not match:
        return None

    athlete, event = match.groups()
    return [
        (
            f"Using Wikipedia as the source, will {athlete} remain the world-record holder for "
            f"{event} on {RESOLUTION_DATE}?"
        ),
        (
            f"According to Wikipedia, does {athlete} still possess the {event} world record on "
            f"{RESOLUTION_DATE}?"
        ),
        (
            f"Will Wikipedia list {athlete} as continuing to hold the world record for {event} "
            f"as of {RESOLUTION_DATE}?"
        ),
        (
            f"By Wikipedia's record on {RESOLUTION_DATE}, is {athlete} still the holder of the "
            f"world record for {event}?"
        ),
    ]


def special_case_paraphrases(question: str) -> list[str] | None:
    q = clean(question)
    special: dict[str, list[str]] = {
        "Military conflict between the US and Venezuela in 2025?": [
            "Will there be a military conflict between the US and Venezuela during 2025?",
            "In 2025, does a US-Venezuela military conflict occur?",
            "Will the United States and Venezuela enter into military conflict in 2025?",
            "Does 2025 see military conflict involving both the US and Venezuela?",
        ],
        "Half-Life 3 confirmed by EOY?": [
            "Will Half-Life 3 be confirmed by the end of the year?",
            "By EOY, will there be confirmation of Half-Life 3?",
            "Does Half-Life 3 receive confirmation before the year ends?",
            "Will the end of the year arrive with Half-Life 3 having been confirmed?",
        ],
        "Gemini 3.0 Flash released by December 31?": [
            "Will Gemini 3.0 Flash be released by December 31?",
            "By December 31, does Gemini 3.0 Flash get released?",
            "Will the release of Gemini 3.0 Flash occur no later than December 31?",
            "Is Gemini 3.0 Flash released on or before December 31?",
        ],
        "Was synthetic video data generated and used in training Sora?": [
            "Did Sora's training use generated synthetic video data?",
            "Was generated synthetic video data part of the data used to train Sora?",
            "Did the training process for Sora include synthetic video data that had been generated?",
            "Is it true that Sora was trained using generated synthetic video data?",
        ],
        "US x Venezuela military engagement by November 30?": [
            "Will there be a military engagement between the US and Venezuela by November 30?",
            "By November 30, does a US-Venezuela military engagement occur?",
            "Will the United States and Venezuela have a military engagement on or before November 30?",
            "Does a military engagement involving both the US and Venezuela happen by November 30?",
        ],
        "Tesla launches unsupervised full self driving (FSD) by October 31?": [
            "Will Tesla launch unsupervised full self driving (FSD) by October 31?",
            "By October 31, does Tesla release unsupervised full self driving (FSD)?",
            "Will Tesla's unsupervised FSD launch happen on or before October 31?",
            "Is unsupervised full self driving from Tesla launched no later than October 31?",
        ],
        "Fact check: Was Tyler Robinson a lone actor?": [
            "Fact check: Did Tyler Robinson act alone?",
            "Was Tyler Robinson the sole actor, according to the fact check?",
            "Does the fact check conclude that Tyler Robinson acted by himself?",
            "Is Tyler Robinson found to have been a lone actor?",
        ],
        "Israel x Iran ceasefire broken by October 31?": [
            "Will the Israel-Iran ceasefire be broken by October 31?",
            "By October 31, does the ceasefire between Israel and Iran break?",
            "Will Israel and Iran's ceasefire fail on or before October 31?",
            "Is the Israel-Iran ceasefire violated by October 31?",
        ],
        "Fed decreases interest rates by 25 bps after October 2025 meeting?": [
            "Will the Fed decrease interest rates by 25 bps after the October 2025 meeting?",
            "After its October 2025 meeting, does the Fed cut interest rates by 25 bps?",
            "Will a 25-basis-point Fed interest-rate decrease follow the October 2025 meeting?",
            "Does the Fed lower rates by 25 bps after the October 2025 meeting?",
        ],
        "Will there be a US recession by EOY2025?": [
            "Will a US recession occur by EOY2025?",
            "By EOY2025, will the US enter a recession?",
            "Does the US experience a recession by EOY2025?",
            "Is there a US recession on or before EOY2025?",
        ],
        "Will there be another deadly clash between Thailand and Cambodia, resulting in three or more fatalities, before 2026?": [
            "Before 2026, will Thailand and Cambodia have another deadly clash resulting in three or more fatalities?",
            "Will another Thailand-Cambodia clash causing at least three deaths occur before 2026?",
            "Does a further deadly Thailand-Cambodia clash with three or more fatalities happen before 2026?",
            "By the start of 2026, is there another clash between Thailand and Cambodia that results in at least three deaths?",
        ],
        "Will there be a stronger hurricane than Erin during the 2025 Atlantic hurricane season?": [
            "Will the 2025 Atlantic hurricane season produce a hurricane whose intensity exceeds Erin's?",
            "Will any Atlantic hurricane in the 2025 season be stronger than Erin?",
            "Does the 2025 Atlantic hurricane season produce a hurricane stronger than Erin?",
            "Is Erin surpassed by another hurricane during the 2025 Atlantic hurricane season?",
        ],
    }
    return special.get(q)


def pattern_will_paraphrases(question: str) -> list[str] | None:
    q = clean(question).rstrip("?").strip()

    patterns = [
        (
            r"^Will (.+) have the top AI model on December 31$",
            lambda m: [
                f"On December 31, will {m[1]} hold the top AI model position?",
                f"Will {m[1]} be the company with the top AI model on December 31?",
                f"Does {m[1]} have the leading AI model as of December 31?",
                f"By the December 31 check, is the top AI model from {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) have the second best AI model at the end of December 2025$",
            lambda m: [
                f"At the end of December 2025, will {m[1]} have the second-best AI model?",
                f"Will {m[1]} hold second place among AI models at the close of December 2025?",
                f"By the end of December 2025, is {m[1]}'s AI model ranked second best?",
                f"Does {m[1]} finish December 2025 with the second-best AI model?",
            ],
        ),
        (
            r"^Will (.+) have the best AI model for math at the end of 2025$",
            lambda m: [
                f"Will {m[1]} finish 2025 with the top-performing math AI model?",
                f"Will {m[1]}'s AI model be the best for math by year-end 2025?",
                f"Does {m[1]} finish 2025 with the top math-focused AI model?",
                f"By the end of 2025, is the best AI model for math from {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) have the best AI model at the end of November 2025$",
            lambda m: [
                f"Will {m[1]} finish November 2025 with the top-ranked AI model?",
                f"Will {m[1]} hold the top AI model spot when November 2025 ends?",
                f"By the close of November 2025, is the leading AI model from {m[1]}?",
                f"Does {m[1]} finish November 2025 with the best AI model?",
            ],
        ),
        (
            r"^Will (.+) have the best AI model on October 31$",
            lambda m: [
                f"At the October 31 check, is the leading AI model from {m[1]}?",
                f"Will {m[1]} hold the top AI model position on October 31?",
                f"As of October 31, is the best AI model from {m[1]}?",
                f"Does {m[1]} have the leading AI model at the October 31 check?",
            ],
        ),
        (
            r"^Will (.+) win the 2025 National Heads-Up Poker Championship$",
            lambda m: [
                f"Will {m[1]} be the winner of the 2025 National Heads-Up Poker Championship?",
                f"Does {m[1]} win the 2025 National Heads-Up Poker Championship?",
                f"Is {m[1]} the 2025 National Heads-Up Poker Championship champion?",
                f"Will the 2025 National Heads-Up Poker Championship be won by {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) win Best Makeup and Hairstyling at the 98th Academy Awards$",
            lambda m: [
                f"Will {m[1]} receive the 98th Academy Awards Oscar for Best Makeup and Hairstyling?",
                f"Does {m[1]} win the 98th Academy Awards category for Best Makeup and Hairstyling?",
                f"At the 98th Academy Awards, is Best Makeup and Hairstyling awarded to {m[1]}?",
                f"Will the Best Makeup and Hairstyling Oscar at the 98th Academy Awards go to {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) be TIME's Person of the Year for 2025$",
            lambda m: [
                f"Will {m[1]} be selected as TIME's 2025 Person of the Year?",
                f"Does TIME name {m[1]} its Person of the Year for 2025?",
                f"Is {m[1]} TIME's Person of the Year in 2025?",
                f"Will {m[1]} receive TIME's Person of the Year title for 2025?",
            ],
        ),
        (
            r"^Will (.+) win (.+)$",
            lambda m: [
                f"Will {m[1]} come out on top in {m[2]}?",
                f"Does {m[2]} end with {m[1]} as the winner?",
                f"Will victory in {m[2]} go to {m[1]}?",
                f"Will {m[2]} be won by {m[1]}?",
            ],
        ),
        (
            r"^Will there be (.+)$",
            lambda m: [
                f"Does {m[1]} occur?",
                f"Will {m[1]} take place?",
                f"Is there {m[1]}?",
                f"Does the outcome include {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) reach (\$[0-9,kK]+) in (.+)$",
            lambda m: [
                f"During {m[3]}, does {m[1]}'s price get to at least {m[2]}?",
                f"Does {m[1]} hit or exceed {m[2]} during {m[3]}?",
                f"Will {m[1]} trade at or above {m[2]} in {m[3]}?",
                f"By the end of {m[3]}, has {m[1]} reached at least {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) close at (\$[0-9,]+[–-][0-9,]+) in (.+)$",
            lambda m: [
                f"On the final trading day of {m[3]}, will {m[1]}'s official close be in the {m[2]} range?",
                f"Does {m[1]}'s final {m[3]} official closing price fall in the {m[2]} bracket?",
                f"Will {m[1]} end {m[3]} with a close in the {m[2]} band?",
                f"Is {m[1]}'s year-end {m[3]} close within the {m[2]} range?",
            ],
        ),
        (
            r"^Will (.+) dip to (\$[0-9,kK]+) by (.+)$",
            lambda m: [
                f"By {m[3]}, will {m[1]} fall to {m[2]} or lower?",
                f"Does {m[1]} drop to {m[2]} or below on or before {m[3]}?",
                f"Will {m[1]} touch {m[2]} or lower by {m[3]}?",
                f"Is {m[1]} at {m[2]} or below before {m[3]} ends?",
            ],
        ),
        (
            r"^Will (.+) dip to (\$[0-9,kK]+) in (.+)$",
            lambda m: [
                f"In {m[3]}, will {m[1]} fall to {m[2]} or lower?",
                f"Does {m[1]} drop to {m[2]} or below during {m[3]}?",
                f"Will {m[1]} touch {m[2]} or lower in {m[3]}?",
                f"By the end of {m[3]}, has {m[1]} dipped to {m[2]} or below?",
            ],
        ),
        (
            r"^Will (.+) dip below (\$[0-9,kK]+) before (.+)$",
            lambda m: [
                f"Before {m[3]}, will {m[1]} fall below {m[2]}?",
                f"Does {m[1]} trade under {m[2]} before {m[3]}?",
                f"Will {m[1]} move below {m[2]} prior to {m[3]}?",
                f"Is there a pre-{m[3]} dip in {m[1]} below {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) be the top Spotify artist for 2025$",
            lambda m: [
                f"Will {m[1]} finish 2025 as Spotify's top artist?",
                f"Does {m[1]} become the top Spotify artist of 2025?",
                f"For 2025, is Spotify's leading artist {m[1]}?",
                f"Will {m[1]} rank first among Spotify artists in 2025?",
            ],
        ),
        (
            r"^Will '(.+)' be #1 for ([0-9]+) straight weeks$",
            lambda m: [
                f"Will '{m[1]}' spend {m[2]} consecutive weeks at #1?",
                f"Does '{m[1]}' hold the #1 spot for {m[2]} weeks in a row?",
                f"Is '{m[1]}' number one for {m[2]} straight weeks?",
                f"Will '{m[1]}' achieve a {m[2]}-week uninterrupted run at #1?",
            ],
        ),
        (
            r"^Will '(.+)' have the best domestic opening weekend in 2025$",
            lambda m: [
                f"Will '{m[1]}' post 2025's best domestic opening weekend?",
                f"Does '{m[1]}' have the top domestic opening weekend of 2025?",
                f"In 2025, is the strongest domestic opening weekend from '{m[1]}'?",
                f"Will '{m[1]}' lead all 2025 films in domestic opening-weekend performance?",
            ],
        ),
        (
            r"^Will (.+) be the next Mayor of (.+)$",
            lambda m: [
                f"Will {m[1]} become the next mayor of {m[2]}?",
                f"Is {m[1]} the next person to serve as mayor of {m[2]}?",
                f"Does {m[1]} take office as {m[2]}'s next mayor?",
                f"Does {m[2]}'s next mayor turn out to be {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) be the #2 searched person on Google this year$",
            lambda m: [
                f"Will {m[1]} rank as Google's second-most-searched person this year?",
                f"Does {m[1]} finish this year as the #2 searched person on Google?",
                f"Is {m[1]} the second-ranked person in Google searches this year?",
                f"Will Google search data place {m[1]} at #2 among people this year?",
            ],
        ),
        (
            r"^Will (.+) go live in (.+)$",
            lambda m: [
                f"Will {m[1]} become publicly available in {m[2]}?",
                f"Does {m[1]} launch publicly in {m[2]}?",
                f"Will {m[1]} become live during {m[2]}?",
                f"By the end of {m[2]}, is {m[1]} live?",
            ],
        ),
        (
            r"^Will (.+) perform an airdrop by (.+)$",
            lambda m: [
                f"By {m[2]}, will {m[1]} carry out an airdrop?",
                f"Does {m[1]} perform an airdrop on or before {m[2]}?",
                f"Will an airdrop from {m[1]} happen by {m[2]}?",
                f"Is there a {m[1]} airdrop no later than {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) score more points than (.+) in (.+)$",
            lambda m: [
                f"In {m[3]}, will {m[1]} outscore {m[2]}?",
                f"Is {m[1]}'s point total greater than {m[2]}'s in {m[3]}?",
                f"Will {m[1]}'s point total exceed {m[2]}'s in {m[3]}?",
                f"By the end of {m[3]}, has {m[1]} scored more points than {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) cut (.+) in (.+)$",
            lambda m: [
                f"During {m[3]}, will {m[1]} make cuts to {m[2]}?",
                f"Will {m[2]} be reduced by {m[1]} during {m[3]}?",
                f"Will {m[2]} be cut by {m[1]} in {m[3]}?",
                f"By the end of {m[3]}, has {m[1]} cut {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) result in at least ([0-9]+ deaths .+)$",
            lambda m: [
                f"Will {m[1]} lead to at least {m[2]}?",
                f"Does {m[1]} produce at least {m[2]}?",
                f"Is the result of {m[1]} at least {m[2]}?",
                f"Are at least {m[2]} attributable to {m[1]}?",
            ],
        ),
        (
            r"^Will (.+) find (.+) before (.+)$",
            lambda m: [
                f"Will {m[1]} identify {m[2]} before {m[3]}?",
                f"Does {m[1]} find {m[2]} prior to {m[3]}?",
                f"Will {m[2]} be found by {m[1]} before {m[3]}?",
                f"By the start of {m[3]}, has {m[1]} found {m[2]}?",
            ],
        ),
        (
            r"^Will (.+) make over a billion dollars at the worldwide box office$",
            lambda m: [
                f"Will {m[1]} gross more than $1 billion worldwide?",
                f"Does {m[1]}'s worldwide box office exceed one billion dollars?",
                f"Will global ticket sales for {m[1]} pass $1 billion?",
                f"Is {m[1]} above the $1 billion mark at the worldwide box office?",
            ],
        ),
        (
            r"^Will (.+) finish fifth in (.+)$",
            lambda m: [
                f"Will {m[1]} end up fifth in {m[2]}?",
                f"Does {m[1]} finish in fifth place in {m[2]}?",
                f"Is {m[1]} fifth in the final {m[2]} standings?",
                f"Will fifth place in {m[2]} belong to {m[1]}?",
            ],
        ),
        (
            r"^Will global temperature increase by (.+) in (.+)$",
            lambda m: [
                f"In {m[2]}, will the global temperature increase be {m[1]}?",
                f"Does global temperature rise by {m[1]} in {m[2]}?",
                f"Will {m[2]} show a global temperature increase of {m[1]}?",
                f"Is the global temperature increase during {m[2]} {m[1]}?",
            ],
        ),
    ]
    for pattern, builder in patterns:
        match = re.match(pattern, q)
        if match:
            return builder(match)

    if q == "Will the Fed Cut–Cut–Pause in 2025":
        return [
            "In 2025, will the Fed follow a cut-cut-pause sequence?",
            "Will the Fed's 2025 rate path be cut, then cut, then pause?",
            "Does the Fed cut twice and then pause in 2025?",
            "Will 2025 bring a Fed cut-cut-pause pattern?",
        ]

    fed_sequence = re.match(r"^Will the Fed (cut-pause-pause|cut-pause-cut) in 2025$", q)
    if fed_sequence:
        sequence = fed_sequence.group(1)
        spoken = sequence.replace("-", ", then ")
        return [
            f"In 2025, will the Fed follow a {sequence} sequence?",
            f"Will the Fed's 2025 rate path be {spoken}?",
            f"Does the Fed {spoken} in 2025?",
            f"Will 2025 bring a Fed {sequence} pattern?",
        ]

    cut_match = re.match(r"^Will (\d\+?) Fed rate cuts happen in 2025$", q)
    if cut_match:
        count = cut_match.group(1)
        count_text = f"{count[:-1]} or more" if count.endswith("+") else f"exactly {count}"
        return [
            f"Will there be {count_text} Fed rate cuts in 2025?",
            f"Does the Fed make {count_text} rate cuts during 2025?",
            f"In 2025, do {count_text} Federal Reserve rate cuts occur?",
            f"Will 2025 include {count_text} Fed interest-rate cuts?",
        ]

    return None


VERB_RE = re.compile(
    r"\b(be|have|win|cut|close|reach|dip|perform|go|score|make|decrease|increase|find|result|"
    r"launch|break|happen|finish|hold|get|become)\b",
    flags=re.IGNORECASE,
)


def declarative_from_will(question: str) -> str | None:
    q = strip_question_mark(question)
    if not q.lower().startswith("will "):
        return None

    body = q[5:].strip()
    lower = body.lower()
    if lower.startswith("there be "):
        return "there will be " + body[9:]
    if lower == "there be":
        return "there will be"

    match = VERB_RE.search(body)
    if not match:
        return None

    subject = body[: match.start()].strip()
    predicate = body[match.start() :].strip()
    if not subject or not predicate:
        return None
    return clean(f"{subject} will {predicate}")


def generic_will_paraphrases(question: str) -> list[str] | None:
    statement = declarative_from_will(question)
    if statement is None:
        return None

    original_without_q = strip_question_mark(question)
    return [
        f"Is it the case that {statement}?",
        f"Will the claim be true that {statement}?",
        f"Does the outcome resolve yes for the proposition that {statement}?",
        f"Will reality match the statement: {original_without_q}?",
    ]


PARAPHRASERS = [
    stock_price_paraphrases,
    macro_increase_paraphrases,
    weather_temperature_paraphrases,
    wiki_vaccine_paraphrases,
    wiki_elo_paraphrases,
    wiki_fide_rank_paraphrases,
    wiki_record_paraphrases,
    special_case_paraphrases,
    pattern_will_paraphrases,
    generic_will_paraphrases,
]


def paraphrase(question: str) -> tuple[list[str], str]:
    for fn in PARAPHRASERS:
        variants = fn(question)
        if variants is not None:
            if len(variants) != 4:
                raise ValueError(f"{fn.__name__} returned {len(variants)} variants")
            return variants, fn.__name__.replace("_paraphrases", "")
    raise ValueError(f"No paraphrase rule for question: {question!r}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_index, row in enumerate(rows):
        original = row["question"]
        generated, method = paraphrase(original)
        group = [original, *generated]

        if len(set(group)) != 5:
            raise ValueError(f"Duplicate variants for {row['id']}: {group!r}")

        for variant_index, text in enumerate(group):
            rendered_question = clean(render_question_dates(text, row))
            record = {
                "source_question_index": group_index,
                "id": row["id"],
                "variant_index": variant_index,
                "variant_id": f"{row['id']}::{variant_index}",
                "is_original": variant_index == 0,
                "question": rendered_question,
                "source": row.get("source"),
                "outcome": row.get("outcome"),
                "resolved_at": row.get("resolved_at"),
                "paraphrase_method": "original" if variant_index == 0 else method,
            }
            for field in CONTEXT_FIELDS:
                record[field] = normalize_record_value(row.get(field))
            records.append(
                record
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_question_index",
        "id",
        "variant_index",
        "variant_id",
        "is_original",
        "question",
        "source",
        "outcome",
        "resolved_at",
        "paraphrase_method",
        *CONTEXT_FIELDS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    args = parse_args()
    rows = read_rows(Path(args.input))
    records = build_records(rows)

    expected = len(rows) * 5
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} records, produced {len(records)}")

    write_jsonl(Path(args.output), records)
    write_csv(Path(args.csv_output), records)
    print(f"questions: {len(rows)}")
    print(f"records:   {len(records)}")
    print(f"jsonl:     {args.output}")
    print(f"csv:       {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
