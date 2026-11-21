from __future__ import annotations

import io
import os
from datetime import date
from typing import Final

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

START_DATE: Final[str] = "2000-01-01"
DATA_CUTOFF: Final[str] = os.getenv("FRED_DATA_CUTOFF", date.today().isoformat())
FRED_URL: Final[str] = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Deliberately small, economically motivated variable universe.
SERIES = {
    "pcepi": {
        "id": "PCEPI",
        "name": "PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "core_pce": {
        "id": "PCEPILFE",
        "name": "Core PCE Price Index",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "shelter": {
        "id": "CUSR0000SAH1",
        "name": "CPI Shelter",
        "source": "BLS",
        "frequency": "Monthly",
    },
    "unemployment": {
        "id": "UNRATE",
        "name": "Unemployment Rate",
        "source": "BLS",
        "frequency": "Monthly",
    },
    "real_consumption": {
        "id": "DPCERA3M086SBEA",
        "name": "Real Personal Consumption Expenditures",
        "source": "BEA",
        "frequency": "Monthly",
    },
    "inflation_expectations": {
        "id": "MICH",
        "name": "1-Year Inflation Expectations",
        "source": "University of Michigan",
        "frequency": "Monthly",
    },
    "oil": {
        "id": "WTISPLC",
        "name": "WTI Crude Oil Price",
        "source": "EIA / FRED",
        "frequency": "Monthly",
    },
}


def _api_key() -> str:
    return os.getenv("FRED_API_KEY", "").strip()


def _monthly_last(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = out.index.to_period("M").to_timestamp()
    return out.groupby(level=0).last()


def fetch_series(series_id: str) -> pd.Series:
    key = _api_key()
    if not key:
        response = requests.get(
            FRED_CSV_URL,
            params={"id": series_id, "cosd": START_DATE, "coed": DATA_CUTOFF},
            timeout=30,
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if frame.empty or len(frame.columns) < 2:
            raise RuntimeError(f"No observations returned for FRED series {series_id}.")
        frame.columns = ["date", "value", *list(frame.columns[2:])]
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        return (
            frame.dropna(subset=["date", "value"])
            .set_index("date")["value"]
            .sort_index()
        )

    params = {
        "api_key": key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START_DATE,
        "observation_end": DATA_CUTOFF,
        "realtime_start": DATA_CUTOFF,
        "realtime_end": DATA_CUTOFF,
    }

    response = requests.get(
        FRED_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    if "error_code" in payload:
        raise RuntimeError(
            f"FRED error {payload['error_code']}: "
            f"{payload.get('error_message', 'Unknown error')}"
        )

    observations = payload.get("observations", [])
    if not observations:
        raise RuntimeError(
            f"No observations returned for FRED series {series_id}."
        )

    frame = pd.DataFrame(observations)
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    )
    frame["value"] = pd.to_numeric(
        frame["value"],
        errors="coerce",
    )

    frame = (
        frame.dropna(
            subset=["date", "value"]
        )
        .set_index("date")
        .sort_index()
    )

    frame = frame[
        ~frame.index.duplicated(
            keep="last"
        )
    ]

    return frame["value"]


def fetch_model_data() -> pd.DataFrame:
    data = {}

    for key, cfg in SERIES.items():
        data[key] = _monthly_last(
            fetch_series(cfg["id"])
        )

    df = pd.concat(
        data,
        axis=1,
    ).sort_index()

    # A single missing month in one series (e.g. a delayed BLS
    # release during a shutdown) otherwise poisons every lagged
    # feature derived from it for up to 12 months downstream,
    # silently truncating the model to stale data. Carry the last
    # known reading forward for genuinely isolated one-month gaps
    # only -- a longer gap is left as NaN rather than fabricated.
    df[list(SERIES.keys())] = (
        df[list(SERIES.keys())].ffill(limit=1)
    )

    # Target.
    df["pce_inflation"] = (
        df["pcepi"]
        .pct_change(12)
        * 100
    )

    # Core and shelter inflation.
    df["core_pce_inflation"] = (
        df["core_pce"]
        .pct_change(12)
        * 100
    )

    df["shelter_inflation"] = (
        df["shelter"]
        .pct_change(12)
        * 100
    )

    # Demand.
    df["consumption_growth"] = (
        df["real_consumption"]
        .pct_change(12)
        * 100
    )

    # Energy.
    df["oil_yoy"] = (
        df["oil"]
        .pct_change(12)
        * 100
    )

    df["oil_3m"] = (
        df["oil"]
        .pct_change(3)
        * 100
    )

    # Expectations and labour momentum.
    df["expectations_change"] = (
        df["inflation_expectations"]
        .diff(3)
    )

    df["unemployment_change"] = (
        df["unemployment"]
        .diff(3)
    )

    df = df.loc[
        pd.Timestamp(START_DATE):
        pd.Timestamp(DATA_CUTOFF)
    ].copy()

    return df


def latest_status(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for key, cfg in SERIES.items():
        series = data[key].dropna()

        rows.append(
            {
                "Variable": cfg["name"],
                "FRED ID": cfg["id"],
                "Source": cfg["source"],
                "Frequency": cfg["frequency"],
                "Latest": (
                    series.index[-1].strftime("%Y-%m")
                    if not series.empty
                    else "N/A"
                ),
            }
        )

    return pd.DataFrame(rows)
