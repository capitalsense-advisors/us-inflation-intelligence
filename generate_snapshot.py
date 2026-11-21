from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import DATA_CUTOFF, SERIES, fetch_model_data, fetch_series, latest_status
from model import build_forecasts, local_driver_table, model_diagnostics


OUTPUT = Path(__file__).resolve().parent / "snapshot" / "macro-snapshot.json"
TRANSMISSION_SERIES = {
    "real_yield_10y": ("DFII10", "10Y real yield", "%"),
    "treasury_2y": ("DGS2", "2Y Treasury", "%"),
    "breakeven_10y": ("T10YIE", "10Y breakeven", "%"),
    "broad_dollar": ("DTWEXBGS", "Broad US dollar", "index"),
    "fed_balance_sheet": ("WALCL", "Fed balance sheet", "USD millions"),
}


def finite(value, default=None):
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def transmission_snapshot() -> dict:
    output = {}
    for key, (series_id, label, unit) in TRANSMISSION_SERIES.items():
        series = fetch_series(series_id).dropna()
        monthly = series.copy()
        monthly.index = monthly.index.to_period("M").to_timestamp()
        monthly = monthly.groupby(level=0).last()
        latest = finite(monthly.iloc[-1])
        prior = finite(monthly.iloc[-4]) if len(monthly) >= 4 else None
        change_3m = latest - prior if latest is not None and prior is not None else None
        if key in {"broad_dollar", "fed_balance_sheet"} and prior not in (None, 0):
            change_3m = (latest / prior - 1) * 100
        output[key] = {
            "id": series_id,
            "label": label,
            "unit": unit,
            "value": latest,
            "change3m": finite(change_3m),
            "latestObservation": series.index[-1].date().isoformat(),
        }
    return output


def context_label(score: int) -> str:
    if score >= 2:
        return "supportive"
    if score <= -2:
        return "restrictive"
    return "mixed"


def asset_context(transmission: dict) -> dict:
    real_yield = transmission["real_yield_10y"]["change3m"] or 0
    two_year = transmission["treasury_2y"]["change3m"] or 0
    dollar = transmission["broad_dollar"]["change3m"] or 0
    liquidity = transmission["fed_balance_sheet"]["change3m"] or 0
    breakeven = transmission["breakeven_10y"]["change3m"] or 0

    common = (-1 if real_yield > 0.10 else 1 if real_yield < -0.10 else 0) + (-1 if dollar > 1 else 1 if dollar < -1 else 0)
    crypto = common + (-1 if two_year > 0.15 else 1 if two_year < -0.15 else 0) + (1 if liquidity > 1 else -1 if liquidity < -1 else 0)
    gold = common + (1 if breakeven > 0.10 else -1 if breakeven < -0.10 else 0)

    return {
        "gold": {"tone": context_label(gold), "label": "Gold", "score": gold, "summary": "Real yields, the dollar and inflation expectations define the current backdrop."},
        "btc": {"tone": context_label(crypto), "label": "Bitcoin", "score": crypto, "summary": "Real yields, front-end rates, the dollar and liquidity define the current backdrop."},
        "eth": {"tone": context_label(crypto), "label": "Ethereum", "score": crypto, "summary": "Liquidity and risk appetite dominate, usually with higher beta than BTC."},
    }


def balanced_pressure(latest: pd.Series) -> int:
    score = 50
    score += np.clip((finite(latest.get("pce_inflation"), 2) - 2) * 8, -12, 16)
    score += np.clip((finite(latest.get("core_pce_inflation"), 2) - 2) * 6, -10, 14)
    score += np.clip((finite(latest.get("shelter_inflation"), 3) - 3) * 3, -7, 10)
    score += np.clip((finite(latest.get("inflation_expectations"), 2.5) - 2.5) * 4, -6, 8)
    score += np.clip(finite(latest.get("oil_3m"), 0) * 0.15, -7, 7)
    return int(np.clip(round(score), 0, 100))


def main() -> None:
    data = fetch_model_data()
    forecasts = build_forecasts(data, fast_mode=True)
    target = data.dropna(subset=["pce_inflation"])
    latest = target.iloc[-1]
    diagnostics = model_diagnostics(forecasts)
    transmission = transmission_snapshot()
    local_drivers = local_driver_table(forecasts)

    first_month = target.index[-1].to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    curve = []
    for index, (_, row) in enumerate(forecasts["forecast_table"].iterrows()):
        curve.append({
            "date": (first_month + pd.DateOffset(months=index)).date().isoformat(),
            "estimate": finite(row["Forecast"]),
            "lower": finite(row["Lower"]),
            "upper": finite(row["Upper"]),
        })

    performance = diagnostics["performance"]
    weight_map = forecasts["ensemble_weights"]
    audit = []
    for _, row in performance.iterrows():
        name = str(row.get("Model", row.get("model", "Model")))
        audit.append({
            "model": name,
            "weight": finite(weight_map.get(name), 0),
            "mae": finite(row.get("MAE", row.get("mae"))),
        })

    snapshot = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataCutoff": DATA_CUTOFF,
        "latestObservation": target.index[-1].date().isoformat(),
        "status": "ready",
        "source": "FRED · BEA · BLS · EIA · University of Michigan",
        "headline": {
            "pce": finite(latest.get("pce_inflation")),
            "corePce": finite(latest.get("core_pce_inflation")),
            "shelter": finite(latest.get("shelter_inflation")),
            "pressureScore": balanced_pressure(latest),
            "regime": forecasts["regime"],
            "confidence": forecasts["confidence"],
            "forecast3m": finite(forecasts["point_forecast_3m"]),
            "forecast6m": finite(forecasts["point_forecast_6m"]),
        },
        "forecast": curve,
        "forecastMeta": {"selectedModel": forecasts["selected_model"], "brief": forecasts["macro_brief"]},
        "transmission": transmission,
        "assetContext": asset_context(transmission),
        "drivers": [
            {
                "feature": row["Driver"],
                "impact": finite(row["Contribution (pp)"], 0),
                "direction": "up" if finite(row["Contribution (pp)"], 0) >= 0 else "down",
            }
            for _, row in local_drivers.iterrows()
        ],
        "diagnostics": audit,
        "diagnosticMeta": {"coverage": diagnostics["coverage"].to_dict(orient="records")},
        "seriesStatus": latest_status(data).to_dict(orient="records"),
        "disclaimer": "Macro context and model research only. Not an entry signal or investment advice.",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
