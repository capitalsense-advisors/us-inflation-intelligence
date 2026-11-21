# US Inflation Intelligence

A focused, India-free U.S. macro research terminal for answering one question:

> **Is U.S. inflation moving sustainably toward the Federal Reserve's 2% objective?**

## Current scope

The first version intentionally keeps the model universe small:

- PCE Price Index: `PCEPI`
- Core PCE Price Index: `PCEPILFE`
- CPI Shelter: `CUSR0000SAH1`
- Unemployment Rate: `UNRATE`
- Real Personal Consumption Expenditures: `DPCERA3M086SBEA`
- University of Michigan 1-Year Inflation Expectation: `MICH`
- WTI Crude Oil: `WTISPLC`

Historical window:

- Start: January 2000
- Data cutoff: latest available observation (override with `FRED_DATA_CUTOFF` for reproducible vintages)

## Daily RiskMode snapshot

`generate_snapshot.py` creates `snapshot/macro-snapshot.json` for the separate
RiskMode Macro Insights screen. The included GitHub Actions workflow checks for
new FRED observations every day at 06:17 UTC, regenerates the model output, and
commits the snapshot only when it changes. A FRED API key is recommended for
vintage-aware production updates; the public FRED CSV endpoint is used as a
zero-cost fallback.

## Run locally

1. Install Python 3.11+.
2. Install dependencies:

```bash
pip install streamlit pandas numpy requests python-dotenv plotly scikit-learn statsmodels xgboost
```

3. Create `.env`:

```text
FRED_API_KEY=YOUR_KEY_HERE
```

4. Run:

```bash
streamlit run app.py
```

## Files

```text
app.py       Streamlit interface
data.py      FRED API + cleaning + feature creation
model.py     Forecasting, validation, intervals, drivers
README.md    Project documentation
```

## Important methodology notes

- PCE inflation is calculated from the PCE price index as 12-month percentage change.
- Time-series evaluation uses expanding/walk-forward validation.
- XGBoost prediction ranges use temporal conformal calibration rather than pretending XGBoost has native confidence intervals.
- SARIMAX is the econometric benchmark.
- The project avoids a large indicator kitchen sink.
- Current regime labels are intentionally simple and will be tested against a GMM alternative before becoming a final production feature.
- The initial multi-step XGBoost forecasting logic is deliberately conservative and is the next modeling milestone. It should be replaced with a dedicated direct/recursive multi-horizon implementation before treating the forecast as production-grade.

## Data provenance

FRED is used as the common access layer. Underlying sources are:

- BEA: PCE, Core PCE, Real Personal Consumption Expenditures
- BLS: CPI Shelter, Unemployment Rate
- EIA / FRED splice: WTI crude oil
- University of Michigan: inflation expectations

## Disclaimer

Research/portfolio tool only. Not an official Federal Reserve forecast and not investment advice.

Thanks!
