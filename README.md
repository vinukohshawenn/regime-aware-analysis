# ⚡ Regime Signal Lab
### Regime-Aware Direction Classification for Gold & Silver Futures

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Can any model reliably predict the next-day direction of Gold and Silver returns — and does that signal actually generate alpha?**
> This project answers that question across 6 model classes, 2 regime systems, and 5 years of out-of-sample data.

**[→ Live Dashboard](https://regime-aware-analysis-ekpbaiphwe8kgtn4cnpsot.streamlit.app/)**

---

## The Research Question

Most financial ML projects stop at RMSE. This one doesn't.

Instead of asking *"how accurate is your return forecast?"*, this project asks:
1. **Which direction?** — binary classification: UP or DOWN
2. **In what market environment?** — evaluated separately across low-vol and high-vol regimes defined by GARCH
3. **Does it make money?** — walk-forward backtesting with a long/short strategy

---

## Key Findings

| Model | Gold Acc | Gold Alpha | Silver Acc | Silver Alpha |
|---|:---:|:---:|:---:|:---:|
| ARIMA | 51.2% | **+3.4%** ✅ | 55.2% | +1.5% ✅ |
| ARIMAX | 48.4% | -11.4% | 55.1% | -21.8% |
| XGBoost | **52.4%** | -29.6% | 53.4% | -84.3% |
| Random Forest | 52.2% | -30.3% | **55.0%** | -87.7% |
| LSTM | 49.1% | -38.0% | 51.8% | -103.0% |

> Alpha = strategy cumulative return minus buy-and-hold, out-of-sample (May 2021 – Mar 2026)

**The headline finding:** ARIMA — the simplest model — is the *only* one generating positive alpha on Gold. More complex models achieve marginally higher accuracy but get the *big moves* wrong. Accuracy does not equal profitability.

**ARIMAX vs ARIMA:** Adding VIX, 10-year yield, and USD Index as exogenous variables hurts performance on both assets. Macro features don't carry incremental directional signal beyond past returns at the 1-day horizon.

---

## Architecture

```
gold_silver_revised.csv  (raw)
          │
          ▼
┌──────────────────────┐
│ 01 · EDA &           │  Stationarity tests, ACF/PACF,
│ Hypothesis Testing   │  distributions, correlations
└──────────────────────┘
          │
          ▼
┌──────────────────────┐
│ 02 · GARCH           │  GARCH(1,1) on Au & Ag returns
│ Volatility & Regimes │  → conditional vol → Low/High vol regimes
└──────────────────────┘
          │
          ▼
┌──────────────────────┐
│ 03 · Feature         │  22 features: lags, rolling stats,
│ Engineering          │  GARCH vol, macro, Au/Ag ratio
└──────────────────────┘  Binary target: next-day direction (UP/DOWN)
          │          
    ┌─────┴───────┬─────────┐
    ▼            ▼          ▼
┌────────┐   ┌────────┐  ┌──────┐
│ 04     │   │ 05     │  │ 06   │
│ ARIMA  │   │XGBoost │  │ LSTM │
│ ARIMAX │   │   RF   │  │      │
└────────┘   └────────┘  └──────┘
    │            │          │
    └────────────┴──────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │  07 · Master Comparison  │
            │  Leaderboard · Heatmaps  │
            │  Backtests · Summary     │
            └─────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Streamlit Dashboard   │
            │   5 interactive tabs    │
            └─────────────────────────┘
```

---

## Models

| Notebook | Model | Classification Method |
|---|---|---|
| `04_ARIMA_ARIMAX` | ARIMA(1,0,1) | Sign of walk-forward forecast |
| `04_ARIMA_ARIMAX` | ARIMAX(1,0,1) | Sign of forecast + macro exogenous |
| `05_ML_Classification` | XGBoost Classifier | Probability threshold 0.5 |
| `05_ML_Classification` | Random Forest | Probability threshold 0.5 |
| `06_LSTM_Classification` | LSTM (2-layer, sigmoid) | Sigmoid output threshold 0.5 |

**Walk-forward validation:** ARIMA/ARIMAX refit every 21 trading days on an expanding window. No look-ahead bias. Unified 70/30 train/test split across all models.

---

## Regime System

Two regime definitions applied consistently across all model evaluations:

**GARCH Volatility Regimes** — statistically grounded
- GARCH(1,1) fitted on daily log returns
- Conditional volatility extracted as a time-varying risk measure
- Median split: `low_vol` / `high_vol`

**Period Regimes** — interpretable narrative context
- `pre_covid` — before March 2020
- `covid` — March 2020 to December 2021
- `post_covid` — January 2022 onwards

---

## Feature Set (22 features)

| Group | Features |
|---|---|
| Returns | `au_returns`, `ag_returns` |
| Rolling stats | `au_vol`, `ag_vol`, `au_mean`, `ag_mean` |
| Return lags | `au_lag1`, `au_lag2`, `au_lag5`, `ag_lag1`, `ag_lag2`, `ag_lag5` |
| Macro | `vix`, `interest`, `usd_idx` + daily momentum changes |
| GARCH | `garch_vol_au`, `garch_vol_ag` + 1-day lag each |
| Ratio | `au_ag_ratio_z` — gold-silver ratio, z-scored |

---

## Dashboard — 5 Tabs

| Tab | Content |
|---|---|
| 🏆 Leaderboard | Accuracy, F1, alpha for all models; accuracy bar chart; key metric cards |
| 🌡️ Regime Heatmap | Colour-coded accuracy per model × regime; rolling 63-day accuracy chart |
| 📈 Backtest | Cumulative L/S vs buy-and-hold; regime-split charts; monthly P&L heatmap |
| 🔬 Model Deep-Dive | Confusion matrix; probability distribution; per-regime accuracy breakdown |
| 📊 Price & Signals | Price chart with buy/sell overlays; GARCH vol panel; regime shading |

---

## Data Sources

| Asset | Ticker | Period |
|---|---|---|
| Gold Futures | `GC=F` | Feb 2010 – Mar 2026 |
| Silver Futures | `SI=F` | Feb 2010 – Mar 2026 |
| VIX | `^VIX` | Feb 2010 – Mar 2026 |
| 10-Year Treasury Yield | `^TNX` | Feb 2010 – Mar 2026 |
| US Dollar Index | `DX-Y.NYB` | Feb 2010 – Mar 2026 |

**Train/Test Split:** 70/30 fixed · **Test period:** May 2021 – March 2026 (1,216 observations)

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/regime-signal-lab
cd regime-signal-lab
pip install -r requirements.txt
streamlit run app.py
```

The `data/` folder must contain all prediction CSVs and `gold_silver_garch.csv`.
To regenerate predictions from scratch, run notebooks `01` through `07` in order using Jupyter or Google Colab.

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Select repo · Branch: `main` · Main file path: `app.py`
4. Click **Deploy** — live in ~2 minutes, free permanent URL

---

## Project Structure

```
regime-signal-lab/
│
├── app.py                               ← Streamlit dashboard (~960 lines)
├── requirements.txt
├── README.md
│
├── data/
│   ├── gold_silver_garch.csv            ← Price history + GARCH vol + regime labels
│   ├── preds_arima_au.csv               ← Walk-forward predictions, Gold
│   ├── preds_arima_ag.csv               ← Walk-forward predictions, Silver
│   ├── preds_arimax_au.csv
│   ├── preds_arimax_ag.csv
│   ├── preds_xgb_au.csv
│   ├── preds_xgb_ag.csv
│   ├── preds_rf_au.csv
│   ├── preds_rf_ag.csv
│   ├── preds_lstm_au.csv
│   └── preds_lstm_ag.csv
│
└── notebooks/
    ├── 01_EDA_Hypothesis_Testing.ipynb
    ├── 02_GARCH_Volatility_Regime.ipynb
    ├── 03_Feature_Engineering.ipynb
    ├── 04_ARIMA_ARIMAX_Classification.ipynb
    ├── 05_Logistic_Regression.ipynb
    ├── 06_ML_Classification.ipynb
    ├── 07_LSTM_Classification.ipynb
    └── 08_Master_Comparison.ipynb
```

---

## Tech Stack

`Python 3.10` · `pandas` · `numpy` · `statsmodels` · `arch` · `scikit-learn` · `xgboost` · `tensorflow` · `plotly` · `streamlit`

---

## Limitations & Future Work

- **No transaction costs** — backtest assumes frictionless execution; real deployment would require modelling bid-ask spreads and slippage
- **Static ML split** — ML and LSTM models use a single 70/30 split; proper walk-forward retraining for all models would be more rigorous
- **Binary target** — a 3-class system (Strong Up / Flat / Strong Down) using a ±0.5% threshold could sharpen signals by filtering low-conviction days
- **Single horizon** — only 1-step-ahead; regime effects may be more pronounced at 3-5 day horizons

---

*Built as a portfolio project exploring regime-aware model comparison in commodity futures markets.*
