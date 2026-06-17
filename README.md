# ⚡ AlphaRegime
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
| XGBoost | **52.4%** | -29.6% | 53.4% | -241.4% |
| Random Forest | 52.2% | -30.3% | **55.0%** | -244.8% |
| LSTM | 49.1% | -38.0% | 51.8% | -103.0% |

> Alpha = strategy cumulative return minus buy-and-hold, out-of-sample (May 2021 – Mar 2026)

**The headline finding:** ARIMA — the simplest model — is the *only* one generating positive alpha on Gold. More complex models achieve marginally higher accuracy but get the *big moves* wrong. Accuracy does not equal profitability.

**ARIMAX vs ARIMA:** Adding VIX, 10-year yield, and USD Index as exogenous variables hurts performance on both assets. Macro features don't carry incremental directional signal beyond past returns at the 1-day horizon.

**Note on Silver: The test period (May 2021 – Mar 2026) coincided with an exceptional Silver bull run (+157% buy-and-hold). Long/short strategies structurally underperform in strong directional trends — negative Silver alpha reflects the market environment, not model failure. Directional accuracy (51–55%) remains consistent across both assets.**
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
├── app.py                               ← Streamlit dashboard
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

## Questions that you might be having, which I have answers for

-**Why did you frame this as classification and not regression?**
Regression predicts the magnitude of the return. Nobody trades magnitude — you trade direction. Even a perfect RMSE score is useless if the model is wrong about up vs down. Classification directly optimises for what matters.

-**Why ARIMA on a classification task?**
ARIMA forecasts the return value. We take the sign of that forecast — positive means predict UP, negative means predict DOWN. It's a clean conversion and gives you a genuine statistical baseline to compare ML models against.

-**Why does ARIMA beat every other model on alpha despite having the lowest accuracy on Gold (51.2%)?**
Because it's right on the days that move the most. XGBoost gets direction correct more often but misfires on high-magnitude days — those losses compound hard in a long/short strategy. ARIMA's mean-reversion signal, despite being simple, happens to align with large moves better. This is the accuracy-profitability divergence — a well documented phenomenon in quantitative finance.

-**Why did ARIMAX underperform ARIMA on both assets?**
Adding VIX, 10-year yield, and USD Index as exogenous variables introduced noise at the 1-day horizon. Macro variables tend to drive multi-day or multi-week trends — at the single day level they don't add directional signal beyond what past returns already carry. ARIMAX overfit to macro fluctuations that don't resolve within 24 hours.

-**Why is the Silver alpha so catastrophically negative for XGBoost (-241%) and Random Forest (-245%)?**
The test period coincided with a +157% Silver bull run. A long/short strategy shorts on DOWN signals — during a sustained uptrend those short positions compound losses every day the market keeps rising. The alpha number isn't purely a model quality measure here, it's partly a market structure problem. Directional accuracy for both models on Silver is 53–55%, which is consistent with Gold — the models aren't broken, the strategy is structurally penalised by the trend.

-**Why did LSTM underperform despite being the most complex model?**
Two reasons. First, the target — next-day log returns — is close to a random walk. LSTM's strength is learning long-range temporal dependencies, but if those dependencies barely exist in daily return series, the model has nothing useful to learn and adds noise. Second, a static 70/30 train split means the LSTM never retrains on recent data, so its learned patterns drift out of relevance as market conditions shift.

-**Why did you use a 70/30 split and not cross-validation?**
Standard k-fold cross-validation shuffles the data, which creates look-ahead bias in time series — you'd be training on future data to predict the past. Time series requires a temporal split. Walk-forward cross-validation would be the ideal alternative, and it was implemented for ARIMA/ARIMAX (refit every 21 days). Applying it to ML and LSTM would be a meaningful extension.

-**Why use GARCH for regime definition instead of COVID dates?**
COVID dates are arbitrary and not replicable on new data — you can't know in advance when a crisis starts. GARCH conditional volatility is computed purely from the return series itself, updates continuously, and has a rigorous statistical foundation in the ARCH effects literature. It's a regime definition that would work on any asset in any time period.

-**All models cluster between 49–55% accuracy. Isn't that basically random?**
It looks that way but 53–55% sustained out-of-sample directional accuracy on a daily horizon is actually commercially meaningful. Most high-frequency trading signals work at 51–52% with high volume. The problem here isn't accuracy — it's that the long/short backtest magnifies errors during trending markets. The signal quality is real, the strategy construction is naive.

-**What would I do differently?**
Three things. First, walk-forward retraining for ML and LSTM, not just ARIMA. Second, a three-class target — Strong Up / Flat / Strong Down with a ±0.5% threshold — to filter low-conviction noise days where no model has an edge. Third, model the short position separately: only go long on UP signals, stay flat on DOWN signals rather than going short. That single change would have dramatically reduced the Silver alpha blowout.

---

## Author
**Vinay Sathish**
---
*Built as a portfolio project exploring regime-aware model comparison in commodity futures markets.*

---

##Disclaimer
**README and all other files are undergoing some modifications. Data is stil March, I've extended the data till June. App works perfectly. Some modifications has to be made. Thanks for understanding**
--

