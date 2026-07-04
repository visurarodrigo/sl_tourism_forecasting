# Sri Lanka Tourism Forecasting - End-to-End Data Science Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sltourismforecasting-visura.streamlit.app/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Forecasting monthly international tourist arrivals to Sri Lanka (2018–2026) using SARIMA, Prophet, and XGBoost — with an interactive multi-page Streamlit dashboard.**

---

## 🔗 Live Demo

**👉 [sltourismforecasting-visura.streamlit.app](https://sltourismforecasting-visura.streamlit.app/)**

---

## 📌 Project Overview

Sri Lanka's tourism sector is one of its most critical foreign exchange earners, yet it has experienced extreme volatility - from the 2019 Easter attacks, to a complete collapse during COVID-19, to a gradual recovery through 2022–2026. This project builds a full time-series forecasting pipeline on real monthly arrival data (2018–2026) to model and predict this recovery.

**Three forecasting approaches are compared:**
- **SARIMA** — classical statistical model capturing trend and seasonal patterns
- **Prophet** — Facebook's decomposable additive model, robust to outliers and holidays
- **XGBoost** — gradient boosted tree model using engineered lag and calendar features

The best model (SARIMA) achieves a **MAPE of 13.03%** on held-out test data.

---

## 📊 Dashboard Preview

| Executive Overview | Forecast Explorer |
|---|---|
| ![Executive Overview](Figures/app%20previews/Executive%20Overview.png) | ![Forecast Explorer](Figures/app%20previews/Forecast%20Explorer.png) |

| Model Diagnostics & Performance | Source Market Insights |
|---|---|
| ![Model Diagnostics](Figures/app%20previews/Model%20Diagnostics%20%26%20Performance.png) | ![Source Market Insights](Figures/app%20previews/Source%20Market%20Insights.png) |

---

## 🏆 Model Performance

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **SARIMA** ✅ Best | 29,570 | 24,453 | **13.03%** |
| XGBoost | 34,698 | 31,808 | 15.96% |
| Prophet | 40,686 | 28,501 | 17.68% |

SARIMA outperforms both ML-based models on this dataset, which is expected — monthly tourism arrivals exhibit strong, consistent seasonality that SARIMA handles natively. XGBoost performs competitively with engineered lag features. Prophet underperforms likely due to the structural breaks introduced by COVID-19.

---

## 📁 Project Structure

```
sl_tourism_forecasting/
│
├── Data/
│   ├── 01_raw/                          # Original SLTDA data (monthly + country-wise)
│   │   ├── sl_tourism_arrivals.csv
│   │   ├── sl_tourism_country_wise.csv
│   │   └── Country wise data/           # Raw Excel files 2018–2026
│   ├── 02_processed/                    # Cleaned, normalized datasets
│   └── 03_model_outputs/                # Forecast CSVs, metrics, feature importance
│
├── Figures/
│   ├── app previews/                    # Dashboard screenshots
│   └── plots/                           # EDA and model diagnostic plots
│
├── src/                                 # Modular pipeline scripts (numbered sequentially)
│   ├── 01_scrape_data.py
│   ├── 02–05_*.py                       # Data processing and cleaning
│   ├── 06–08_*.py                       # Stationarity analysis and differencing
│   ├── 09_sarima_model.py
│   ├── 10_prophet_model.py
│   ├── 11_xgboost_model.py
│   └── generate_forecasts.py
│
├── app/
│   └── app.py                           # Streamlit multi-page dashboard
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Collection & Cleaning
- Raw monthly arrival data sourced from the **Sri Lanka Tourism Development Authority (SLTDA)**
- Country-wise data scraped and consolidated from annual Excel reports (2018–2026)
- Country names normalized across years (e.g. "United Kingdom" vs "UK")
- Missing months imputed using forward-fill for pandemic months with zero entries

### 2. Exploratory Data Analysis
- Trend decomposition revealed a clear pre-pandemic seasonal peak in **December–January** and a secondary peak in **July–August**
- ACF/PACF analysis confirmed significant autocorrelation at lags 1, 12, and 24
- First-order and seasonal differencing applied to achieve stationarity (ADF test confirmed)

![ACF and PACF](Figures/plots/ACF%20and%20PACF%20for%20Sri%20lanka%20Tourism%20Arrivals.png)

### 3. Modelling

**SARIMA(1,1,1)(1,1,1)[12]**
- Order selected via AIC minimization across a grid search
- Trained on 2018–2023, tested on 2024–2026

![SARIMA Forecast](Figures/plots/SARIMA%20Model%20-%20Training%2C%20Actual%20vs%20Forecast.png)

**Prophet**
- Additive model with yearly seasonality enabled
- COVID-19 period flagged as a holiday/event to reduce bias from structural break

![Prophet Forecast](Figures/plots/Prophet%20Forecast%20-%20Actual%20vs%20Predicted.png)

**XGBoost**
- Features: lag-1, lag-2, lag-12, rolling mean (3/6/12 months), month, quarter, year
- Trained with time-series cross-validation (no data leakage)

![XGBoost Feature Importance](Figures/plots/XGBoost%20Feature%20Importance.png)

### 4. Dashboard (Streamlit)
4-page interactive app:
- **Executive Overview** — KPI cards, total arrivals trend, YoY growth
- **Forecast Explorer** — interactive model selector, forecast horizon slider, confidence intervals
- **Model Diagnostics & Performance** — residual plots, RMSE/MAE/MAPE comparison table
- **Source Market Insights** — country-wise arrival heatmaps, top 10 markets over time

---

## 📦 Dataset

The cleaned dataset is publicly available on Kaggle:

**👉 [Sri Lanka Tourism Dataset 2018–2026 (May)](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may)**

Includes:
- `sl_tourism_arrivals_clean.csv` — monthly total arrivals with trend/seasonality columns
- `sl_tourism_country_wise_final.csv` — normalized country-wise monthly arrivals

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/visurarodrigo/sl_tourism_forecasting.git
cd sl_tourism_forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run the full pipeline
python src/01_scrape_data.py
# ... run scripts 02 through generate_forecasts.py sequentially

# 4. Launch the dashboard
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data processing | Pandas, NumPy |
| Statistical modelling | Statsmodels (SARIMA) |
| ML modelling | XGBoost, Scikit-learn |
| Forecasting | Prophet (Meta) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Dataset hosting | Kaggle |

---

## 👤 Author

**Visura Rodrigo**
3rd Year Undergraduate — Data Science & Business Analytics, KDU Sri Lanka

[![GitHub](https://img.shields.io/badge/GitHub-visurarodrigo-black?logo=github)](https://github.com/visurarodrigo)
[![Kaggle](https://img.shields.io/badge/Kaggle-visurarodrigo-blue?logo=kaggle)](https://www.kaggle.com/visurarodrigo)