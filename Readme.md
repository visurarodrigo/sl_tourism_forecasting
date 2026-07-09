# Sri Lanka Tourism Forecasting — End-to-End Data Science Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sltourismforecasting-visura.streamlit.app/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle)](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow?logo=powerbi)](Dashboard/sl_tourism_DB.pbix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Forecasting monthly international tourist arrivals to Sri Lanka (2018–2026) using SARIMA, Prophet, and XGBoost - with an interactive Streamlit app and a Power BI analytics dashboard.**

---

## 🔗 Live Links

| Deliverable | Link |
|---|---|
| 🟢 Streamlit Forecast App | [sltourismforecasting-visura.streamlit.app](https://sltourismforecasting-visura.streamlit.app/) |
| 📊 Power BI Dashboard | [`Dashboard/sl_tourism_DB.pbix`](Dashboard/sl_tourism_DB.pbix) |
| 📦 Kaggle Dataset | [Sri Lanka Tourism Dataset 2018–2026](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may) |

---

## 📌 Project Overview

Sri Lanka's tourism sector is one of its most critical foreign exchange earners, yet it has experienced extreme volatility - the 2019 Easter Sunday attacks, a near-total collapse during COVID-19, and a slow recovery through 2022–2026. This project builds a full time-series forecasting pipeline on real monthly arrival data to model, compare, and visualise this trajectory.

**Three forecasting approaches are compared:**
- **SARIMA** - classical statistical model capturing trend and seasonal structure
- **Prophet** - Meta's decomposable additive model, robust to outliers and structural breaks
- **XGBoost** - gradient boosted trees using engineered lag and calendar features

The best model (SARIMA) achieves a **MAPE of 13.03%** on held-out test data.

---

## 🏆 Model Performance

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **SARIMA** ✅ Best | 29,570 | 24,453 | **13.03%** |
| XGBoost | 34,698 | 31,808 | 15.96% |
| Prophet | 40,686 | 28,501 | 17.68% |

> SARIMA outperforms both ML-based models here - monthly tourism arrivals have strong, consistent seasonality that SARIMA handles natively. XGBoost is competitive with lag features. Prophet underperforms due to the structural break introduced by COVID-19, which is difficult to model even with event flagging.

---

## 📊 Streamlit Dashboard Preview

| Executive Overview | Forecast Explorer |
|---|---|
| ![Executive Overview](Figures/app%20previews/Executive%20Overview.png) | ![Forecast Explorer](Figures/app%20previews/Forecast%20Explorer.png) |

| Model Diagnostics & Performance | Source Market Insights |
|---|---|
| ![Model Diagnostics](Figures/app%20previews/Model%20Diagnostics%20%26%20Performance.png) | ![Source Market Insights](Figures/app%20previews/Source%20Market%20Insights.png) |

**4 pages:**
- **Executive Overview** - KPI cards, total arrivals trend, YoY growth rate
- **Forecast Explorer** - interactive model selector, forecast horizon slider, confidence intervals
- **Model Diagnostics & Performance** - residual plots, error metric comparison table
- **Source Market Insights** - country-wise arrival heatmaps, top 10 source markets over time

---

## 📈 Power BI Dashboard Preview

| Overview | Forecasting |
|---|---|
| ![Page 01 Overview](Figures/powerbi%20previews/Page%2001%20-%20Overview%20.jpg) | ![Page 02 Forecasting](Figures/powerbi%20previews/Page%2002%20-%20Forecasting%20.jpg) |

| Seasonality | Country Analysis |
|---|---|
| ![Page 03 Seasonality](Figures/powerbi%20previews/Page%2003%20-%20Seasonality%20.jpg) | ![Page 04 Country](Figures/powerbi%20previews/Page%2004%20-%20Country.jpg) |

**4 pages:**
- **Overview** - total arrivals KPIs, trend line, year-over-year comparison
- **Forecasting** - SARIMA, Prophet, and XGBoost forecast vs actual overlay, MAPE comparison
- **Seasonality** - monthly seasonal patterns, peak and trough identification
- **Country Analysis** - source market breakdown, top countries by arrival volume

> 📁 Power BI file: [`Dashboard/sl_tourism_DB.pbix`](Dashboard/sl_tourism_DB.pbix)

---

## 📁 Project Structure

```
sl_tourism_forecasting/
│
├── Dashboard/
│   └── sl_tourism_DB.pbix               # Power BI dashboard (4 pages)
│
├── Data/
│   ├── 01_raw/                          # Original SLTDA data
│   │   ├── sl_tourism_arrivals.csv
│   │   ├── sl_tourism_country_wise.csv
│   │   └── Country wise data/           # Annual Excel files 2018–2026
│   ├── 02_processed/                    # Cleaned, normalized datasets
│   └── 03_model_outputs/                # Forecast CSVs, metrics, feature importance
│
├── Figures/
│   ├── app previews/                    # Streamlit dashboard screenshots
│   ├── powerbi previews/                # Power BI dashboard screenshots
│   └── plots/                           # EDA and model diagnostic plots
│
├── src/                                 # Modular pipeline scripts (run sequentially)
│   ├── 01_scrape_data.py
│   ├── 02_process_country_data.py
│   ├── 03_normalize_country_names.py
│   ├── 04_final_cleanup_data.py
│   ├── 05_clean_main_dataset.py
│   ├── 06_stationarity_eda.py
│   ├── 07_differencing_stationarity.py
│   ├── 08_combined_differencing.py
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
- Country-wise data consolidated from annual Excel reports (2018–2026)
- Country names normalized across years (e.g. "United Kingdom" vs "UK")
- Full pipeline scripted across 11 modular Python scripts for reproducibility

### 2. Exploratory Data Analysis
- Trend decomposition revealed a clear pre-pandemic seasonal peak in **December–January** and a secondary peak in **July–August**
- ACF/PACF analysis confirmed significant autocorrelation at lags 1, 12, and 24
- First-order and seasonal differencing applied to achieve stationarity (ADF test confirmed)

![ACF and PACF](Figures/plots/ACF%20and%20PACF%20for%20Sri%20lanka%20Tourism%20Arrivals.png)

![Differencing](Figures/plots/Original%20Data%20%26%201st%20Order%20Dif%20%26%20Seasonal%20Dif.png)

### 3. Modelling

**SARIMA(1,1,1)(1,1,1)[12]**
- Order selected via AIC minimization across a grid search
- Trained on 2018–2023, tested on 2024–2026

![SARIMA Forecast](Figures/plots/SARIMA%20Model%20-%20Training%2C%20Actual%20vs%20Forecast.png)

**Prophet**
- Additive model with yearly seasonality enabled
- COVID-19 period flagged as a structural event to reduce bias

![Prophet Forecast](Figures/plots/Prophet%20Forecast%20-%20Actual%20vs%20Predicted.png)

![Prophet Components](Figures/plots/Prophet%20Forecast%20-%20Component%20Plots%20%28Trend%20%26%20Seasonality%29.png)

**XGBoost**
- Features: lag-1, lag-2, lag-12, rolling mean (3/6/12 months), month, quarter, year
- Trained with time-series cross-validation (no data leakage)

![XGBoost Forecast](Figures/plots/XGBoost%20Actual%20vs%20Forecast.png)

![XGBoost Feature Importance](Figures/plots/XGBoost%20Feature%20Importance.png)

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
# run scripts 02 through generate_forecasts.py sequentially

# 4. Launch the Streamlit dashboard
streamlit run app/app.py
```

For the Power BI dashboard, open `Dashboard/sl_tourism_DB.pbix` in Power BI Desktop.

---

## 📦 Dataset

The cleaned dataset is publicly available on Kaggle:

**👉 [Sri Lanka Tourism Dataset 2018–2026 (May)](https://www.kaggle.com/datasets/visurarodrigo/sri-lanka-tourism-dataset-2018-2026-may)**

Includes:
- `sl_tourism_arrivals_clean.csv` - monthly total arrivals with trend/seasonality columns
- `sl_tourism_country_wise_final.csv` - normalized country-wise monthly arrivals
- `forecast_sarima.csv`, `forecast_prophet.csv`, `forecast_xgboost.csv` - model outputs
- `model_metrics.csv` - RMSE, MAE, MAPE for all 3 models

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
| Streamlit dashboard | Streamlit |
| BI dashboard | Power BI Desktop  |
| Deployment | Streamlit Cloud |
| Dataset hosting | Kaggle |

---

## 👤 Author

**Visura Rodrigo**

[![GitHub](https://img.shields.io/badge/GitHub-visurarodrigo-black?logo=github)](https://github.com/visurarodrigo)
[![Kaggle](https://img.shields.io/badge/Kaggle-visurarodrigo-blue?logo=kaggle)](https://www.kaggle.com/visurarodrigo)