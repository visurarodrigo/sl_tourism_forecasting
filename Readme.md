# Raw Data Collection

This directory contains the raw datasets, processed time-series data, and the engineering scripts used to collect them for the Sri Lanka Tourism Forecasting project.

## Directory Structure
- `01_scrape_data.py`: The custom Python script used to scrape and clean the monthly total tourist arrivals.
- `sl_tourism_arrivals_year_wise_2018_2026.csv`: The final, cleaned monthly time-series dataset (Total Arrivals).
- `Country wise data/`: A collection of official Excel files containing monthly tourist arrivals broken down by source country (2018–2026).

## Data Source
All data was sourced from the official **Sri Lanka Tourism Development Authority (SLTDA)**:
- [Monthly Inbound Tourist Arrivals Reports](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-2026)
- [Tourist Arrivals by Country](https://www.sltda.gov.lk/en/tourist-arrivals-from-all-countries)

## Data Engineering & Collection Process
1. **Web Scraping (Total Arrivals):** Because historical monthly totals were not available as a single downloadable file, a custom Python script was written using `requests` and `pandas.read_html` to extract HTML tables directly from the SLTDA website for the years 2018 through 2026.
2. **Data Cleaning:** The raw scraped data required significant cleaning. This included mapping abbreviated month names, handling the 2020 pandemic anomalies (where missing data was recorded as `---` instead of `0`), and formatting the datetime index.
3. **Country-wise Data:** The official statistical Excel reports provided by SLTDA were downloaded and stored for future exploratory analysis and interactive dashboard integration.

## Dataset Summary (Total Arrivals CSV)
- **Frequency:** Monthly
- **Timeframe:** January 2018 – May 2026
- **Target Variable:** `Arrivals` (Total number of tourist arrivals)
- **Key Events Captured:** Pre-pandemic baseline, 2019 Easter Sunday impact, 2020-2021 Pandemic lockdowns, and the 2022-2026 economic recovery.