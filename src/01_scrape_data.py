import pandas as pd
import requests
import time
import io

all_data = []
years_to_scrape = range(2018, 2027)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for year in years_to_scrape:
    url = f"https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports-{year}"
    print(f"Scraping data for {year}...")
    
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        df = df.iloc[:, [0, 2]] 
        df.columns = ['Month', 'Arrivals']
        
        # Remove 'Total' row
        df = df[~df['Month'].astype(str).str.contains('Total', case=False, na=False)]
        df['Year'] = year
        all_data.append(df)
        time.sleep(2)
    except Exception as e:
        print(f"  -> Error scraping {year}: {e}")

master_df = pd.concat(all_data, ignore_index=True)
print("\n--- Cleaning data... ---")

# FIX 1: Remove the header row that got scraped as data
master_df = master_df[master_df['Month'].astype(str).str.lower().str.strip() != 'month']

month_map = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
    'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10,
    'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}

master_df['Month_Clean'] = master_df['Month'].astype(str).str.lower().str.strip()
master_df['Month_Num'] = master_df['Month_Clean'].map(month_map)

# Drop any rows that still failed to map (safety net)
master_df = master_df.dropna(subset=['Month_Num'])
master_df['Month_Num'] = master_df['Month_Num'].astype(int)

# FIX 2: Create Date using string formatting (avoids Pandas column name strictness)
# This creates strings like "2018-1-01" which pd.to_datetime easily understands
master_df['Date'] = pd.to_datetime(
    master_df['Year'].astype(str) + '-' + master_df['Month_Num'].astype(str) + '-01'
)

# Clean Arrivals (handling the 2020 "---" pandemic anomaly)
master_df['Arrivals'] = master_df['Arrivals'].astype(str)
master_df['Arrivals'] = master_df['Arrivals'].replace(['---', '\\-\\-\\-', 'nan', 'None', '-'], '0')
master_df['Arrivals'] = master_df['Arrivals'].str.replace(',', '').str.replace(' ', '')
master_df['Arrivals'] = pd.to_numeric(master_df['Arrivals'], errors='coerce').fillna(0).astype(int)

# Final formatting
final_df = master_df[['Date', 'Arrivals']].sort_values('Date').reset_index(drop=True)
final_df.to_csv('sl_tourism_arrivals.csv', index=False)

print("\n--- Scraping Complete! ---")
print(f"Total rows: {len(final_df)}")
print("\nFirst 5 rows:")
print(final_df.head())
print("\nLast 5 rows:")
print(final_df.tail())