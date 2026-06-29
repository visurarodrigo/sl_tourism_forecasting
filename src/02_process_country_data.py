import pandas as pd
import os
import glob

# ==========================================
# 1. BULLETPROOF PATH FINDING
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'Data', '01_raw', 'Country wise data')

print(f"--- Path Debug ---")
print(f"Looking for data in: {data_dir}")

if not os.path.exists(data_dir):
    print(f"ERROR: The folder '{data_dir}' does not exist!")
    exit()

files = glob.glob(os.path.join(data_dir, '*.xlsx'))
print(f"Found {len(files)} Excel files to process.\n")

if not files:
    print("No .xlsx files found in the folder. Exiting.")
    exit()

# ==========================================
# 2. DATA PROCESSING LOOP
# ==========================================
all_dfs = []

month_map = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'september': 9, 'sept': 9, 'oct': 10,
    'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

fragments_to_merge = ['REPUBLIC', 'GUINEA', 'ZUID AFRIKA', 'DEM REPOF THE', 
                      'ARAB JAMAHIR', 'NORTHERNRHODESIA', 'PEOPLES DEM.REP.', 
                      'SLOVAK REPUBLIC', 'YEMEN ARAB REP.']

for file in files:
    year = int(os.path.basename(file).split('.')[0])
    print(f"Processing {year}...")
    
    # --- PHASE A: DYNAMIC HEADER DETECTION ---
    df_raw = pd.read_excel(file, header=None)
    
    def is_exact_country(val):
        return str(val).strip().lower() == 'country'
        
    header_rows = df_raw[df_raw.apply(lambda row: row.apply(is_exact_country).any(), axis=1)]
    
    if header_rows.empty:
        print(f"  -> WARNING: Could not find exact 'Country' header in {year}. Skipping.")
        continue
        
    header_row_idx = header_rows.index[0]
    df = pd.read_excel(file, header=header_row_idx)
    
    # --- PHASE B: CLEAN COLUMN NAMES (THE FIX) ---
    # Strip ALL whitespace from column names to fix hidden spaces like "January "
    df.columns = [str(col).strip() for col in df.columns]
    
    # Find the column that contains 'country' and rename it to 'Country'
    country_col = None
    for col in df.columns:
        if 'country' in str(col).lower():
            country_col = col
            break
            
    if country_col is None:
        print(f"  -> WARNING: Could not find 'Country' column in {year}. Skipping.")
        continue
        
    df.rename(columns={country_col: 'Country'}, inplace=True)
    
    # Filter out the 'Total' row at the bottom and any empty rows
    df = df[df['Country'].notna() & ~df['Country'].astype(str).str.contains('Total', case=False)]
    df = df.reset_index(drop=True)
    
    # --- PHASE C: FIX THE "SPLIT COUNTRY" TRAP ---
    for i in range(1, len(df)):
        curr_country = str(df.loc[i, 'Country']).upper().strip()
        if curr_country in fragments_to_merge:
            prev_country = str(df.loc[i-1, 'Country']).strip()
            df.loc[i-1, 'Country'] = f"{prev_country} {curr_country}"
            df.loc[i, 'Country'] = 'DROP_ME'
            
    df = df[df['Country'] != 'DROP_ME'].copy()
    
    # --- PHASE D: IDENTIFY & CLEAN MONTH COLUMNS ---
    month_cols = []
    for col in df.columns:
        # Strip whitespace again just in case, then check against map
        col_clean = str(col).lower().strip()
        if col_clean in month_map:
            month_cols.append((col, month_map[col_clean]))
            
    month_cols.sort(key=lambda x: x[1])
    
    if not month_cols:
        print(f"  -> WARNING: No valid month columns found in {year}. Skipping.")
        continue
        
    cols_to_keep = ['Country'] + [c[0] for c in month_cols]
    df_clean = df[cols_to_keep].copy()
    
    # Clean Country names (Title Case)
    df_clean['Country'] = df_clean['Country'].astype(str).str.strip().str.title()
    
    # Clean the numerical data
    for col, _ in month_cols:
        df_clean[col] = df_clean[col].astype(str).str.replace(',', '').str.strip()
        df_clean[col] = df_clean[col].replace(['', 'nan', 'None', '---', '-', ' '], '0')
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
    # --- PHASE E: MELT TO LONG FORMAT ---
    id_vars = ['Country']
    value_vars = [c[0] for c in month_cols]
    df_melted = df_clean.melt(id_vars=id_vars, value_vars=value_vars, var_name='Month_Col', value_name='Arrivals')
    
    col_to_month = {c[0]: c[1] for c in month_cols}
    df_melted['Month'] = df_melted['Month_Col'].map(col_to_month)
    df_melted['Year'] = year
    
    df_melted['Date'] = pd.to_datetime(df_melted['Year'].astype(str) + '-' + df_melted['Month'].astype(str) + '-01')
    df_final = df_melted[['Date', 'Country', 'Arrivals']]
    
    all_dfs.append(df_final)
    print(f"  -> Successfully processed {year} ({len(df_final)} rows)")

# ==========================================
# 3. COMBINE AND SAVE
# ==========================================
if not all_dfs:
    print("\nERROR: No data was successfully processed. Cannot create master file.")
    exit()

master_df = pd.concat(all_dfs, ignore_index=True)

output_path = os.path.join(project_root, 'Data', '01_raw', 'sl_tourism_country_wise.csv')
master_df.to_csv(output_path, index=False)

print(f"\n--- Processing Complete! ---")
print(f"Total rows: {len(master_df)}")
print(f"Saved to: {output_path}")

print("\nTop 10 Countries by Total Arrivals (2018-2026):")
top_countries = master_df.groupby('Country')['Arrivals'].sum().sort_values(ascending=False).head(10)
print(top_countries)