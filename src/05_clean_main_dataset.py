import pandas as pd
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_path = os.path.join(project_root, 'Data', '01_raw', 'sl_tourism_arrivals.csv')
output_dir = os.path.join(project_root, 'Data', '02_processed')
output_path = os.path.join(output_dir, 'sl_tourism_arrivals_clean.csv')

# Create processed folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("Loading main tourism arrivals data...")
df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')

print(f"Original data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Total rows: {len(df)}")

# ==========================================
# FLAGSHIP MOVE: REMOVE FUTURE ZEROS
# ==========================================
print("\nRemoving future zero values (June-Dec 2026)...")
initial_count = len(df)

# Keep only rows where arrivals are greater than 0
df = df[df['Arrivals'] > 0]

removed_count = initial_count - len(df)
print(f"Removed {removed_count} rows with zero arrivals")
print(f"New data shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# ==========================================
# VERIFY DATA QUALITY
# ==========================================
print("\n--- Data Quality Check ---")
print(f"Any null values: {df['Arrivals'].isnull().any()}")
print(f"Any negative values: {(df['Arrivals'] < 0).any()}")
print(f"Any zero values: {(df['Arrivals'] == 0).any()}")
print(f"Min arrivals: {df['Arrivals'].min()}")
print(f"Max arrivals: {df['Arrivals'].max()}")
print(f"Mean arrivals: {df['Arrivals'].mean():.0f}")

# Show first and last 5 rows
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Last 5 Rows ---")
print(df.tail())

# Save the clean dataset
df.to_csv(output_path)

print(f"\n✓ Clean dataset saved to: {output_path}")
print(f"✓ Ready for time series modeling!")