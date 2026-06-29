import pandas as pd
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_path = os.path.join(project_root, 'Data', '01_raw', 'sl_tourism_country_wise.csv')
output_dir = os.path.join(project_root, 'Data', '02_processed')
output_path = os.path.join(output_dir, 'sl_tourism_country_wise_clean.csv')

# Create the processed folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("Loading country-wise data...")
df = pd.read_csv(input_path, parse_dates=['Date'])

# ==========================================
# FLAGSHIP MOVE: ENTITY RESOLUTION
# ==========================================
# Map known variations to a single standard name
name_corrections = {
    'India, Republic Of': 'India',
    'Iran, Islamic Republic Of': 'Iran',
    'Korea, Republic Of': 'South Korea',
    'Democratic Republic Of Congo': 'Democratic Republic of Congo',
    'Republic Of Congo': 'Republic of Congo',
    'Viet Nam': 'Vietnam',
    'Lao, People S Democratic Republic': 'Laos',
    'Tanzania, United Republic Of': 'Tanzania'
}

print("Normalizing country names...")
df['Country'] = df['Country'].replace(name_corrections)

# Save the clean dataset
df.to_csv(output_path, index=False)

print(f"\nNormalized data saved to: {output_path}")
print("\nTop 10 Countries (Normalized):")
top_10 = df.groupby('Country')['Arrivals'].sum().sort_values(ascending=False).head(10)
print(top_10)