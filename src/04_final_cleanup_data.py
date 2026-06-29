import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_country_wise_clean.csv')
output_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_country_wise_final.csv')

print("Loading normalized data...")
df = pd.read_csv(input_path, parse_dates=['Date'])

print(f"Before final cleanup: {df['Country'].nunique()} unique countries")

# Filter out letter headers and non-country entries
letters = [chr(i) for i in range(65, 91)]  # A-Z
non_countries = ['Antarctica', 'European Union']

df = df[~df['Country'].isin(letters + non_countries)]

print(f"After final cleanup: {df['Country'].nunique()} unique countries")

# Save the final clean dataset
df.to_csv(output_path, index=False)

print(f"\nFinal dataset saved to: {output_path}")
print(f"Total rows: {len(df)}")

# Show updated top 20
print("\nTop 20 Countries (Final):")
top_20 = df.groupby('Country')['Arrivals'].sum().sort_values(ascending=False).head(20)
print(top_20)