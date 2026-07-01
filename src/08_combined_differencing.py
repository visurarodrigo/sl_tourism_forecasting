import pandas as pd
from statsmodels.tsa.stattools import adfuller
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*60)
print("COMBINED DIFFERENCING: Trend + Seasonality Removal")
print("="*60)

df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

# Apply BOTH: 1st difference THEN seasonal difference
# This is what SARIMA does internally with parameters (p,1,q)(P,1,Q)12
df_diff1 = df['Arrivals'].diff()  # Remove trend
df_diff_combined = df_diff1.diff(12)  # Remove seasonality
df_diff_combined = df_diff_combined.dropna()

print(f"\nApplied: 1st-Order + Seasonal Differencing")
print(f"Shape after combined differencing: {df_diff_combined.shape}")

result = adfuller(df_diff_combined)
print(f"\nADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")

if result[1] <= 0.05:
    print("\n✅ SUCCESS! Data is now STATIONARY!")
    print("   -> We can proceed with SARIMA(p,1,q)(P,1,Q)12")
    print("   -> d=1 (1st order differencing)")
    print("   -> D=1 (seasonal differencing)")
else:
    print("\n❌ Still non-stationary (unexpected)")

