import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*60)
print("PHASE 3: DIFFERENCING TO ACHIEVE STATIONARITY")
print("="*60)

print("\nLoading data...")
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

# ==========================================
# 1. FIRST-ORDER DIFFERENCING (Removes Trend)
# ==========================================
print("\nApplying 1st-Order Differencing (Current Month - Previous Month)...")
# diff() calculates y(t) - y(t-1)
df_diff1 = df['Arrivals'].diff().dropna()

print(f"New Shape: {df_diff1.shape}")
print("Running ADF Test on 1st Difference...")
result_d1 = adfuller(df_diff1)

print(f"ADF Statistic: {result_d1[0]:.4f}")
print(f"p-value: {result_d1[1]:.4f}")

if result_d1[1] <= 0.05:
    print("✅ SUCCESS! 1st Difference is STATIONARY.")
    print("   -> We can now use ARIMA(p,1,q)")
else:
    print("❌ Still Non-Stationary. Trying Seasonal Differencing...")
    
    # ==========================================
    # 2. SEASONAL DIFFERENCING (Removes Seasonality)
    # ==========================================
    # We subtract the value from 12 months ago: y(t) - y(t-12)
    print("\nApplying Seasonal Differencing (Current Month - Same Month Last Year)...")
    df_diff_seasonal = df['Arrivals'].diff(12).dropna()
    
    print("Running ADF Test on Seasonal Difference...")
    result_ds = adfuller(df_diff_seasonal)
    
    print(f"ADF Statistic: {result_ds[0]:.4f}")
    print(f"p-value: {result_ds[1]:.4f}")
    
    if result_ds[1] <= 0.05:
        print("✅ SUCCESS! Seasonal Difference is STATIONARY.")
        print("   -> We can now use SARIMA(p,d,q)(P,1,Q)12")
    else:
        print("❌ Still Non-Stationary. We may need BOTH (Seasonal + 1st Order).")

# ==========================================
# 3. VISUALIZING THE DIFFERENCE
# ==========================================
print("\nGenerating Comparison Plots...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Transforming Non-Stationary Data to Stationary', fontsize=16, fontweight='bold')

# Plot 1: Original
axes[0].plot(df.index, df['Arrivals'], color='royalblue')
axes[0].set_title('Original Data (Non-Stationary: Trend + Seasonality)', fontweight='bold')
axes[0].set_ylabel('Arrivals')
axes[0].grid(True, alpha=0.3)

# Plot 2: 1st Difference
axes[1].plot(df_diff1.index, df_diff1, color='darkorange')
axes[1].set_title('1st-Order Difference (Removes Trend)', fontweight='bold')
axes[1].set_ylabel('Change in Arrivals')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color='red', linestyle='--')

# Plot 3: Seasonal Difference
axes[2].plot(df_diff_seasonal.index, df_diff_seasonal, color='green')
axes[2].set_title('Seasonal Difference (Removes Seasonality)', fontweight='bold')
axes[2].set_ylabel('Change vs Last Year')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(0, color='red', linestyle='--')

plt.tight_layout()
plt.show()