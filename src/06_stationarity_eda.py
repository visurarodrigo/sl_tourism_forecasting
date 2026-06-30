import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*60)
print("PHASE 2: STATISTICAL EDA & STATIONARITY TESTING")
print("="*60)

print("\nLoading main time series data...")
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

print(f"✓ Data loaded successfully!")
print(f"  - Shape: {df.shape}")
print(f"  - Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"  - Total months: {len(df)}")

# ==========================================
# 1. THE AUGMENTED DICKY-FULLER (ADF) TEST
# ==========================================
print("\n" + "="*60)
print("RUNNING AUGMENTED DICKY-FULLER (ADF) TEST")
print("="*60)
print("\nTesting for stationarity...")

result = adfuller(df['Arrivals'])

print(f'\n📊 ADF Statistic: {result[0]:.4f}')
print(f'📊 p-value: {result[1]:.4f}')
print('\n📊 Critical Values:')
for key, value in result[4].items():
    print(f'   {key}: {value:.4f}')

print("\n" + "-"*60)
if result[1] <= 0.05:
    print("✅ CONCLUSION: Data is STATIONARY")
    print("   - p-value <= 0.05")
    print("   - We can reject the null hypothesis")
    print("   - Data has constant mean and variance over time")
else:
    print("❌ CONCLUSION: Data is NON-STATIONARY")
    print("   - p-value > 0.05")
    print("   - We cannot reject the null hypothesis")
    print("   - Data has trend and/or seasonality")
    print("   - We MUST apply differencing (the 'I' in ARIMA)")
print("-"*60)

# ==========================================
# 2. ACF AND PACF PLOTS
# ==========================================
print("\nGenerating ACF and PACF plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ACF and PACF for Sri Lanka Tourism Arrivals', fontsize=16, fontweight='bold')

# ACF (Autocorrelation Function)
acf_vals = acf(df['Arrivals'], nlags=40)
axes[0].plot(acf_vals, color='royalblue', marker='o', markersize=4)
axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Lag (Months)', fontsize=10)
axes[0].set_ylabel('Correlation', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Highlight seasonal lags (12, 24, 36)
for lag in [12, 24, 36]:
    if lag < len(acf_vals):
        axes[0].axvline(x=lag, color='red', linestyle='--', alpha=0.5)
        axes[0].text(lag, max(acf_vals)*0.9, f'Lag {lag}', 
                    rotation=90, color='red', fontsize=8, ha='right')

# PACF (Partial Autocorrelation Function)
pacf_vals = pacf(df['Arrivals'], nlags=40)
axes[1].plot(pacf_vals, color='darkorange', marker='o', markersize=4)
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Lag (Months)', fontsize=10)
axes[1].set_ylabel('Correlation', fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Highlight seasonal lags
for lag in [12, 24, 36]:
    if lag < len(pacf_vals):
        axes[1].axvline(x=lag, color='red', linestyle='--', alpha=0.5)
        axes[1].text(lag, max(pacf_vals)*0.9, f'Lag {lag}', 
                    rotation=90, color='red', fontsize=8, ha='right')

plt.tight_layout()
plt.show()

# ==========================================
# 3. INTERPRETATION GUIDE
# ==========================================
print("\n" + "="*60)
print("INTERPRETING THE RESULTS")
print("="*60)

print("\n ACF Chart (Left):")
print("   - Look for tall spikes at Lag 12, 24, 36...")
print("   - These prove STRONG YEARLY SEASONALITY")
print("   - This tells us we need SARIMA (Seasonal ARIMA)")
print("   - Not just regular ARIMA!")

print("\n PACF Chart (Right):")
print("   - Shows 'direct' correlation between months")
print("   - Helps us choose the 'p' parameter in ARIMA")
print("   - Look for where it cuts off or tails off")

print("\n Next Steps:")
if result[1] > 0.05:
    print("   1. Apply first-order differencing to remove trend")
    print("   2. Apply seasonal differencing to remove seasonality")
    print("   3. Re-test for stationarity")
    print("   4. Then proceed to ARIMA parameter tuning (p,d,q)")

print("\n" + "="*60)
print("Phase 2 Complete! Ready for modeling.")
print("="*60)