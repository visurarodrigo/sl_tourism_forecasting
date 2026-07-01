import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Bulletproof path finding
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'Data', '02_processed', 'sl_tourism_arrivals_clean.csv')

print("="*70)
print("PHASE 4: BUILDING SARIMA MODEL")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
print(f"✓ Data loaded: {df.shape[0]} months")
print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

# ==========================================
# 1. TRAIN/TEST SPLIT (Time Series Style)
# ==========================================
# For time series, we CANNOT randomly split! We must use chronological order.
# We'll use the last 12 months (1 year) as our test set
print("\nSplitting data chronologically...")
test_size = 12  # Last 12 months for testing
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

print(f"✓ Training set: {train.shape[0]} months ({train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')})")
print(f"✓ Test set: {test.shape[0]} months ({test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')})")

# ==========================================
# 2. BUILD SARIMA MODEL
# ==========================================
# Based on our analysis: d=1, D=1, seasonal_period=12
# Let's start with reasonable values for p, q, P, Q
# We'll use (1,1,1)(1,1,1,12) as a starting point
print("\nBuilding SARIMA model...")
print("  Parameters: (p=1, d=1, q=1)(P=1, D=1, Q=1, s=12)")

model = SARIMAX(
    train['Arrivals'],
    order=(1, 1, 1),           # (p, d, q) - non-seasonal
    seasonal_order=(1, 1, 1, 12),  # (P, D, Q, s) - seasonal
    enforce_stationarity=False,
    enforce_invertibility=False
)

print("\nFitting model to training data...")
results = model.fit(disp=False, maxiter=200)

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
print(results.summary())

# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================
print("\n" + "="*70)
print("GENERATING FORECASTS")
print("="*70)

# Forecast on test set
print("\nForecasting on test data...")
forecast = results.get_forecast(steps=test_size)
predicted_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)  # 95% confidence interval

# ==========================================
# 4. CALCULATE ERROR METRICS
# ==========================================
print("\n" + "="*70)
print("MODEL PERFORMANCE METRICS")
print("="*70)

actual = test['Arrivals'].values
predicted = predicted_mean.values

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"\n✓ RMSE (Root Mean Squared Error): {rmse:,.2f}")
print(f"  → Lower is better. Measures average error magnitude.")
print(f"\n✓ MAE (Mean Absolute Error): {mae:,.2f}")
print(f"  → Lower is better. Average absolute difference.")
print(f"\n✓ MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"  → Lower is better. Percentage error.")
print(f"  → <10% = Excellent, 10-20% = Good, 20-50% = Acceptable")

# ==========================================
# 5. VISUALIZE RESULTS
# ==========================================
print("\nGenerating visualization...")

fig, ax = plt.subplots(figsize=(14, 7))

# Plot training data
ax.plot(train.index, train['Arrivals'], label='Training Data', color='royalblue', linewidth=2)

# Plot test data (actual)
ax.plot(test.index, test['Arrivals'], label='Actual (Test)', color='green', linewidth=2, marker='o')

# Plot predictions
ax.plot(test.index, predicted_mean, label='SARIMA Forecast', color='red', linewidth=2, marker='x')

# Plot confidence intervals
ax.fill_between(
    test.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='red',
    alpha=0.1,
    label='95% Confidence Interval'
)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Tourist Arrivals', fontsize=12, fontweight='bold')
ax.set_title('SARIMA Model: Training, Actual vs Forecast', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(x=train.index[-1], color='black', linestyle='--', linewidth=1, label='Train/Test Split')

plt.tight_layout()
plt.show()

# ==========================================
# 6. SAVE RESULTS
# ==========================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create results dataframe
results_df = pd.DataFrame({
    'Date': test.index,
    'Actual': actual,
    'Predicted': predicted,
    'Error': actual - predicted,
    'Error_Percentage': ((actual - predicted) / actual) * 100
})

# Save to CSV
output_path = os.path.join(project_root, 'Data', '02_processed', 'sarima_forecast_results.csv')
results_df.to_csv(output_path, index=False)

print(f"\n✓ Forecast results saved to: {output_path}")

# Show sample predictions
print("\nSample Predictions (First 5 months):")
print(results_df.head())

print("\n" + "="*70)
print("SARIMA MODEL COMPLETE!")
print("="*70)
print(f"\nModel Performance:")
print(f"  - RMSE: {rmse:,.2f}")
print(f"  - MAE: {mae:,.2f}")
print(f"  - MAPE: {mape:.2f}%")
print(f"\nNext Steps:")
print(f"  1. Tune hyperparameters (p, q, P, Q) to improve performance")
print(f"  2. Build Prophet model for comparison")
print(f"  3. Build XGBoost model for comparison")
print(f"  4. Compare all three models") 
